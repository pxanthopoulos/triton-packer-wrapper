#include <algorithm>
#include <cstdlib>
#include <fstream>
#include <iostream>
#include <string>
#include <map>
#include <sstream>
#include <vector>
#include <limits>
#include <atomic>
#include <chrono>

#include "llvm/ADT/MapVector.h"
#include "mlir/Support/LLVM.h"
#include "mlir/Analysis/Liveness.h"
#include "mlir/Analysis/SliceAnalysis.h"
#include "llvm/ADT/SmallVector.h"

namespace mlir
{

    template <typename T>
    class Interval
    {
    public:
        Interval() = default;

        Interval(T S, T E) : Start(S), End(E) { assert(Start <= End); }

        [[nodiscard]] T start() const { return Start; }

        [[nodiscard]] T end() const { return End; }

        [[nodiscard]] T size() const { return End - Start; }

        [[nodiscard]] bool contains(T Addr) const { return Start <= Addr && Addr < End; }

        [[nodiscard]] bool intersects(const Interval &R) const
        {
            return Start < R.End && R.Start < End;
        }

        bool operator==(const Interval &R) const
        {
            return Start == R.Start && End == R.End;
        }

        bool operator!=(const Interval &R) const { return *this != R; }

        bool operator<(const Interval &R) const
        {
            return std::make_pair(Start, End) < std::make_pair(R.Start, R.End);
        }

    private:
        T Start = std::numeric_limits<T>::min();
        T End = std::numeric_limits<T>::max();
    };

    template <class T>
    Interval(T, T) -> Interval<T>;

    using BufferId = size_t;

    struct BufferT
    {
        /// Explicit: triton_gpu.alloc_tensor
        /// Scratch: triton_gpu.convert_layout
        /// Virtual: triton.call
        enum class BufferKind
        {
            Explicit,
            Scratch,
            Virtual
        };

        /// MT: thread-safe
        inline static std::atomic<BufferId> nextId = 0;

        BufferKind kind;
        BufferId id;
        size_t size;
        size_t alignment;
        size_t offset;

        bool operator==(const BufferT &other) const { return id == other.id; }

        bool operator<(const BufferT &other) const { return id < other.id; }

        BufferT() : BufferT(BufferKind::Explicit, 0) {}

        BufferT(BufferKind kind, size_t size, size_t alignment = 4,
                size_t offset = 0)
            : kind(kind), id(nextId++), size(size), alignment(alignment),
              offset(offset) {}
    };

    using BufferRangeMapT = llvm::MapVector<BufferT *, Interval<size_t>>;

    using GraphT = DenseMap<BufferT *, DenseSet<BufferT *>>;

    BufferRangeMapT bufferRange;

    void fillBufferRange(const std::string &filepath)
    {
        std::ifstream file(filepath);
        if (!file.is_open())
        {
            std::cerr << "Error: Unable to open file " << filepath << std::endl;
            exit(1);
        }

        std::string header;
        getline(file, header);

        std::string line;
        while (getline(file, line))
        {
            std::stringstream ss(line);
            std::string value;

            // skip id
            getline(ss, value, ',');

            getline(ss, value, ',');
            auto lower = static_cast<size_t>(stoi(value));

            getline(ss, value, ',');
            auto upper = static_cast<size_t>(stoi(value));

            getline(ss, value, ',');
            auto size = static_cast<size_t>(stoi(value));

            Interval<size_t> interval = Interval<size_t>(lower, upper);
            auto *buffer = new BufferT(BufferT::BufferKind::Explicit, size, 1, 0);

            bufferRange[buffer] = interval;
        }
        file.close();
    }

    void allocate(llvm::SmallVector<BufferT *> &buffers,
                  GraphT &interference,
                  DenseMap<BufferT *, size_t> &bufferStart)
    {
        // Reset shared memory size
        // First-fit graph coloring
        // Neighbors are nodes that interfere with each other.
        // We color a node by finding the index of the first available
        // non-neighboring node or the first neighboring node without any color.
        // Nodes with the same color do not interfere with each other.
        DenseMap<BufferT *, int> colors;
        for (auto value : buffers)
        {
            colors[value] = (value == buffers[0]) ? 0 : -1;
        }
        llvm::SmallVector<bool> available(buffers.size());
        for (auto x : buffers)
        {
            std::fill(available.begin(), available.end(), true);
            for (auto y : interference.lookup(x))
            {
                int color = colors[y];
                if (color >= 0)
                {
                    available[color] = false;
                }
            }
            auto it = std::find(available.begin(), available.end(), true);
            colors[x] = std::distance(available.begin(), it);
        }
        // Finalize allocation
        // color0: [0, 7), [0, 8), [0, 15) -> [0, 7), [0, 8), [0, 15)
        // color1: [7, 9) -> [0 + 1 * 15, 9 + 1 * 15) -> [15, 24)
        // color2: [8, 12) -> [8 + 2 * 15, 12 + 2 * 15) -> [38, 42)
        // TODO(Keren): We are wasting memory here.
        // Nodes with color2 can actually start with 24.
        for (auto x : buffers)
        {
            size_t adj = 0;
            for (auto y : interference.lookup(x))
            {
                adj = std::max(adj, bufferStart.lookup(y) + y->size);
            }
            x->offset = bufferStart.lookup(x) + colors.lookup(x) * adj;
            bufferStart[x] = x->offset;
        }
    }

    void calculateStarts(llvm::SmallVector<BufferT *> &buffers,
                         DenseMap<BufferT *, size_t> &bufferStart)
    {
        //  v = values in shared memory
        //  t = triplet of (size, start, end)
        //  shared memory space
        //  -
        //  |         *******t4
        //  | /|\ v2 inserts t4, t5, and t6
        //  |  |
        //  | ******t5         ************t6
        //  | ^^^^^v2^^^^^^
        //  |  |      *********************t2
        //  | \|/ v2 erases t1
        //  | ******t1 ^^^^^^^^^v1^^^^^^^^^ ************t3
        //  |---------------------------------------------| liveness range
        //    1 2 3 4 5 6 7 8 9 10 11 12 13 14 15 16 17 ...
        // If the available triple's range is less than a given buffer range,
        // we won't know if there has been an overlap without using graph coloring.
        // Start -> Liveness Range
        using TripleMapT = std::multimap<size_t, Interval<size_t>>;
        TripleMapT tripleMap;
        tripleMap.insert(std::make_pair(0, Interval<size_t>()));
        SmallVector<BufferT *> xBuffers = buffers;
        while (!xBuffers.empty())
        {
            auto tripleIt = tripleMap.begin();
            auto size = tripleIt->first;
            auto range = tripleIt->second;
            tripleMap.erase(tripleIt);
            auto bufferIt =
                std::find_if(xBuffers.begin(), xBuffers.end(), [&](auto *buffer)
                             {
                        auto xRange = bufferRange[buffer];
                        bool res = xRange.intersects(range);
                        for (auto val: tripleMap)
                            res = res &&
                                  !val.second.intersects(xRange); // only one buffer intersect
                        return res; });
            if (bufferIt != xBuffers.end())
            {
                auto buffer = *bufferIt;
                auto xSize = buffer->size;
                auto xRange = bufferRange.lookup(buffer);
                // TODO(Keren): A buffer's size shouldn't be determined here, have to
                // clean it up
                size_t alignment = buffer->alignment;
                size_t alignSize = ((size + alignment - 1) / alignment) * alignment;
                bufferStart[buffer] = alignSize;
                tripleMap.insert({alignSize + xSize,
                                  Interval{std::max(range.start(), xRange.start()),
                                           std::min(range.end(), xRange.end())}});
                // We could either insert (range.start, xRange.start) or (range.start,
                // xRange.end), both are correct and determine the potential buffer
                // offset, and the graph coloring algorithm will solve the interference,
                // if any
                if (range.start() < xRange.start())
                    tripleMap.insert({size, Interval{range.start(), xRange.end()}});
                if (xRange.end() < range.end())
                    tripleMap.insert({size, Interval{xRange.start(), range.end()}});
                xBuffers.erase(bufferIt);
            }
        }
    }

    void buildInterferenceGraph(llvm::SmallVector<BufferT *> &buffers,
                                DenseMap<BufferT *, size_t> &bufferStart,
                                GraphT &interference)
    {
        // Reset interference graph
        interference.clear();
        for (auto x : buffers)
        {
            for (auto y : buffers)
            {
                if (x == y)
                    continue;
                auto xStart = bufferStart.lookup(x);
                auto yStart = bufferStart.lookup(y);
                auto xSize = x->size;
                auto ySize = y->size;
                Interval xSizeRange = {xStart, xStart + xSize};
                Interval ySizeRange = {yStart, yStart + ySize};
                auto xOpRange = bufferRange.lookup(x);
                auto yOpRange = bufferRange.lookup(y);
                if (xOpRange.intersects(yOpRange) &&
                    xSizeRange.intersects(ySizeRange))
                {
                    interference[x].insert(y);
                }
            }
        }
    }

    void computeOffsets()
    {
        const char *path = std::getenv("BASE_PATH");
        const char *name = std::getenv("TRACE_NAME");

        std::string path_string = std::string(path);

        if (!(path && name))
        {
            std::cerr << "ERROR: One or more environment variables not set!" << std::endl;
            exit(1);
        }

        auto start = std::chrono::high_resolution_clock::now();

        SmallVector<BufferT *> buffers;
        for (auto bufferIter : bufferRange)
        {
            buffers.emplace_back(bufferIter.first);
        }

        DenseMap<BufferT *, size_t> bufferStart;
        calculateStarts(buffers, bufferStart);

        // NOTE: The original paper doesn't consider interference between
        // the bumped ranges. Buffers that previously do not interfere with
        // could interfere after offset bumping if their liveness ranges overlap.
        // Therefore, we rerun the interference graph algorithm after bumping so
        // that we regroup the buffers and color them again. Since we always
        // increase the buffer offset and keep reducing conflicts, we will
        // eventually reach a fixed point.
        GraphT interference;
        buildInterferenceGraph(buffers, bufferStart, interference);
        do
        {
            allocate(buffers, interference, bufferStart);
            buildInterferenceGraph(buffers, bufferStart, interference);
        } while (!interference.empty());

        auto end = std::chrono::high_resolution_clock::now();
        auto duration = std::chrono::duration_cast<std::chrono::microseconds>(end - start);
        std::cout << duration.count();

        std::string filename = std::string(name) + "-out.csv";
        std::string new_path =
            path_string + "/csv-out/";
        std::ofstream outfile(new_path + filename, std::ios::trunc);

        if (outfile.is_open())
        {
            outfile << "id,lower,upper,size,offset" << std::endl;
            for (auto bufferIter : bufferRange)
            {
                outfile << bufferIter.first->id << "," << bufferIter.second.start()
                        << "," << bufferIter.second.end() << ","
                        << bufferIter.first->size << ","
                        << bufferStart.lookup(bufferIter.first) << std::endl;
            }
            outfile.close();
        }
        else
        {
            std::cout << "Could not open file: " << new_path << filename
                      << std::endl;
            exit(1);
        }
    }
}

int main(int argc, char **argv)
{

    if (argc != 2)
    {
        std::cerr << "Usage: " << argv[0] << " <absolute_path_to_csv_file>" << std::endl;
        return 1;
    }

    std::string filepath = argv[1];
    mlir::fillBufferRange(filepath);

    mlir::computeOffsets();
    return 0;
}
