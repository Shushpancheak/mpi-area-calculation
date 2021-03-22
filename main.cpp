#include <mpi.h>
#include <omp.h>

#include <iomanip>
#include <iostream>
#include <cassert>
#include <cstdlib>

typedef long double ldouble;

constexpr ldouble EPS = 1e-10;
constexpr size_t  THREADS_PROCS_COUNT = 16; // Number threads times number of procs.

ldouble Abs(ldouble a) {
    return a < 0 ? -a : a;
}

// @return The value of Function considered at x.
ldouble Function(ldouble x) {
    return 4 / (1 + x * x);
}

// @return Area underneath Function trapezoid between x1 and x2.
ldouble CalcArea(ldouble x1, ldouble x2) {
    return (Function(x1) + Function(x2)) / 2 * (x2 - x1);
}

// @return Sum of areas underneath given parts
ldouble CalcParts(ldouble left, ldouble delta, size_t parts) {
    ldouble sum = 0;
    for (size_t i = 0; i < parts; ++i, left += delta) {
        sum += CalcArea(left, left + delta);
    }
    return sum;
}

// Works just as CalcParts except for the fact that it uses
// omp for parallelism
ldouble CalcPartsParallel(ldouble delta, size_t from, size_t to, size_t threads) {
    ldouble sum = 0;
#pragma omp parallel for reduction (+:sum) num_threads(threads)
    for (size_t i = from; i < to; ++i) {
        sum += CalcArea(delta * i, delta * (i + 1));
    }
    return sum;
}

class MpiTimer {
public:
    MpiTimer() { start_ = MPI_Wtime(); }
    ~MpiTimer() = default;

    ldouble GetTimeElapsed() { return MPI_Wtime() - start_; }

private:
    ldouble start_;
};

struct CalcRes {
    ldouble area;
    ldouble time;
};

// Prints only if it is the master proc.
void Print(int world_rank, const char* msg) {
    if (world_rank) {
        return;
    }

    std::cout << msg << std::endl;
}

int main(int argc, char** argv) {
    MPI_Init(&argc, &argv);

    CalcRes res[3] {}; // Three results from area calculation:
                       // 1) Done by the main process.
                       // 2) Done by multiple processes.
                       // 3) Done by multiple processes with multiple threads.

    size_t parts = strtoull(argv[1], nullptr, 10);

    int world_rank = 0;
    int world_size = 0;
    MPI_Comm_rank(MPI_COMM_WORLD, &world_rank);
    MPI_Comm_size(MPI_COMM_WORLD, &world_size);


    size_t this_parts = parts / world_size;         // Number of parts for this process.
    size_t from       = world_rank * this_parts;    // Start part for this process.
    size_t to         = from + this_parts;          // End
    if (world_rank == world_size - 1) {
        to = parts;
        this_parts += parts % world_size;
    }
    ldouble delta = 1. / parts;
    ldouble left = from * delta;

    Print(world_rank, "\nStarting calculation for a single process...");

    // Stage 1: The main process performs the task by itself.
    if (world_rank == 0) {
        MpiTimer timer;
        res[0].area = CalcParts(left, delta, parts);
        res[0].time = timer.GetTimeElapsed();

        std::cout << "Area as calculated by a single proc: " << res[0].area << std::endl;
        std::cout << "Time taken: " << res[0].time << std::endl;
    }


    // Stage 2: Multiple processes.
    // Waiting for the main proc to end its task.
    MPI_Barrier(MPI_COMM_WORLD);

    Print(world_rank, "\nStarting calculation for multiple processes...");
    MpiTimer timer_1;
    ldouble this_area = CalcParts(left, delta, this_parts); // Calculate area for this proc.

    if (world_rank != 0) { // If it is a slave proc, send the calculated area.
        MPI_Send(&this_area, 1, MPI_LONG_DOUBLE, 0, 0, MPI_COMM_WORLD);
    } else {
        res[1].area = this_area;
        std::cout << "Result from process 0: " << this_area << "\n";

        ldouble cur_area = 0;
        for (int i = 1; i < world_size; ++i) {
            MPI_Recv(&cur_area, 1, MPI_LONG_DOUBLE, i, 0, MPI_COMM_WORLD, MPI_STATUSES_IGNORE);
            res[1].area += cur_area;
            std::cout << "Result from process " << i << ": ";
            std::cout << cur_area << "\n";
        }

        res[1].time = timer_1.GetTimeElapsed();

        std::cout << "Overall result for multiple processes: " << res[1].area << std::endl;
        assert(Abs(res[0].area - res[1].area) < EPS);
        std::cout << "Time taken: " << res[1].time << std::endl;
        std::cout << "Acceleration: " << std::endl << res[0].time / res[1].time << std::endl;
    }

    // Stage 3: Multiple processes with multiple threads.
    size_t threads = THREADS_PROCS_COUNT / world_size;

    // Waiting for the every process to end its task.
    MPI_Barrier(MPI_COMM_WORLD);

    Print(world_rank, "\nStarting calculation for processes with multiple threads");
    MpiTimer timer_2;
    this_area = CalcPartsParallel(delta, from, to, threads);

    if (world_rank != 0) { // If it is a slave proc, send the calculated area.
        MPI_Send(&this_area, 1, MPI_LONG_DOUBLE, 0, 0, MPI_COMM_WORLD);
    } else {
        std::cout << "Number of threads: " << threads
                  << "\nNumber of processes: " << world_size
                  << "\n";

        res[2].area = this_area;
        std::cout << "Result from process 0: " << this_area << "\n";

        ldouble cur_area = 0;
        for (int i = 1; i < world_size; ++i) {
            MPI_Recv(&cur_area, 1, MPI_LONG_DOUBLE, i, 0, MPI_COMM_WORLD, MPI_STATUSES_IGNORE);
            res[2].area += cur_area;
            std::cout << "Result from process " << i << ": ";
            std::cout << cur_area << "\n";
        }

        res[2].time = timer_2.GetTimeElapsed();

        std::cout << "Overall result for multiple processes with multiple threads: " << res[2].area << std::endl;
        assert(Abs(res[0].area - res[2].area) < EPS);
        std::cout << "Time taken: " << res[2].time << std::endl;
        std::cout << "Acceleration: " << std::endl << res[0].time / res[2].time << std::endl;
    }

    MPI_Finalize();
    return 0;
}
