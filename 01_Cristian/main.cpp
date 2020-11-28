#include <iostream>
#include <limits>
#include <iomanip>
#include <random>
#include <unistd.h>

#include "mpi.h"

const auto N_TRIES = 5;

void serverRoutine() {
    int size, buf;
    MPI_Status status;
    MPI_Comm_size(MPI_COMM_WORLD, &size);

    std::random_device r;
    std::default_random_engine rng(r());
    std::uniform_int_distribution<int> uniform_dist(5000, 10000);
    std::cout << std::fixed << std::setprecision(6);

    for (auto i = 0; i < (size - 1) * N_TRIES; ++i) {
        MPI_Recv(&buf, 1, MPI_INT, MPI_ANY_SOURCE, MPI_ANY_TAG, MPI_COMM_WORLD, &status);
        usleep(uniform_dist(rng));
        const auto t = MPI_Wtime();
        std::cout << "[SERVER] Received request " << status.MPI_TAG
                  << " from client " << status.MPI_SOURCE << std::endl;
        usleep(uniform_dist(rng));
        MPI_Send(&t, 1, MPI_DOUBLE, status.MPI_SOURCE, status.MPI_TAG, MPI_COMM_WORLD);
    }

    MPI_Barrier(MPI_COMM_WORLD);
    const auto localT = MPI_Wtime();
    std::cout << "[SERVER] Current time is \033[1;33m" << localT << "\033[0m" << std::endl;
}

void clientRoutine() {
    int rank;
    MPI_Status status;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);

    double t, offset;
    auto minTRound = std::numeric_limits<double>::max();
    std::cout << std::fixed << std::setprecision(6);

    for (auto i = 0; i < N_TRIES; ++i) {
        std::cout << "[CLIENT " << rank << "] Sending request " << i << " to server" << std::endl;

        const auto start = MPI_Wtime();
        MPI_Send(&rank, 1, MPI_INT, 0, i, MPI_COMM_WORLD);
        MPI_Recv(&t, 1, MPI_DOUBLE, 0, i, MPI_COMM_WORLD, &status);
        const auto tRound = MPI_Wtime() - start;

        std::cout << "[CLIENT " << rank << "] Received a response {'time': " << t << ", 'tag': "
                  << status.MPI_TAG << "} from server" << std::endl;

        if (tRound < minTRound) {
            minTRound = tRound;
            const auto localT = t + tRound / 2;
            offset = localT - MPI_Wtime();
            std::cout << "[CLIENT " << rank << "] New minimum T_round found: \033[1;31m" << tRound
                      << "\033[0m, updated local time to \033[1;31m" << localT << "\033[0m" << std::endl;
        }
    }

    MPI_Barrier(MPI_COMM_WORLD);
    const auto localT = MPI_Wtime() + offset;
    std::cout << "[CLIENT " << rank << "] Current time is \033[1;33m" << localT << "\033[0m" << std::endl;
}

int main(int argc, char **argv) {
    int rank;
    MPI_Init(&argc, &argv);
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);

    if (rank == 0) {
        serverRoutine();
    } else {
        clientRoutine();
    }

    MPI_Finalize();
    return 0;
}
