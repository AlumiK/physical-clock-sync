#include <iostream>
#include <limits>
#include <iomanip>
#include <chrono>
#include <unistd.h>
#include <mpi.h>

const auto N_TRIES = 3;

uint64_t now() {
    return duration_cast<std::chrono::microseconds>(std::chrono::system_clock::now().time_since_epoch()).count();
}

void serverRoutine() {
    int size, buf;
    MPI_Comm_size(MPI_COMM_WORLD, &size);

#pragma omp parallel for
    for (auto i = 1; i < size; ++i) {
        MPI_Status status;
        for (auto j = 0; j < N_TRIES; ++j) {
            MPI_Recv(&buf, 1, MPI_INT, i, j, MPI_COMM_WORLD, &status);
            const auto t = now();
            std::cout << "[SERVER] Received request " << status.MPI_TAG
                      << " from client " << status.MPI_SOURCE << " at " << t << std::endl;
            MPI_Send(&t, 1, MPI_UINT64_T, i, j, MPI_COMM_WORLD);
        }
    }

    MPI_Barrier(MPI_COMM_WORLD);
    const auto localT = now();
    std::cout << "[SERVER] Current timestamp(us) is \033[1;33m" << localT << "\033[0m" << std::endl;
}

void clientRoutine() {
    int rank;
    MPI_Status status;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);

    uint64_t t, offset;
    auto minTRound = std::numeric_limits<uint64_t>::max();

    for (auto i = 0; i < N_TRIES; ++i) {
        std::cout << "[CLIENT " << rank << "] Sending request " << i << " to server" << std::endl;

        const auto start = now();
        MPI_Send(&rank, 1, MPI_INT, 0, i, MPI_COMM_WORLD);
        MPI_Recv(&t, 1, MPI_UINT64_T, 0, i, MPI_COMM_WORLD, &status);
        const auto tRound = now() - start;

        std::cout << "[CLIENT " << rank << "] Received response " << status.MPI_TAG << " from server"
                  << std::endl;

        if (tRound < minTRound) {
            minTRound = tRound;
            const auto localT = t + tRound / 2;
            offset = localT - now();
            std::cout << "[CLIENT " << rank << "] New minimum T_round found: \033[1;31m" << tRound
                      << "\033[0m, set timestamp(us) to \033[1;31m" << localT << "\033[0m" << std::endl;
        }
    }

    MPI_Barrier(MPI_COMM_WORLD);
    const auto localT = now() + offset;
    std::cout << "[CLIENT " << rank << "] Current timestamp(us) is \033[1;33m" << localT << "\033[0m" << std::endl;
}

int main(int argc, char **argv) {
    int rank, provided;
    MPI_Init_thread(&argc, &argv, MPI_THREAD_MULTIPLE, &provided);
    if (provided != MPI_THREAD_MULTIPLE) {
        std::cout << "Multithreading is not supported!" << std::endl;
        return EXIT_FAILURE;
    }
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);

    if (rank == 0) {
        std::cout << "[SERVER] Initializing..." << std::endl;
        serverRoutine();
    } else {
        sleep(2);
        clientRoutine();
    }

    MPI_Finalize();
    return EXIT_SUCCESS;
}
