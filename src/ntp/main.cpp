#include <iostream>
#include <chrono>
#include <random>
#include <unistd.h>
#include <mpi.h>

uint64_t now() {
    return duration_cast<std::chrono::milliseconds>(std::chrono::system_clock::now().time_since_epoch()).count();
}

int main(int argc, char **argv) {
    int size, rank, buf = 0;
    MPI_Status status;
    MPI_Init(&argc, &argv);
    MPI_Comm_size(MPI_COMM_WORLD, &size);
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);

    std::random_device r;
    std::default_random_engine rng(r());
    std::uniform_int_distribution<int> uniform_dist(10000, 20000);

    uint64_t offset = 0;
    uint64_t t[4];
    for (auto i = 1; i < size; ++i) {
        if (rank == 0) {
            std::cout << "Synchronizing server " << i - 1 << " and server " << i << "..." << std::endl;
        }
        if (rank == i - 1) {
            std::cout << "[SERVER " << rank << "] Sending a request to server " << i << std::endl;
            MPI_Send(&buf, 1, MPI_INT, i, 0, MPI_COMM_WORLD);

            MPI_Recv(t, 1, MPI_UINT64_T, i, 1, MPI_COMM_WORLD, &status);
            t[1] = now();
            std::cout << "[SERVER " << rank << "] Received m from server " << status.MPI_SOURCE << " at "
                      << t[1] << std::endl;

            usleep(uniform_dist(rng));

            std::cout << "[SERVER " << rank << "] Sending m' to server " << i << std::endl;
            t[2] = now();
            MPI_Send(t + 1, 2, MPI_UINT64_T, i, 2, MPI_COMM_WORLD);
        } else if (rank == i) {
            MPI_Recv(&buf, 1, MPI_INT, i - 1, 0, MPI_COMM_WORLD, &status);
            std::cout << "[SERVER " << rank << "] Received a request from server " << status.MPI_SOURCE << std::endl;

            std::cout << "[SERVER " << rank << "] Sending m to server " << i - 1 << std::endl;
            t[0] = now();
            MPI_Send(t, 1, MPI_UINT64_T, i - 1, 1, MPI_COMM_WORLD);

            MPI_Recv(t + 1, 2, MPI_UINT64_T, i - 1, 2, MPI_COMM_WORLD, &status);
            t[3] = now();
            std::cout << "[SERVER " << rank << "] Received m' from server " << status.MPI_SOURCE << " at "
                      << t[3] << std::endl;

            const auto localT = (t[3] + t[2] + t[1] - t[0]) / 2;
            offset = localT - now();
            std::cout << "[SERVER " << rank << "] Set timestamp(ms) to \033[1;31m" << localT << "\033[0m" << std::endl;
        }
        MPI_Barrier(MPI_COMM_WORLD);
    }

    const auto localT = now() + offset;
    std::cout << "[SERVER " << rank << "] Current timestamp(ms) is \033[1;33m" << localT << "\033[0m" << std::endl;

    MPI_Finalize();
    return 0;
}
