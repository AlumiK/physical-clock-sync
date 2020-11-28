#include <iostream>
#include <iomanip>
#include <random>
#include <unistd.h>

#include "mpi.h"

void masterRoutine() {
    MPI_Wtime();
    int size, rank = 0;
    MPI_Status status;
    MPI_Comm_size(MPI_COMM_WORLD, &size);
    std::cout << std::fixed << std::setprecision(6);

    double starts[size];
    double t[size];
    for (auto i = 1; i < size; ++i) {
        std::cout << "[MASTER] Sending request to slave " << i << std::endl;
        starts[i] = MPI_Wtime();
        MPI_Send(&rank, 1, MPI_INT, i, 0, MPI_COMM_WORLD);
    }
    for (auto i = 1; i < size; ++i) {
        double slaveT;
        MPI_Recv(&slaveT, 1, MPI_DOUBLE, MPI_ANY_SOURCE, 0, MPI_COMM_WORLD, &status);
        const auto slaveRank = status.MPI_SOURCE;
        const auto tRound = MPI_Wtime() - starts[slaveRank];
        t[slaveRank] = slaveT + tRound / 2;
        std::cout << "[MASTER] Received a response {'time': " << slaveT << "} from slave"
                  << status.MPI_SOURCE << std::endl;
    }
    t[rank] = MPI_Wtime();

    auto avgT = 0.0;
    for (const auto i : t) {
        avgT += i;
    }
    avgT /= size;
    for (auto i = 1; i < size; ++i) {
        const auto offset = avgT - t[i];
        MPI_Send(&offset, 1, MPI_DOUBLE, i, 1, MPI_COMM_WORLD);
    }
    const auto offset = avgT - t[rank];

    MPI_Barrier(MPI_COMM_WORLD);
    const auto localT = MPI_Wtime() + offset;
    std::cout << "[MASTER] Current time is \033[1;33m" << localT << "\033[0m" << std::endl;
}

void slaveRoutine() {
    MPI_Wtime();
    int rank, buf;
    MPI_Status status;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);

    std::random_device r;
    std::default_random_engine rng(r());
    std::uniform_int_distribution<int> uniform_dist(50000, 100000);
    std::cout << std::fixed << std::setprecision(6);

    MPI_Recv(&buf, 1, MPI_INT, 0, 0, MPI_COMM_WORLD, &status);
    usleep(uniform_dist(rng));
    auto t = MPI_Wtime();
    std::cout << "[SLAVE " << rank << "] Received a request from master at " << t << std::endl;
    usleep(uniform_dist(rng));
    MPI_Send(&t, 1, MPI_DOUBLE, 0, 0, MPI_COMM_WORLD);

    double offset;
    MPI_Recv(&offset, 1, MPI_DOUBLE, 0, 1, MPI_COMM_WORLD, &status);

    MPI_Barrier(MPI_COMM_WORLD);
    const auto localT = MPI_Wtime() + offset;
    std::cout << "[SLAVE " << rank << "] Current time is \033[1;33m" << localT << "\033[0m" << std::endl;
}

int main(int argc, char **argv) {
    int rank;
    MPI_Init(&argc, &argv);
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);

    if (rank == 0) {
        masterRoutine();
    } else {
        slaveRoutine();
    }

    MPI_Finalize();
    return 0;
}
