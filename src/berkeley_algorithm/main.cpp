#include <iostream>
#include <iomanip>
#include <chrono>
#include <mpi.h>

uint64_t now() {
    return duration_cast<std::chrono::microseconds>(std::chrono::system_clock::now().time_since_epoch()).count();
}

void masterRoutine() {
    int size, rank = 0;
    MPI_Comm_size(MPI_COMM_WORLD, &size);

    uint64_t t[size];

#pragma omp parallel for
    for (auto i = 1; i < size; ++i) {
        MPI_Status status;
        std::cout << "[MASTER] Sending a request to slave " << i << std::endl;
        uint64_t slaveT;
        const auto start = now();
        MPI_Send(&rank, 1, MPI_INT, i, 0, MPI_COMM_WORLD);
        MPI_Recv(&slaveT, 1, MPI_UINT64_T, i, 0, MPI_COMM_WORLD, &status);
        const auto T_round = now() - start;
        t[i] = slaveT + T_round / 2;
        std::cout << "[MASTER] Received a response from slave" << i << ", T_round is " << T_round << std::endl;
    }
    t[rank] = now();

    uint64_t avgT = 0;
    for (const auto i : t) {
        avgT += i;
    }
    avgT /= size;
    for (auto i = 1; i < size; ++i) {
        const auto offset = avgT - t[i];
        MPI_Send(&offset, 1, MPI_UINT64_T, i, 1, MPI_COMM_WORLD);
    }
    const auto offset = avgT - t[rank];

    MPI_Barrier(MPI_COMM_WORLD);
    const auto localT = now() + offset;
    std::cout << "[MASTER] Current timestamp(us) is \033[1;33m" << localT << "\033[0m" << std::endl;
}

void slaveRoutine() {
    int rank, buf;
    MPI_Status status;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);

    MPI_Recv(&buf, 1, MPI_INT, 0, 0, MPI_COMM_WORLD, &status);
    const auto t = now();
    std::cout << "[SLAVE " << rank << "] Received a request from master at " << t << std::endl;
    MPI_Send(&t, 1, MPI_UINT64_T, 0, 0, MPI_COMM_WORLD);

    uint64_t offset;
    MPI_Recv(&offset, 1, MPI_UINT64_T, 0, 1, MPI_COMM_WORLD, &status);

    MPI_Barrier(MPI_COMM_WORLD);
    const auto localT = now() + offset;
    std::cout << "[SLAVE " << rank << "] Current timestamp(us) is \033[1;33m" << localT << "\033[0m" << std::endl;
}

int main(int argc, char **argv) {
    int rank, provided;
    MPI_Init_thread(&argc, &argv, MPI_THREAD_MULTIPLE, &provided);
    if (provided != MPI_THREAD_MULTIPLE) {
        std::cout << "Multithreading is not supported!" << std::endl;
        exit(-1);
    }
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);

    if (rank == 0) {
        masterRoutine();
    } else {
        slaveRoutine();
    }

    MPI_Finalize();
    return 0;
}
