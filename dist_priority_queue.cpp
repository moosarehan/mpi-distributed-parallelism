/*
 * =============================================================================
 * Question No 3: Concurrent Distributed Priority Queue
 * =============================================================================
 *
 * This program implements a distributed priority queue where multiple workers
 * send tasks to a root process. The root maintains a binary heap and processes
 * tasks using non-blocking operations and polling.
 *
 * Comparison of 3 methods:
 *   1. Blocking collection
 *   2. Non-blocking with polling (MPI_Test)
 *   3. Non-blocking with batching
 *
 * Compilation:
 *   mpicxx -O2 -o dist_priority_queue dist_priority_queue.cpp
 *
 * Execution:
 *   mpiexec -n 8 ./dist_priority_queue <Tasks per worker K>
 * =============================================================================
 */

#include <mpi.h>
#include <cstdio>
#include <cstdlib>
#include <vector>
#include <queue>
#include <algorithm>
#include <chrono>
#include <thread>

// Task structure
struct Task {
    int priority;
    int payload;

    // Min-heap: higher priority (smaller value) comes first
    bool operator>(const Task& other) const {
        return priority > other.priority;
    }
};

// Tag constants
const int TAG_TASK = 10;
const int TAG_TERMINATE = 99;

/* ──────────────────────────────────────────────────────────────────────────── *
 *  ROOT LOGIC: Blocking Collection                                            *
 * ──────────────────────────────────────────────────────────────────────────── */
void root_blocking(int P, int K, std::priority_queue<Task, std::vector<Task>, std::greater<Task>>& pq) {
    int total_tasks = (P - 1) * K;
    for (int i = 0; i < total_tasks; ++i) {
        Task t;
        MPI_Recv(&t, 2, MPI_INT, MPI_ANY_SOURCE, TAG_TASK, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
        pq.push(t);
    }
}

/* ──────────────────────────────────────────────────────────────────────────── *
 *  ROOT LOGIC: Non-blocking with Polling (MPI_Test)                          *
 * ──────────────────────────────────────────────────────────────────────────── */
void root_nonblocking_polling(int P, int K, std::priority_queue<Task, std::vector<Task>, std::greater<Task>>& pq, double& productive_time, double& wait_time) {
    int total_tasks = (P - 1) * K;
    int received_total = 0;
    std::vector<int> received_per_worker(P, 0);
    
    std::vector<Task> buffers(P);
    std::vector<MPI_Request> requests(P, MPI_REQUEST_NULL);

    // Initial receives
    for (int i = 1; i < P; ++i) {
        MPI_Irecv(&buffers[i], 2, MPI_INT, i, TAG_TASK, MPI_COMM_WORLD, &requests[i]);
    }

    while (received_total < total_tasks) {
        int completed_idx = -1;
        auto t_wait_start = MPI_Wtime();
        MPI_Waitany(P, requests.data(), &completed_idx, MPI_STATUS_IGNORE);
        auto t_wait_end = MPI_Wtime();
        wait_time += (t_wait_end - t_wait_start);

        if (completed_idx != MPI_UNDEFINED && completed_idx >= 1 && completed_idx < P) {
            auto t_prod_start = MPI_Wtime();
            pq.push(buffers[completed_idx]);
            received_total++;
            received_per_worker[completed_idx]++;
            auto t_prod_end = MPI_Wtime();
            productive_time += (t_prod_end - t_prod_start);

            if (received_per_worker[completed_idx] < K) {
                MPI_Irecv(&buffers[completed_idx], 2, MPI_INT, completed_idx, TAG_TASK, MPI_COMM_WORLD, &requests[completed_idx]);
            }
        }
    }
}

/* ──────────────────────────────────────────────────────────────────────────── *
 *  ROOT LOGIC: Non-blocking with Batching                                     *
 * ──────────────────────────────────────────────────────────────────────────── */
const int BATCH_SIZE = 5;
struct TaskBatch {
    Task tasks[BATCH_SIZE];
};

void root_batching(int P, int K, std::priority_queue<Task, std::vector<Task>, std::greater<Task>>& pq) {
    int total_batches = ((P - 1) * K) / BATCH_SIZE;
    for (int i = 0; i < total_batches; ++i) {
        TaskBatch batch;
        MPI_Recv(&batch, 2 * BATCH_SIZE, MPI_INT, MPI_ANY_SOURCE, TAG_TASK, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
        for (int b = 0; b < BATCH_SIZE; ++b) {
            pq.push(batch.tasks[b]);
        }
    }
}

/* ──────────────────────────────────────────────────────────────────────────── *
 *  WORKER LOGIC                                                               *
 * ──────────────────────────────────────────────────────────────────────────── */
void worker_logic(int rank, int K, int method) {
    srand(42 + rank);
    if (method == 1) { // Blocking
        for (int i = 0; i < K; ++i) {
            Task t = { rand() % 1000, rank };
            MPI_Send(&t, 2, MPI_INT, 0, TAG_TASK, MPI_COMM_WORLD);
        }
    } else if (method == 2) { // Non-blocking
        std::vector<Task> tasks(K);
        std::vector<MPI_Request> reqs(K);
        for (int i = 0; i < K; ++i) {
            tasks[i] = { rand() % 1000, rank };
            MPI_Isend(&tasks[i], 2, MPI_INT, 0, TAG_TASK, MPI_COMM_WORLD, &reqs[i]);
        }
        MPI_Waitall(K, reqs.data(), MPI_STATUSES_IGNORE);
    } else if (method == 3) { // Batching
        int batches = K / BATCH_SIZE;
        for (int i = 0; i < batches; ++i) {
            TaskBatch batch;
            for (int b = 0; b < BATCH_SIZE; ++b) {
                batch.tasks[b] = { rand() % 1000, rank };
            }
            MPI_Send(&batch, 2 * BATCH_SIZE, MPI_INT, 0, TAG_TASK, MPI_COMM_WORLD);
        }
    }
}

/* ══════════════════════════════════════════════════════════════════════════ *
 *  MAIN                                                                     *
 * ══════════════════════════════════════════════════════════════════════════ */
int main(int argc, char** argv) {
    MPI_Init(&argc, &argv);

    int rank, P;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &P);

    int K = (argc > 1) ? atoi(argv[1]) : 100;
    // For batching method to work cleanly, K should be multiple of BATCH_SIZE
    K = (K / BATCH_SIZE) * BATCH_SIZE;

    std::priority_queue<Task, std::vector<Task>, std::greater<Task>> pq;
    double t_start, t_end;
    double results[3]; // times
    double prod_time = 0, wait_time = 0;

    for (int method = 1; method <= 3; ++method) {
        // Clear queue
        while(!pq.empty()) pq.pop();
        MPI_Barrier(MPI_COMM_WORLD);

        t_start = MPI_Wtime();
        if (rank == 0) {
            if (method == 1) root_blocking(P, K, pq);
            else if (method == 2) root_nonblocking_polling(P, K, pq, prod_time, wait_time);
            else if (method == 3) root_batching(P, K, pq);
        } else {
            worker_logic(rank, K, method);
        }
        MPI_Barrier(MPI_COMM_WORLD);
        t_end = MPI_Wtime();
        results[method-1] = t_end - t_start;

        if (rank == 0 && method == 2) {
            // Print sorted list at the end of method 2 (or any)
            printf("\nFinal Sorted Priority List (Sample of first 10):\n");
            int count = 0;
            while(!pq.empty() && count < 10) {
                Task t = pq.top(); pq.pop();
                printf("Priority: %-4d | Payload (Rank): %d\n", t.priority, t.payload);
                count++;
            }
        }
    }

    if (rank == 0) {
        printf("\n+-------------------------------------------------------------+\n");
        printf("|       Distributed Priority Queue Performance Analysis       |\n");
        printf("+-------------------------------------------------------------+\n");
        printf("| Method                     | Time (s)    | Throughput (t/s) |\n");
        printf("+----------------------------+-------------+------------------+\n");
        double total_tasks = (P - 1) * K;
        printf("| Blocking Collection        | %11.6f | %16.2f |\n", results[0], total_tasks/results[0]);
        printf("| Non-blocking + Waitany     | %11.6f | %16.2f |\n", results[1], total_tasks/results[1]);
        printf("| Blocking + Batching        | %11.6f | %16.2f |\n", results[2], total_tasks/results[2]);
        printf("+----------------------------+-------------+------------------+\n");
        
        printf("\nRoot Process CPU Utilization (Non-blocking Method):\n");
        printf("  Productive Work (Heap Inserts): %11.6f s (%.2f%%)\n", prod_time, (prod_time/results[1])*100);
        printf("  MPI Waiting / Idle Time:        %11.6f s (%.2f%%)\n", wait_time, (wait_time/results[1])*100);
        printf("+-------------------------------------------------------------+\n");
    }

    MPI_Finalize();
    return 0;
}
