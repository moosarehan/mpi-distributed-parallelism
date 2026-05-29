/*
 * =============================================================================
 * Hierarchical Tree-Based Broadcast and Reduction on a Large Cluster
 * =============================================================================
 *
 * This program implements and compares:
 *   1. Tree-based broadcast  vs  Naive (flat) broadcast
 *   2. Tree-based reduction  vs  Naive (flat) reduction
 *
 * Processes are organized into a logical k-ary tree rooted at process r.
 * The tree topology is computed so that the root is always at position 0
 * in the logical ordering (achieved by rank remapping).
 *
 * Compile:
 *   mpicxx -O2 -o tree_collectives tree_collectives.cpp
 *
 * Run (example with 8 processes, k=2, root=0, N=10000):
 *   mpirun -np 8 ./tree_collectives 2 0 10000
 *
 * Arguments:
 *   argv[1] = k       (tree fan-out, e.g. 2, 4, 8)
 *   argv[2] = r       (root process rank, 0 <= r < P)
 *   argv[3] = N       (size of local array)
 *
 * =============================================================================
 */

#include <mpi.h>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <cmath>
#include <vector>
#include <ctime>

/* ──────────────────────────────────────────────────────────────────────────── *
 *  Tag constants to separate broadcast and reduction messages                 *
 * ──────────────────────────────────────────────────────────────────────────── */
static const int TAG_TREE_BCAST  = 100;
static const int TAG_TREE_REDUCE = 200;
static const int TAG_NAIVE_BCAST = 300;
static const int TAG_NAIVE_REDUCE = 400;

/* ──────────────────────────────────────────────────────────────────────────── *
 *  Utility: map between actual MPI rank and logical rank (root-remapped)      *
 *  Logical rank 0 always corresponds to the actual root process r.            *
 * ──────────────────────────────────────────────────────────────────────────── */
static inline int actual_to_logical(int actual_rank, int root, int P) {
    return (actual_rank - root + P) % P;
}

static inline int logical_to_actual(int logical_rank, int root, int P) {
    return (logical_rank + root) % P;
}

/* ──────────────────────────────────────────────────────────────────────────── *
 *  k-ary tree helper: parent and children in logical rank space               *
 * ──────────────────────────────────────────────────────────────────────────── */

// Returns logical rank of parent (-1 if this is the root, i.e. logical_rank==0)
static inline int tree_parent(int logical_rank, int k) {
    if (logical_rank == 0) return -1;
    return (logical_rank - 1) / k;
}

// Fills 'children' with the logical ranks of children of 'logical_rank'
static void tree_children(int logical_rank, int k, int P,
                           std::vector<int>& children) {
    children.clear();
    for (int i = 1; i <= k; ++i) {
        int child = k * logical_rank + i;
        if (child < P) {
            children.push_back(child);
        }
    }
}

/* ══════════════════════════════════════════════════════════════════════════ *
 *  TREE-BASED BROADCAST                                                     *
 *  Root sends buffer down the k-ary tree level by level.                    *
 * ══════════════════════════════════════════════════════════════════════════ */
void tree_broadcast(double* buf, int N, int root, int k,
                    int rank, int P, MPI_Comm comm) {

    int logical = actual_to_logical(rank, root, P);
    int parent_logical = tree_parent(logical, k);

    std::vector<int> children;
    tree_children(logical, k, P, children);

    /* ---- Receive from parent (if not root) ---- */
    if (parent_logical >= 0) {
        int parent_actual = logical_to_actual(parent_logical, root, P);
        MPI_Recv(buf, N, MPI_DOUBLE, parent_actual, TAG_TREE_BCAST,
                 comm, MPI_STATUS_IGNORE);
    }

    /* ---- Forward to children ---- */
    for (int i = 0; i < (int)children.size(); ++i) {
        int child_actual = logical_to_actual(children[i], root, P);
        MPI_Send(buf, N, MPI_DOUBLE, child_actual, TAG_TREE_BCAST, comm);
    }
}

/* ══════════════════════════════════════════════════════════════════════════ *
 *  TREE-BASED REDUCTION (SUM)                                               *
 *  Leaves send up to parents; each parent accumulates children then sends   *
 *  up.  Result ends at root.                                                *
 * ══════════════════════════════════════════════════════════════════════════ */
void tree_reduce(double* sendbuf, double* recvbuf, int N, int root, int k,
                 int rank, int P, MPI_Comm comm) {

    int logical = actual_to_logical(rank, root, P);
    int parent_logical = tree_parent(logical, k);

    std::vector<int> children;
    tree_children(logical, k, P, children);

    /* Start with own data */
    std::vector<double> accum(sendbuf, sendbuf + N);
    std::vector<double> tmp(N);

    /* ---- Receive from children and accumulate ---- */
    for (int i = 0; i < (int)children.size(); ++i) {
        int child_actual = logical_to_actual(children[i], root, P);
        MPI_Recv(tmp.data(), N, MPI_DOUBLE, child_actual, TAG_TREE_REDUCE,
                 comm, MPI_STATUS_IGNORE);
        for (int j = 0; j < N; ++j) {
            accum[j] += tmp[j];
        }
    }

    /* ---- Send accumulated result to parent (if not root) ---- */
    if (parent_logical >= 0) {
        int parent_actual = logical_to_actual(parent_logical, root, P);
        MPI_Send(accum.data(), N, MPI_DOUBLE, parent_actual,
                 TAG_TREE_REDUCE, comm);
    } else {
        /* Root: copy result into recvbuf */
        memcpy(recvbuf, accum.data(), N * sizeof(double));
    }
}

/* ══════════════════════════════════════════════════════════════════════════ *
 *  NAIVE (FLAT) BROADCAST                                                   *
 *  Root sends directly to every other process (point-to-point).             *
 * ══════════════════════════════════════════════════════════════════════════ */
void naive_broadcast(double* buf, int N, int root,
                     int rank, int P, MPI_Comm comm) {
    if (rank == root) {
        for (int dest = 0; dest < P; ++dest) {
            if (dest != root) {
                MPI_Send(buf, N, MPI_DOUBLE, dest, TAG_NAIVE_BCAST, comm);
            }
        }
    } else {
        MPI_Recv(buf, N, MPI_DOUBLE, root, TAG_NAIVE_BCAST,
                 comm, MPI_STATUS_IGNORE);
    }
}

/* ══════════════════════════════════════════════════════════════════════════ *
 *  NAIVE (FLAT) REDUCTION (SUM)                                             *
 *  Every non-root process sends its data directly to root.                  *
 * ══════════════════════════════════════════════════════════════════════════ */
void naive_reduce(double* sendbuf, double* recvbuf, int N, int root,
                  int rank, int P, MPI_Comm comm) {
    if (rank == root) {
        /* Start with root's own data */
        memcpy(recvbuf, sendbuf, N * sizeof(double));
        std::vector<double> tmp(N);
        for (int src = 0; src < P; ++src) {
            if (src != root) {
                MPI_Recv(tmp.data(), N, MPI_DOUBLE, src, TAG_NAIVE_REDUCE,
                         comm, MPI_STATUS_IGNORE);
                for (int j = 0; j < N; ++j) {
                    recvbuf[j] += tmp[j];
                }
            }
        }
    } else {
        MPI_Send(sendbuf, N, MPI_DOUBLE, root, TAG_NAIVE_REDUCE, comm);
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

    /* ── Parse arguments ── */
    if (argc < 4) {
        if (rank == 0) {
            fprintf(stderr,
                "Usage: %s <k (fan-out)> <root> <N (array size)>\n", argv[0]);
        }
        MPI_Finalize();
        return 1;
    }

    int k    = atoi(argv[1]);
    int root = atoi(argv[2]);
    int N    = atoi(argv[3]);

    if (P < 8) {
        if (rank == 0) {
            fprintf(stderr, "Error: P must be >= 8 (got %d)\n", P);
        }
        MPI_Finalize();
        return 1;
    }
    if (k < 2) {
        if (rank == 0) {
            fprintf(stderr, "Error: k must be >= 2 (got %d)\n", k);
        }
        MPI_Finalize();
        return 1;
    }
    if (root < 0 || root >= P) {
        if (rank == 0) {
            fprintf(stderr, "Error: root must be in [0, %d) (got %d)\n", P, root);
        }
        MPI_Finalize();
        return 1;
    }

    /* ── Seed RNG uniquely per process ── */
    srand(42 + rank);

    /* ── Allocate local array A_i filled with random values ── */
    std::vector<double> A_local(N);
    for (int i = 0; i < N; ++i) {
        A_local[i] = (double)(rand() % 1000) / 10.0;   // values in [0, 99.9]
    }

    /* ── Allocate buffers ── */
    std::vector<double> B(N, 0.0);                // broadcast buffer
    std::vector<double> A_sum_tree(N, 0.0);       // tree reduction result
    std::vector<double> A_sum_naive(N, 0.0);      // naive reduction result
    std::vector<double> A_sum_serial(N, 0.0);     // serial reference (MPI_Reduce)

    double t_start, t_end;

    /* ────────────────────────────────────────────────────────────────────── *
     *  1. CORRECTNESS TEST: Tree-based Reduction                            *
     * ────────────────────────────────────────────────────────────────────── */

    MPI_Barrier(MPI_COMM_WORLD);
    tree_reduce(A_local.data(), A_sum_tree.data(), N, root, k,
                rank, P, MPI_COMM_WORLD);

    /* Compute reference via MPI_Reduce for validation */
    MPI_Reduce(A_local.data(), A_sum_serial.data(), N, MPI_DOUBLE,
               MPI_SUM, root, MPI_COMM_WORLD);

    if (rank == root) {
        bool correct = true;
        for (int i = 0; i < N; ++i) {
            if (fabs(A_sum_tree[i] - A_sum_serial[i]) > 1e-6) {
                correct = false;
                break;
            }
        }
        printf("Tree Collectives: %s\n\n", correct ? "CORRECT" : "INCORRECT");
    }

    /* ────────────────────────────────────────────────────────────────────── *
     *  2. CORRECTNESS TEST: Tree-based Broadcast                            *
     * ────────────────────────────────────────────────────────────────────── */

    /* Root prepares broadcast vector */
    if (rank == root) {
        for (int i = 0; i < N; ++i) B[i] = (double)(i + 1);
    }

    MPI_Barrier(MPI_COMM_WORLD);
    tree_broadcast(B.data(), N, root, k, rank, P, MPI_COMM_WORLD);

    /* Verify: every process should have B[i] == i+1 */
    {
        int local_ok = 1;
        for (int i = 0; i < N; ++i) {
            if (fabs(B[i] - (double)(i + 1)) > 1e-9) {
                local_ok = 0;
                break;
            }
        }
        int global_ok = 0;
        MPI_Reduce(&local_ok, &global_ok, 1, MPI_INT, MPI_MIN,
                   root, MPI_COMM_WORLD);
        if (rank == root) {
            printf("Tree Broadcast Verification: %s\n\n",
                   global_ok ? "CORRECT" : "INCORRECT");
        }
    }

    /* ────────────────────────────────────────────────────────────────────── *
     *  3. PERFORMANCE MEASUREMENTS                                          *
     * ────────────────────────────────────────────────────────────────────── */

    const int WARMUP = 2;
    const int ITERS  = 5;

    /* ── 3a. Naive Broadcast Timing ── */
    for (int w = 0; w < WARMUP; ++w) {
        if (rank == root) {
            for (int i = 0; i < N; ++i) B[i] = (double)(i + 1);
        } else {
            memset(B.data(), 0, N * sizeof(double));
        }
        MPI_Barrier(MPI_COMM_WORLD);
        naive_broadcast(B.data(), N, root, rank, P, MPI_COMM_WORLD);
    }

    MPI_Barrier(MPI_COMM_WORLD);
    t_start = MPI_Wtime();
    for (int it = 0; it < ITERS; ++it) {
        if (rank == root) {
            for (int i = 0; i < N; ++i) B[i] = (double)(i + 1);
        } else {
            memset(B.data(), 0, N * sizeof(double));
        }
        MPI_Barrier(MPI_COMM_WORLD);
        naive_broadcast(B.data(), N, root, rank, P, MPI_COMM_WORLD);
    }
    t_end = MPI_Wtime();
    double time_naive_bcast = (t_end - t_start) / ITERS;

    /* ── 3b. Tree Broadcast Timing ── */
    for (int w = 0; w < WARMUP; ++w) {
        if (rank == root) {
            for (int i = 0; i < N; ++i) B[i] = (double)(i + 1);
        } else {
            memset(B.data(), 0, N * sizeof(double));
        }
        MPI_Barrier(MPI_COMM_WORLD);
        tree_broadcast(B.data(), N, root, k, rank, P, MPI_COMM_WORLD);
    }

    MPI_Barrier(MPI_COMM_WORLD);
    t_start = MPI_Wtime();
    for (int it = 0; it < ITERS; ++it) {
        if (rank == root) {
            for (int i = 0; i < N; ++i) B[i] = (double)(i + 1);
        } else {
            memset(B.data(), 0, N * sizeof(double));
        }
        MPI_Barrier(MPI_COMM_WORLD);
        tree_broadcast(B.data(), N, root, k, rank, P, MPI_COMM_WORLD);
    }
    t_end = MPI_Wtime();
    double time_tree_bcast = (t_end - t_start) / ITERS;

    /* ── 3c. Naive Reduction Timing ── */
    for (int w = 0; w < WARMUP; ++w) {
        MPI_Barrier(MPI_COMM_WORLD);
        naive_reduce(A_local.data(), A_sum_naive.data(), N, root,
                     rank, P, MPI_COMM_WORLD);
    }

    MPI_Barrier(MPI_COMM_WORLD);
    t_start = MPI_Wtime();
    for (int it = 0; it < ITERS; ++it) {
        MPI_Barrier(MPI_COMM_WORLD);
        naive_reduce(A_local.data(), A_sum_naive.data(), N, root,
                     rank, P, MPI_COMM_WORLD);
    }
    t_end = MPI_Wtime();
    double time_naive_reduce = (t_end - t_start) / ITERS;

    /* ── 3d. Tree Reduction Timing ── */
    for (int w = 0; w < WARMUP; ++w) {
        MPI_Barrier(MPI_COMM_WORLD);
        tree_reduce(A_local.data(), A_sum_tree.data(), N, root, k,
                    rank, P, MPI_COMM_WORLD);
    }

    MPI_Barrier(MPI_COMM_WORLD);
    t_start = MPI_Wtime();
    for (int it = 0; it < ITERS; ++it) {
        MPI_Barrier(MPI_COMM_WORLD);
        tree_reduce(A_local.data(), A_sum_tree.data(), N, root, k,
                    rank, P, MPI_COMM_WORLD);
    }
    t_end = MPI_Wtime();
    double time_tree_reduce = (t_end - t_start) / ITERS;

    /* ── Gather max times across all ranks for fair reporting ── */
    double times_local[4] = {
        time_naive_bcast, time_tree_bcast,
        time_naive_reduce, time_tree_reduce
    };
    double times_max[4];
    MPI_Reduce(times_local, times_max, 4, MPI_DOUBLE, MPI_MAX,
               root, MPI_COMM_WORLD);

    /* ────────────────────────────────────────────────────────────────────── *
     *  4. PRINT SUMMARY TABLE (root only)                                   *
     * ────────────────────────────────────────────────────────────────────── */
    if (rank == root) {
        printf("+--------------------------------------------------------------+\n");
        printf("|       Hierarchical Tree-Based Collective Operations          |\n");
        printf("+--------------------------------------------------------------+\n");
        printf("|  Configuration:                                              |\n");
        printf("|    Processes (P) = %-4d    Fan-out (k) = %-4d                |\n", P, k);
        printf("|    Root          = %-4d    Array size  = %-8d            |\n", root, N);
        printf("|    Iterations    = %-4d    Warmup      = %-4d                |\n", ITERS, WARMUP);
        printf("+-------------------------+--------------+------------+--------+\n");
        printf("|  Operation              |  Naive (s)   |  Tree (s)  | Speedup|\n");
        printf("+-------------------------+--------------+------------+--------+\n");

        double speedup_bcast  = times_max[0] / (times_max[1] > 0 ? times_max[1] : 1e-15);
        double speedup_reduce = times_max[2] / (times_max[3] > 0 ? times_max[3] : 1e-15);

        printf("|  Broadcast              | %12.6f | %10.6f | %5.2fx |\n",
               times_max[0], times_max[1], speedup_bcast);
        printf("|  Reduction (Sum)        | %12.6f | %10.6f | %5.2fx |\n",
               times_max[2], times_max[3], speedup_reduce);
        printf("+-------------------------+--------------+------------+--------+\n");
    }

    MPI_Finalize();
    return 0;
}
