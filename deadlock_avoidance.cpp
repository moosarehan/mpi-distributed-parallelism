/*
 * =============================================================================
 * Question No 2: Dynamic Deadlock-Avoiding Message Exchange in MPI
 * =============================================================================
 * 
 * This program implements a two-phase message exchange protocol designed to
 * avoid circular waits (deadlocks) without relying on MPI's internal buffering.
 *
 * Phase 1: Exchange between processes with an ODD rank difference.
 * Phase 2: Exchange between processes with an EVEN rank difference.
 *
 * Comparison:
 * 1. Naive Blocking: Standard MPI_Send/MPI_Recv (prone to deadlock).
 * 2. Handshake-based: Custom protocol using rank-based ordering to avoid cycles.
 *
 * Compilation:
 *   mpicxx -O2 -o deadlock_avoidance deadlock_avoidance.cpp
 *
 * Execution:
 *   mpiexec -n 8 ./deadlock_avoidance
 * =============================================================================
 */

#include <mpi.h>
#include <cstdio>
#include <cstdlib>
#include <vector>
#include <algorithm>

// Tags for different message types
enum MsgTags {
    TAG_RTS = 1,      // Request to Send
    TAG_CTS = 2,      // Clear to Send
    TAG_DATA = 3,     // Actual Data
    TAG_NAIVE = 4     // Naive Data
};

/* ──────────────────────────────────────────────────────────────────────────── *
 *  NAIVE BLOCKING COMMUNICATION                                               *
 *  This method is prone to deadlock because everyone might be sending.        *
 *  To make it "runnable" for comparison, we use a simple ordering:           *
 *  rank i sends to j if i < j, else receives.                                *
 * ──────────────────────────────────────────────────────────────────────────── */
void naive_exchange(int phase, int rank, int P, std::vector<std::vector<int>>& send_matrix, std::vector<std::vector<int>>& recv_matrix) {
    for (int j = 0; j < P; ++j) {
        if (rank == j) continue;
        
        int diff = std::abs(rank - j);
        bool in_phase = (phase == 1) ? (diff % 2 != 0) : (diff % 2 == 0);
        
        if (in_phase) {
            // Rank-based ordering to avoid immediate deadlock
            if (rank < j) {
                // Send then receive
                MPI_Send(&send_matrix[rank][j], 1, MPI_INT, j, TAG_NAIVE, MPI_COMM_WORLD);
                MPI_Recv(&recv_matrix[rank][j], 1, MPI_INT, j, TAG_NAIVE, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
            } else {
                // Receive then send
                MPI_Recv(&recv_matrix[rank][j], 1, MPI_INT, j, TAG_NAIVE, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
                MPI_Send(&send_matrix[rank][j], 1, MPI_INT, j, TAG_NAIVE, MPI_COMM_WORLD);
            }
        }
    }
}

/* ══════════════════════════════════════════════════════════════════════════ *
 *  HANDSHAKE-BASED SAFE COMMUNICATION                                       *
 *  Implements a RTS (Request to Send) and CTS (Clear to Send) protocol.     *
 *  Uses rank-based partial ordering to initiate handshakes.                 *
 * ══════════════════════════════════════════════════════════════════════════ */
void handshake_exchange(int phase, int rank, int P, std::vector<std::vector<int>>& send_matrix, std::vector<std::vector<int>>& recv_matrix) {
    for (int j = 0; j < P; ++j) {
        if (rank == j) continue;
        
        int diff = std::abs(rank - j);
        bool in_phase = (phase == 1) ? (diff % 2 != 0) : (diff % 2 == 0);
        
        if (in_phase) {
            int dummy = 1;
            if (rank < j) {
                // I am the initiator for my send
                MPI_Ssend(&dummy, 1, MPI_INT, j, TAG_RTS, MPI_COMM_WORLD);
                MPI_Recv(&dummy, 1, MPI_INT, j, TAG_CTS, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
                MPI_Ssend(&send_matrix[rank][j], 1, MPI_INT, j, TAG_DATA, MPI_COMM_WORLD);
                
                // I am the responder for j's send
                MPI_Recv(&dummy, 1, MPI_INT, j, TAG_RTS + 100, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
                MPI_Ssend(&dummy, 1, MPI_INT, j, TAG_CTS + 100, MPI_COMM_WORLD);
                MPI_Recv(&recv_matrix[rank][j], 1, MPI_INT, j, TAG_DATA + 100, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
            } else {
                // I am the responder for j's send
                MPI_Recv(&dummy, 1, MPI_INT, j, TAG_RTS, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
                MPI_Ssend(&dummy, 1, MPI_INT, j, TAG_CTS, MPI_COMM_WORLD);
                MPI_Recv(&recv_matrix[rank][j], 1, MPI_INT, j, TAG_DATA, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
                
                // I am the initiator for my send
                MPI_Ssend(&dummy, 1, MPI_INT, j, TAG_RTS + 100, MPI_COMM_WORLD);
                MPI_Recv(&dummy, 1, MPI_INT, j, TAG_CTS + 100, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
                MPI_Ssend(&send_matrix[rank][j], 1, MPI_INT, j, TAG_DATA + 100, MPI_COMM_WORLD);
            }
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

    if (P < 8) {
        if (rank == 0) printf("Error: P must be >= 8\n");
        MPI_Finalize();
        return 1;
    }

    // Initialize matrices
    std::vector<std::vector<int>> send_matrix(P, std::vector<int>(P));
    std::vector<std::vector<int>> recv_matrix_naive(P, std::vector<int>(P, -1));
    std::vector<std::vector<int>> recv_matrix_handshake(P, std::vector<int>(P, -1));

    for (int j = 0; j < P; ++j) {
        send_matrix[rank][j] = rank * 100 + j; // Unique message
    }

    double t_naive_start, t_naive_end;
    double t_hs_start, t_hs_end;

    // --- Naive Benchmarking ---
    MPI_Barrier(MPI_COMM_WORLD);
    t_naive_start = MPI_Wtime();
    
    // Phase 1: Odd diff
    naive_exchange(1, rank, P, send_matrix, recv_matrix_naive);
    MPI_Barrier(MPI_COMM_WORLD);
    
    // Phase 2: Even diff
    naive_exchange(2, rank, P, send_matrix, recv_matrix_naive);
    MPI_Barrier(MPI_COMM_WORLD);
    
    t_naive_end = MPI_Wtime();

    // --- Handshake Benchmarking ---
    MPI_Barrier(MPI_COMM_WORLD);
    t_hs_start = MPI_Wtime();
    
    // Phase 1: Odd diff
    handshake_exchange(1, rank, P, send_matrix, recv_matrix_handshake);
    MPI_Barrier(MPI_COMM_WORLD);
    
    // Phase 2: Even diff
    handshake_exchange(2, rank, P, send_matrix, recv_matrix_handshake);
    MPI_Barrier(MPI_COMM_WORLD);
    
    t_hs_end = MPI_Wtime();

    // Print recv array for verification (truncated for brevity)
    printf("Rank %d received data (Handshake): ", rank);
    for (int j = 0; j < P; ++j) {
        if (rank != j) printf("[%d:%d] ", j, recv_matrix_handshake[rank][j]);
    }
    printf("\n");

    MPI_Barrier(MPI_COMM_WORLD);

    if (rank == 0) {
        printf("\n+-------------------------------------------------------+\n");
        printf("|       MPI Deadlock-Avoiding Message Exchange          |\n");
        printf("+-------------------------------------------------------+\n");
        printf("| Method             | Execution Time (s)               |\n");
        printf("+--------------------+----------------------------------+\n");
        printf("| Pure Blocking      | %10.6f                     |\n", t_naive_end - t_naive_start);
        printf("| Handshake-Based    | %10.6f                     |\n", t_hs_end - t_hs_start);
        printf("+--------------------+----------------------------------+\n");
        printf("| Result: DEADLOCK-FREE COMPLETION                      |\n");
        printf("+-------------------------------------------------------+\n");
    }

    MPI_Finalize();
    return 0;
}
