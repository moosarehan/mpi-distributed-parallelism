# MPI Distributed Parallelism

A suite of optimized C++ programs demonstrating advanced distributed parallelism patterns using MPI (Message Passing Interface). Features custom tree-based collective communications, deadlock-free concurrent message exchanges, and a high-throughput parallel task queue.

---

## Programs

### 1. Hierarchical Tree-Based Collectives (`tree_collectives.cpp`)

Implements and benchmarks **k-ary tree-based broadcast and reduction** against naive flat approaches.

- Processes are organized into a logical k-ary tree with configurable fan-out
- Root rank is remapped via modular arithmetic so any process can serve as root
- Correctness is validated against `MPI_Reduce` with floating-point tolerance checks

**Key Insight:** Tree-based collectives distribute network bandwidth across all nodes, enabling superior scaling on large multi-node clusters where flat broadcasts saturate the root's network link.

### 2. Deadlock-Avoiding Message Exchange (`deadlock_avoidance.cpp`)

Implements a **two-phase pairwise message exchange** protocol that avoids circular waits without relying on MPI's internal buffering.

- **Phase 1:** Exchanges between processes with an odd rank difference
- **Phase 2:** Exchanges between processes with an even rank difference
- Compares a rank-ordered blocking approach against a synchronous handshake protocol (RTS/CTS)

**Key Insight:** Rank-based send/receive ordering breaks the circular wait condition, which is the fundamental cause of MPI deadlocks in blocking communication.

### 3. Concurrent Distributed Priority Queue (`dist_priority_queue.cpp`)

Implements a **master-worker priority queue** where multiple workers send tasks to a root process that maintains a min-heap. Compares three collection strategies:

| Method | Description |
|--------|-------------|
| **Blocking** | Root calls `MPI_Recv` sequentially for each task |
| **Non-blocking (Waitany)** | Root posts `MPI_Irecv` per worker and uses `MPI_Waitany` for CPU-efficient event-driven collection |
| **Blocking + Batching** | Workers batch multiple tasks into a single message to reduce per-message overhead |

**Key Insight:** `MPI_Waitany` eliminates CPU-intensive active polling loops by suspending the thread until any worker delivers a task, achieving significantly higher throughput than manual `MPI_Test` polling.

---

## Prerequisites

You need two things installed on your system:

1. **An MPI Implementation**
   - **Windows:** [Microsoft MPI (MS-MPI)](https://learn.microsoft.com/en-us/message-passing-interface/microsoft-mpi) — install both the runtime and the SDK
   - **Linux:** OpenMPI (`sudo apt install openmpi-bin libopenmpi-dev`) or MPICH (`sudo apt install mpich`)
   - **macOS:** OpenMPI via Homebrew (`brew install open-mpi`)

2. **A C++ Compiler**
   - `g++` (GCC), `clang++`, or MSVC with MPI support
   - The compiler must be able to locate MPI headers and libraries

---

## Compilation

### Linux / macOS (using MPI compiler wrapper)

```bash
mpicxx -O2 -o tree_collectives tree_collectives.cpp
mpicxx -O2 -o deadlock_avoidance deadlock_avoidance.cpp
mpicxx -O2 -o dist_priority_queue dist_priority_queue.cpp
```

### Windows (using g++ with MS-MPI)

```powershell
g++ -O2 -o tree_collectives.exe tree_collectives.cpp -I"<MPI_SDK_Include_Path>" "<MPI_SDK_Lib_Path>\msmpi.lib"
g++ -O2 -o deadlock_avoidance.exe deadlock_avoidance.cpp -I"<MPI_SDK_Include_Path>" "<MPI_SDK_Lib_Path>\msmpi.lib"
g++ -O2 -o dist_priority_queue.exe dist_priority_queue.cpp -I"<MPI_SDK_Include_Path>" "<MPI_SDK_Lib_Path>\msmpi.lib"
```

> **Note:** Replace `<MPI_SDK_Include_Path>` and `<MPI_SDK_Lib_Path>` with your actual MS-MPI SDK paths. Typical defaults:
> - Include: `C:\Program Files (x86)\Microsoft SDKs\MPI\Include`
> - Lib: `C:\Program Files (x86)\Microsoft SDKs\MPI\Lib\x64`

---

## Usage

### Tree-Based Collectives

```bash
mpiexec -n <num_processes> ./tree_collectives <fan_out> <root_rank> <array_size>
```

| Argument | Description | Example |
|----------|-------------|---------|
| `num_processes` | Total MPI processes (≥ 8) | `8` |
| `fan_out` | Tree branching factor (≥ 2) | `2` |
| `root_rank` | Root process rank | `0` |
| `array_size` | Size of the data array | `10000` |

```bash
# Example: 8 processes, binary tree, root=0, array of 10000 doubles
mpiexec -n 8 ./tree_collectives 2 0 10000
```

### Deadlock-Avoiding Message Exchange

```bash
mpiexec -n <num_processes> ./deadlock_avoidance
```

```bash
# Example: 8 processes performing pairwise deadlock-free exchange
mpiexec -n 8 ./deadlock_avoidance
```

### Distributed Priority Queue

```bash
mpiexec -n <num_processes> ./dist_priority_queue <tasks_per_worker>
```

| Argument | Description | Example |
|----------|-------------|---------|
| `num_processes` | Total MPI processes (workers + 1 root) | `8` |
| `tasks_per_worker` | Number of tasks each worker sends | `1000` |

```bash
# Example: 7 workers each sending 1000 tasks to root
mpiexec -n 8 ./dist_priority_queue 1000
```

---

## Sample Output

### Tree Collectives
```
Tree Collectives: CORRECT
Tree Broadcast Verification: CORRECT

+--------------------------------------------------------------+
|       Hierarchical Tree-Based Collective Operations          |
+--------------------------------------------------------------+
|  Configuration:                                              |
|    Processes (P) = 8       Fan-out (k) = 2                   |
|    Root          = 0       Array size  = 10000               |
+-------------------------+--------------+------------+--------+
|  Operation              |  Naive (s)   |  Tree (s)  | Speedup|
+-------------------------+--------------+------------+--------+
|  Broadcast              |     0.009185 |   0.021702 |  0.42x |
|  Reduction (Sum)        |     0.020219 |   0.018968 |  1.07x |
+-------------------------+--------------+------------+--------+
```

### Distributed Priority Queue
```
+-------------------------------------------------------------+
|       Distributed Priority Queue Performance Analysis       |
+-------------------------------------------------------------+
| Method                     | Time (s)    | Throughput (t/s) |
+----------------------------+-------------+------------------+
| Blocking Collection        |    0.002448 |       2859944.42 |
| Non-blocking + Waitany     |    0.004514 |       1550628.07 |
| Blocking + Batching        |    0.001890 |       3704683.19 |
+----------------------------+-------------+------------------+
```

---

## License

This project is open source and available for educational and research purposes.
