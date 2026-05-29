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

## Quick Start (Windows)

> This is a step-by-step guide to get the programs running on Windows using VS Code's integrated terminal (PowerShell).

### Step 1: Install MS-MPI (Two Separate Installers)

Download **both** of the following from the [MS-MPI releases page](https://github.com/microsoft/Microsoft-MPI/releases):

| Installer | What it does | File |
|-----------|-------------|------|
| **MS-MPI Runtime** | Provides `mpiexec.exe` to launch parallel processes | `msmpisetup.exe` |
| **MS-MPI SDK** | Provides headers (`mpi.h`) and libraries (`msmpi.lib`) needed to compile MPI programs | `msmpisdk.msi` |

> ⚠️ **You must install both.** The runtime alone lets you run MPI programs but not compile them. The SDK alone lets you compile but not run.

After installation, verify by opening a **new** PowerShell terminal and running:

```powershell
# This should print the MS-MPI version (e.g., 10.1.3)
& "C:\Program Files\Microsoft MPI\Bin\mpiexec.exe" --version
```

### Step 2: Install a C++ Compiler

You need `g++` (GCC). If you don't have it, install **MinGW-w64** via [WinLibs](https://winlibs.com/) or through `winget`:

```powershell
winget install BrechtSanders.WinLibs.POSIX.UCRT
```

Verify it works:

```powershell
g++ --version
```

> If `g++` is not recognized, you need to add its `bin` folder to your system PATH. See [this guide](https://www.architectryan.com/2018/03/17/add-to-the-windows-path-environment-variable/).

### Step 3: Clone the Repository

Open VS Code, press `` Ctrl+` `` to open the integrated terminal, then run:

```powershell
git clone https://github.com/moosarehan/mpi-distributed-parallelism.git
cd mpi-distributed-parallelism
```

### Step 4: Compile

Run these commands in the VS Code terminal (PowerShell). The default MS-MPI SDK install paths are used below — adjust if yours differ:

```powershell
g++ -O2 -o tree_collectives.exe tree_collectives.cpp -I"C:\Program Files (x86)\Microsoft SDKs\MPI\Include" "C:\Program Files (x86)\Microsoft SDKs\MPI\Lib\x64\msmpi.lib"

g++ -O2 -o deadlock_avoidance.exe deadlock_avoidance.cpp -I"C:\Program Files (x86)\Microsoft SDKs\MPI\Include" "C:\Program Files (x86)\Microsoft SDKs\MPI\Lib\x64\msmpi.lib"

g++ -O2 -o dist_priority_queue.exe dist_priority_queue.cpp -I"C:\Program Files (x86)\Microsoft SDKs\MPI\Include" "C:\Program Files (x86)\Microsoft SDKs\MPI\Lib\x64\msmpi.lib"
```

### Step 5: Run

```powershell
# Tree-Based Collectives (8 processes, binary tree, root=0, array size 10000)
& "C:\Program Files\Microsoft MPI\Bin\mpiexec.exe" -n 8 ./tree_collectives.exe 2 0 10000

# Deadlock-Avoiding Message Exchange (8 processes)
& "C:\Program Files\Microsoft MPI\Bin\mpiexec.exe" -n 8 ./deadlock_avoidance.exe

# Distributed Priority Queue (8 processes, 1000 tasks per worker)
& "C:\Program Files\Microsoft MPI\Bin\mpiexec.exe" -n 8 ./dist_priority_queue.exe 1000
```

> 💡 **Why the full path for `mpiexec`?** On Windows, MS-MPI may not add itself to your system PATH automatically. Using the full path `"C:\Program Files\Microsoft MPI\Bin\mpiexec.exe"` guarantees it works regardless of your PATH configuration. The `&` at the start is PowerShell syntax for running executables with spaces in the path.

---

## Quick Start (Linux / macOS)

### Install MPI

```bash
# Ubuntu / Debian
sudo apt install openmpi-bin libopenmpi-dev g++

# macOS (Homebrew)
brew install open-mpi gcc
```

### Clone, Compile, and Run

```bash
git clone https://github.com/moosarehan/mpi-distributed-parallelism.git
cd mpi-distributed-parallelism

# Compile
mpicxx -O2 -o tree_collectives tree_collectives.cpp
mpicxx -O2 -o deadlock_avoidance deadlock_avoidance.cpp
mpicxx -O2 -o dist_priority_queue dist_priority_queue.cpp

# Run
mpiexec -n 8 ./tree_collectives 2 0 10000
mpiexec -n 8 ./deadlock_avoidance
mpiexec -n 8 ./dist_priority_queue 1000
```

---

## Command-Line Arguments

### Tree-Based Collectives

```
mpiexec -n <P> ./tree_collectives <fan_out> <root_rank> <array_size>
```

| Argument | Description | Constraints | Example |
|----------|-------------|-------------|---------|
| `P` | Number of MPI processes | ≥ 8 | `8` |
| `fan_out` | Tree branching factor | ≥ 2 | `2` |
| `root_rank` | Root process rank | 0 ≤ root < P | `0` |
| `array_size` | Elements in the data array | > 0 | `10000` |

### Distributed Priority Queue

```
mpiexec -n <P> ./dist_priority_queue <tasks_per_worker>
```

| Argument | Description | Constraints | Example |
|----------|-------------|-------------|---------|
| `P` | Number of MPI processes | ≥ 2 (1 root + workers) | `8` |
| `tasks_per_worker` | Tasks each worker sends to root | Rounded down to multiple of 5 | `1000` |

### Deadlock-Avoiding Message Exchange

```
mpiexec -n <P> ./deadlock_avoidance
```

| Argument | Description | Constraints | Example |
|----------|-------------|-------------|---------|
| `P` | Number of MPI processes | ≥ 8 | `8` |

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

### Deadlock-Avoiding Message Exchange
```
+-------------------------------------------------------+
|       MPI Deadlock-Avoiding Message Exchange          |
+-------------------------------------------------------+
| Method             | Execution Time (s)               |
+--------------------+----------------------------------+
| Pure Blocking      |   0.002622                       |
| Handshake-Based    |   0.022754                       |
+--------------------+----------------------------------+
| Result: DEADLOCK-FREE COMPLETION                      |
+-------------------------------------------------------+
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

## Troubleshooting & FAQs

### 1. ⚠️ Can I just click the "Run/Play" button in VS Code?
**No.** Default C++ runners in VS Code (like the Code Runner extension) compile programs without linking the MS-MPI SDK libraries (`msmpi.lib` and headers `mpi.h`). More importantly, they run programs as standard sequential applications. To run a parallel MPI program, it **must** be launched via `mpiexec`. Always use the terminal compilation and run commands specified in the **Quick Start** section.

### 2. ❌ Error: `mpi.h: No such file or directory` or `msmpi.lib` not found
This happens if the compiler cannot find the MS-MPI SDK headers or library files.
- **Fix:** Double check that you installed the **MS-MPI SDK** (`msmpisdk.msi`) and not just the runtime.
- **Fix:** If you installed MS-MPI to a custom directory, update the paths in the `-I` (Include) and library path arguments in your `g++` compilation command:
  ```powershell
  g++ -O2 -o tree_collectives.exe tree_collectives.cpp -I"<Your-Custom-Path>\Include" "<Your-Custom-Path>\Lib\x64\msmpi.lib"
  ```

### 3. ❌ Error: `g++` or `mpiexec` is not recognized
This means the folder containing the executable is not in your system's Environment `PATH`.
- **For `g++`:** Ensure your MinGW/WinLibs `bin` directory is added to your environment `PATH` (e.g., `C:\winlibs-...\bin`). Restart your terminal or VS Code after modifying environment variables.
- **For `mpiexec`:** We bypass this issue in the **Quick Start** guide by invoking the absolute path directly: `& "C:\Program Files\Microsoft MPI\Bin\mpiexec.exe"`. If you prefer to type just `mpiexec`, add `"C:\Program Files\Microsoft MPI\Bin"` to your system `PATH`.

### 4. 🛡️ Windows Defender Firewall alert popped up
When you run `mpiexec` for the first time, Windows Defender Firewall may ask for permission.
- **What to do:** Allow it. MPI relies on TCP/IP network sockets for processes to communicate with each other, even when they are running on the same local computer. If you block it, MPI processes will not be able to exchange messages.

### 5. ❌ Error: `P must be >= 8` (or similar constraints)
The programs are specifically optimized to demonstrate distributed parallelism and deadlock avoidance at scale.
- **Fix:** Ensure you pass `-n 8` (or greater) to `mpiexec` when launching `tree_collectives.exe` and `deadlock_avoidance.exe`. Running them with fewer processes will trigger a validation safeguard and abort execution.

---

## License

This project is open source and available for educational and research purposes.

