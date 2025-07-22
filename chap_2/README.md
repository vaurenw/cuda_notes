### Chapter 2: Heterogeneous Data Parallel Computing

**2.1 Data Parallelism**

*   **Core Idea:** Performing computation on independent parts of data concurrently.
*   **Example:** Color to grayscale conversion, where each pixel's calculation is separate: \( L = r \cdot 0.21 + g \cdot 0.72 + b \cdot 0.07 \).
*   **Goal:** Reorganize computation around data for parallel execution.
*   **Data vs. Task Parallelism:** Data parallelism scales well with large datasets; task parallelism involves independent tasks.

**2.2 CUDA C Program Structure**

*   **Purpose:** Extends C for CPU + GPU programming.
*   **Components:**
    *   **Host Code:** Runs on the CPU.
    *   **Device Code (Kernels):** Runs on the GPU in a parallel grid of threads.
*   **Execution Flow:** Host calls a kernel -> launches threads on GPU -> threads execute -> return to host when done.

**2.3 A Vector Addition Kernel**

*   **Host Version:** A standard C loop: `for (i=0; i<n; ++i) C_h[i] = A_h[i] + B_h[i];`
*   **CUDA Version Steps:**
    1.  Allocate device memory (for A, B, C).
    2.  Copy data (A, B) from host to device.
    3.  **Launch Kernel:** Execute parallel addition on the device.
    4.  Copy results (C) from device to host.
    5.  Free device memory.
*   **Naming Convention:** `_h` for host variables, `_d` for device variables (e.g., `A_h`, `A_d`).

**2.4 Device Global Memory & Data Transfer**

*   **Memory Allocation/Deallocation:**
    *   `cudaMalloc((void **)&ptr, size);` (allocate on device)
    *   `cudaFree(ptr);` (free on device)
*   **Data Transfer:**
    *   `cudaMemcpy(dst, src, size, cudaMemcpyHostToDevice);` (host -> device)
    *   `cudaMemcpy(dst, src, size, cudaMemcpyDeviceToHost);` (device -> host)

**2.5 Kernel Functions and Threading**

*   **Function Qualifiers:**
    *   `__global__`: Kernel (runs on device, called from host).
    *   `__device__`: Device function (runs on device, called from device).
    *   `__host__`: Host function (runs on host, called from host).
*   **Built-in Thread Identifiers:**
    *   `threadIdx.{x,y,z}`: Index within a block.
    *   `blockIdx.{x,y,z}`: Index of the block in the grid.
    *   `blockDim.{x,y,z}`: Size of a block.
*   **Global Thread Index (1D):** `i = blockIdx.x * blockDim.x + threadIdx.x;`
*   **Kernel Execution:** Threads execute the same `__global__` function. Use `if (i < n)` to handle vector lengths not evenly divisible by block size.

**2.6 Calling a Kernel**

*   **Syntax:** `myKernel<<< numBlocks, threadsPerBlock >>>( arguments );`
*   **`numBlocks` Calculation:** Use ceiling division to ensure enough blocks: `ceil(total_elements / threadsPerBlock)`.

**2.7 Compilation**

*   **Compiler:** NVCC (NVIDIA CUDA Compiler).
*   **Process:** NVCC separates host code (compiled by C/C++ compiler) and device code (compiled to PTX, then GPU binary). Creates a single executable.

**2.8 Summary of Key CUDA C Extensions**

*   **Function Declarations:** `__global__`, `__device__`, `__host__` keywords.
*   **Kernel Launch:** `<<< >>>` syntax for configuration parameters.
*   **Thread/Block Indexing:** Built-in variables `threadIdx`, `blockIdx`, `blockDim`.
*   **Runtime APIs:** `cudaMalloc`, `cudaFree`, `cudaMemcpy` for memory and data management.

---

An example CUDA C code for vector addition:

First, here's the complete code:

```cuda
// Host code (runs on CPU)
#include <stdio.h>
#include <stdlib.h>

// Kernel definition (runs on GPU)
// __global__ indicates this function will run on the device
// and is called from the host.
__global__ void vectorAddKernel(const float* A, const float* B, float* C, int n) {
    // Calculate the global thread index.
    // threadIdx.x: index of the thread within its block
    // blockIdx.x: index of the block within the grid
    // blockDim.x: number of threads in each block
    // This formula maps each thread to a unique element in the vectors.
    int i = blockIdx.x * blockDim.x + threadIdx.x;

    // Ensure that the thread index 'i' is within the bounds of the vector.
    // This is important because the number of threads launched might be more
    // than the actual number of elements in the vector.
    if (i < n) {
        C[i] = A[i] + B[i];
    }
}

int main() {
    // --- Host setup ---
    int n = 1000000; // Number of elements in the vectors
    size_t size = n * sizeof(float); // Size of vectors in bytes

    // Allocate host memory for vectors A, B, and C
    float *h_A = (float*) malloc(size);
    float *h_B = (float*) malloc(size);
    float *h_C = (float*) malloc(size);

    // Initialize host vectors A and B with some values
    for (int i = 0; i < n; ++i) {
        h_A[i] = (float)i;
        h_B[i] = (float)i * 2.0f;
    }

    // --- Device memory allocation and data transfer ---
    float *d_A, *d_B, *d_C; // Pointers for device memory

    // Allocate device memory
    cudaMalloc((void**)&d_A, size);
    cudaMalloc((void**)&d_B, size);
    cudaMalloc((void**)&d_C, size);

    // Copy vectors from host memory to device memory
    // cudaMemcpyHostToDevice indicates the direction of the copy.
    cudaMemcpy(d_A, h_A, size, cudaMemcpyHostToDevice);
    cudaMemcpy(d_B, h_B,                       size, cudaMemcpyHostToDevice);

    // --- Kernel launch configuration ---
    // Define the dimensions of the grid and blocks.
    // A block size of 256 threads is common and often efficient.
    int threadsPerBlock = 256;
    // Calculate the number of blocks needed.
    // We use ceil(n / threadsPerBlock) to ensure all elements are processed.
    // The calculation `(n + threadsPerBlock - 1) / threadsPerBlock` is an integer
    // arithmetic way to achieve the ceiling division.
    int numBlocks = (n + threadsPerBlock - 1) / threadsPerBlock;

    // --- Kernel launch ---
    // Call the kernel: vectorAddKernel<<< numBlocks, threadsPerBlock >>>(d_A, d_B, d_C, n);
    // The <<<...>>> syntax specifies the execution configuration (grid and block dimensions).
    // This launches 'numBlocks' blocks, each containing 'threadsPerBlock' threads.
    vectorAddKernel<<<numBlocks, threadsPerBlock>>>(d_A, d_B, d_C, n);

    // Wait for the kernel to complete. cudaDeviceSynchronize() ensures all
    // GPU operations launched so far have finished.
    cudaDeviceSynchronize();

    // --- Data transfer back and cleanup ---
    // Copy the result vector C from device memory back to host memory.
    cudaMemcpy(h_C, d_C, size, cudaMemcpyDeviceToHost);

    // Free device memory
    cudaFree(d_A);
    cudaFree(d_B);
    cudaFree(d_C);

    // Free host memory
    free(h_A);
    free(h_B);
    free(h_C);

    // --- Verification (Optional) ---
    // Check if the results are correct
    int errors = 0;
    for (int i = 0; i < n; ++i) {
        if (h_C[i] != (float)i + (float)i * 2.0f) {
            errors++;
            if (errors < 10) { // Print first few errors
                printf("Error at index %d: Expected %f, Got %f\n", i, (float)i + (float)i * 2.0f, h_C[i]);
            }
        }
    }

    if (errors == 0) {
        printf("Vector addition successful!\n");
    } else {
        printf("Vector addition failed with %d errors.\n", errors);
    }

    return 0;
}
```

---

### Explanation of the Code

This example demonstrates a complete CUDA C program for performing vector addition ( \( C = A + B \) ) where vectors A and B are on the host (CPU) and the computation is offloaded to the device (GPU).

**1. Header Files and Includes:**

*   `#include <stdio.h>`: Standard input/output for `printf`.
*   `#include <stdlib.h>`: For memory allocation (`malloc`, `free`) and `exit`.
*   `#include <cuda_runtime.h>`: This is crucial. It provides access to the CUDA API functions (like `cudaMalloc`, `cudaMemcpy`, `cudaFree`, `cudaDeviceSynchronize`) and CUDA-specific types and keywords.

**2. Kernel Definition (`__global__ void vectorAddKernel(...)`)**

*   **`__global__` Keyword:**
    *   This is a CUDA function qualifier.
    *   It signifies that this function is a *kernel*.
    *   Kernels are executed on the **device** (GPU).
    *   They are called from the **host** (CPU).
    *   When a `__global__` function is called, it launches a *grid* of *blocks*, and each block contains multiple *threads*.

*   **Function Signature:**
    *   `void vectorAddKernel(const float* A, const float* B, float* C, int n)`
    *   `A`, `B`: Pointers to the input vectors on the **device**. They are marked `const` because the kernel only reads from them.
    *   `C`: Pointer to the output vector on the **device**. The kernel will write to this.
    *   `n`: The number of elements in the vectors.

*   **Calculating the Global Thread Index (`int i = blockIdx.x * blockDim.x + threadIdx.x;`)**
    *   This is the heart of data parallelism here. Each thread in the grid needs to know which element of the vector it should process.
    *   `threadIdx.x`: The index of the current thread within its specific block. If a block has 256 threads, their `threadIdx.x` values will range from 0 to 255.
    *   `blockIdx.x`: The index of the current block within the entire grid. If there are 100 blocks, their `blockIdx.x` values will range from 0 to 99.
    *   `blockDim.x`: The number of threads within each block (e.g., 256).
    *   The formula `blockIdx.x * blockDim.x` calculates the starting index for the block. Adding `threadIdx.x` then gives a unique, global index `i` for each thread across the entire grid. This ensures each thread processes a different element.

*   **Boundary Check (`if (i < n)`)**
    *   We launch threads in blocks. The total number of threads launched might be slightly more than the number of elements (`n`) in the vector, especially if `n` isn't perfectly divisible by the `threadsPerBlock`.
    *   This `if` statement is crucial. It ensures that only threads with a valid index `i` (i.e., `i` less than `n`) actually perform the addition. Threads with `i >= n` do nothing, preventing out-of-bounds access.

*   **The Computation (`C[i] = A[i] + B[i];`)**
    *   Inside the `if` block, each thread `i` reads the `i`-th elements from device vectors `A` and `B`, adds them, and stores the result in the `i`-th element of device vector `C`.

**3. `main()` Function (Host Code)**

*   **Host Memory Setup:**
    *   `int n = 1000000;`: Defines the size of the vectors.
    *   `size_t size = n * sizeof(float);`: Calculates the total memory needed for each vector in bytes.
    *   `float *h_A = (float*) malloc(size);` etc.: Standard C `malloc` is used to allocate memory for the vectors `A`, `B`, and `C` on the **host** (CPU's RAM). The `h_` prefix denotes host memory.
    *   **Initialization:** The host vectors `h_A` and `h_B` are filled with sample data. `h_A[i]` gets `i`, and `h_B[i]` gets `i * 2.0f`.

*   **Device Memory Allocation:**
    *   `float *d_A, *d_B, *d_C;`: Declares pointers that will point to memory on the **device** (GPU). The `d_` prefix denotes device memory.
    *   `cudaMalloc((void**)&d_A, size);` etc.: These are CUDA API calls.
        *   `cudaMalloc` allocates a specified `size` (in bytes) in the **device's** global memory.
        *   The first argument is a pointer to a pointer (`(void**)&d_A`). This is because `cudaMalloc` needs to modify the pointer variable (`d_A`) itself to store the address of the allocated device memory.
        *   `void**` is used because `cudaMalloc` is a generic function that can allocate memory for any data type.

*   **Data Transfer (Host to Device):**
    *   `cudaMemcpy(d_A, h_A, size, cudaMemcpyHostToDevice);` etc.: These are CUDA API calls for data copying.
        *   `cudaMemcpy` takes four arguments: destination pointer, source pointer, number of bytes to copy, and the direction of the transfer.
        *   `cudaMemcpyHostToDevice`: Specifies that data is being copied from the host's memory (`h_A`) to the device's memory (`d_A`).
    *   This step is critical for making the data available to the GPU for computation.

*   **Kernel Launch Configuration:**
    *   `int threadsPerBlock = 256;`: Defines how many threads will be in each block. 256 is a common choice, often aligned with GPU architecture.
    *   `int numBlocks = (n + threadsPerBlock - 1) / threadsPerBlock;`: Calculates the number of blocks needed. This is a standard integer arithmetic trick for ceiling division: if `n` is 1000 and `threadsPerBlock` is 256, it computes `(1000 + 256 - 1) / 256 = 1255 / 256 = 4` (integer division). This ensures enough blocks to cover all `n` elements.

*   **Kernel Launch:**
    *   `vectorAddKernel<<<numBlocks, threadsPerBlock>>>(d_A, d_B, d_C, n);`
    *   This is how you *invoke* a `__global__` kernel from the host.
    *   The `<<<numBlocks, threadsPerBlock>>>` syntax (known as the execution configuration or launch configuration) specifies:
        *   The first parameter (`numBlocks`): The number of blocks to launch in the grid.
        *   The second parameter (`threadsPerBlock`): The number of threads per block.
    *   The arguments `(d_A, d_B, d_C, n)` are passed to the `vectorAddKernel` function on the device.

*   **Synchronization:**
    *   `cudaDeviceSynchronize();`: This call is important. GPU operations (like kernel launches) are asynchronous by default. This means the CPU can continue executing host code immediately after launching a kernel, without waiting for the GPU to finish.
    *   `cudaDeviceSynchronize()` forces the CPU to wait until all previously issued CUDA operations (including the `vectorAddKernel` launch) have completed on the GPU. This is necessary here to ensure the `vectorAddKernel` has finished before we attempt to copy the results back.

*   **Data Transfer (Device to Host):**
    *   `cudaMemcpy(h_C, d_C, size, cudaMemcpyDeviceToHost);`: Copies the computed results from the device vector `d_C` back to the host vector `h_C`. The `cudaMemcpyDeviceToHost` direction is specified.

*   **Cleanup:**
    *   `cudaFree(d_A);` etc.: Releases the memory previously allocated on the device using `cudaMalloc`. It's good practice to free device memory when it's no longer needed.
    *   `free(h_A);` etc.: Frees the memory allocated on the host using `malloc`.

*   **Verification (Optional):**
    *   A simple loop checks if the computed values in `h_C` are correct (i.e., `h_C[i]` should equal `h_A[i] + h_B[i]`). This helps confirm the parallel computation worked as expected.
