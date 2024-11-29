import sys
import time
import datetime
import cupy as cp
import matplotlib.pyplot as plt

#=======================================================================
def initdat(nmax):
    """
    Initialize a square lattice (size nmax x nmax) with spins -1 or +1.
    """
    arr = cp.random.choice([-1, 1], size=(nmax, nmax)).astype(cp.int8)
    return arr

#=======================================================================
def plotdat(arr, pflag, nmax):
    """
    Visualize the lattice. Use pflag to control style.
    """
    if pflag == 0:
        return
    plt.imshow(cp.asnumpy(arr), cmap='gray', vmin=-1, vmax=1)
    plt.colorbar()
    plt.title('Ising Model Lattice')
    plt.show()

#=======================================================================
def mc_move(lattice, nmax, J, h, inv_T, threads_per_block):
    """
    Perform one Monte Carlo step using a checkerboard update and thread-safe random states.
    """
    kernel_code = r"""
    extern "C" {
    __device__ int mod(int a, int b) {
        return (a % b + b) % b;
    }

    __global__ void mc_move_kernel(
        signed char* lattice, int nmax, float J, float h, float inv_T, int* accept, int checkerboard, unsigned long long seed
    ) {
        int idx = blockIdx.x * blockDim.x + threadIdx.x;
        if (idx >= nmax * nmax) return;

        int i = idx / nmax;
        int j = idx % nmax;

        // Checkerboard condition
        if ((i + j) % 2 != checkerboard) return;

        // Periodic boundary conditions
        int ipp = mod(i + 1, nmax);
        int inn = mod(i - 1, nmax);
        int jpp = mod(j + 1, nmax);
        int jnn = mod(j - 1, nmax);

        int neighbors = lattice[ipp * nmax + j] +
                        lattice[inn * nmax + j] +
                        lattice[i * nmax + jpp] +
                        lattice[i * nmax + jnn];

        int spin = lattice[i * nmax + j];
        float deltaE = 2.0f * spin * (J * neighbors + h);

        // Thread-safe random number generation
        unsigned long long state = seed + idx;
        state ^= state >> 21; state ^= state << 35; state ^= state >> 4;
        state *= 2685821657736338717ull;
        float rand_val = (state & 0xFFFFFF) / (float)(0x1000000);

        // Metropolis-Hastings acceptance
        if (deltaE <= 0 || rand_val < expf(-deltaE * inv_T)) {
            lattice[i * nmax + j] = -spin;
            atomicAdd(accept, 1);
        }
    }
    }
    """

    # Compile the kernel
    module = cp.RawModule(code=kernel_code)
    kernel = module.get_function("mc_move_kernel")

    # Prepare GPU data
    accept = cp.zeros(1, dtype=cp.int32)
    blocks_per_grid = (nmax * nmax + threads_per_block - 1) // threads_per_block
    seed = cp.random.randint(1, 2**32, dtype=cp.uint64)

    # First pass: Update "black" cells
    kernel(
        (blocks_per_grid,), (threads_per_block,),
        (lattice, cp.int32(nmax), cp.float32(J), cp.float32(h), cp.float32(inv_T), accept, cp.int32(0), seed)
    )

    # Second pass: Update "white" cells
    kernel(
        (blocks_per_grid,), (threads_per_block,),
        (lattice, cp.int32(nmax), cp.float32(J), cp.float32(h), cp.float32(inv_T), accept, cp.int32(1), seed)
    )

    # Return the acceptance ratio
    return accept.get() / (nmax**2)



#=======================================================================
def savedat(arr, nsteps, T, runtime, accept_ratios, nmax):
    """
    Save the lattice and metadata to a file.
    """
    current_datetime = datetime.datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    filename = f"Ising_Output_{current_datetime}.txt"
    with open(filename, "w") as f:
        f.write(f"# Ising Model Simulation\n")
        f.write(f"# Lattice size: {nmax}x{nmax}\n")
        f.write(f"# Temperature: {T}\n")
        f.write(f"# Steps: {nsteps}\n")
        f.write(f"# Runtime: {runtime:.4f} s\n")
        f.write(f"# Acceptance ratios: {accept_ratios}\n")
        f.write("# Final lattice configuration:\n")
        cp.savetxt(f, cp.asnumpy(arr), fmt="%d")

#=======================================================================
def main(program, nsteps, nmax, T, J, h, pflag, threads_per_block):
    """
    Main function for the Ising model simulation.
    """
    lattice = initdat(nmax)
    plotdat(lattice, pflag, nmax)

    accept_ratios = []
    inv_T = 1.0 / T  # Precompute inverse temperature
    start_time = time.time()

    for step in range(nsteps):
        accept_ratio = mc_move(lattice, nmax, J, h, inv_T, threads_per_block)
        accept_ratios.append(accept_ratio)

    end_time = time.time()
    runtime = end_time - start_time
    flips_per_ns = ((nmax * nmax * nsteps) / runtime) * 1e-9

    savedat(lattice, nsteps, T, runtime, accept_ratios, nmax)
    plotdat(lattice, pflag, nmax)
    print(f"Simulation complete. Runtime: {runtime:.4f} s")
    print(f"Performance: {flips_per_ns:.9f} flips/ns")

#=======================================================================
if __name__ == "__main__":
    if len(sys.argv) == 8:
        PROGNAME = sys.argv[0]
        NSTEPS = int(sys.argv[1])
        SIZE = int(sys.argv[2])
        TEMPERATURE = float(sys.argv[3])
        J = float(sys.argv[4])
        H = float(sys.argv[5])
        PLOTFLAG = int(sys.argv[6])
        THREADS_PER_BLOCK = int(sys.argv[7])
        main(PROGNAME, NSTEPS, SIZE, TEMPERATURE, J, H, PLOTFLAG, THREADS_PER_BLOCK)
    else:
        print("Usage: python IsingModel.py <NSTEPS> <SIZE> <TEMPERATURE> <J> <H> <PLOTFLAG> <THREADS_PER_BLOCK>")
