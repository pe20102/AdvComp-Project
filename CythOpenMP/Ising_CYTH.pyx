# distutils: language = c++
# distutils: extra_compile_args = -fopenmp
# distutils: extra_link_args = -fopenmp

import numpy as np
cimport numpy as np
from cython.parallel import prange
from libc.stdlib cimport rand, srand
from libc.math cimport exp
from libc.time cimport clock, CLOCKS_PER_SEC
cdef extern from "limits.h":
    int RAND_MAX
cdef extern from "omp.h":
    void omp_set_num_threads(int num_threads)
    int omp_get_num_threads()
    int omp_get_max_threads()

#=======================================================================
def initdat(int nmax):
    """
    Initialize a square lattice with spins -1 or +1.
    """
    cdef np.ndarray[np.int8_t, ndim=2] lattice = np.empty((nmax, nmax), dtype=np.int8)
    lattice[:, :] = np.random.choice([-1, 1], size=(nmax, nmax))
    return lattice

#=======================================================================
def plotdat(np.ndarray[np.int8_t, ndim=2] lattice, int pflag, int nmax):
    """
    Visualize the lattice. Use pflag to control style:
        pflag = 0 for no plot;
        pflag = 1 for grayscale plot.
    """
    if pflag == 0:
        return
    import matplotlib.pyplot as plt
    plt.imshow(lattice, cmap='gray', vmin=-1, vmax=1)
    plt.colorbar()
    plt.title('Ising Model Lattice')
    plt.show()

#=======================================================================
cdef void savedat(np.ndarray[np.int8_t, ndim=2] lattice, int nsteps, double T, double runtime, 
                  list accept_ratios, int nmax):
    """
    Save the lattice and metadata to a file.
    """
    import datetime
    current_datetime = datetime.datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    filename = f"Ising_Output_{current_datetime}.txt"
    with open(filename, "w") as f:
        f.write(f"# Ising Model Simulation\n")
        f.write(f"# Lattice size: {nmax}x{nmax}\n")
        f.write(f"# Temperature: {T}\n")
        f.write(f"# Steps: {nsteps}\n")
        f.write(f"# Runtime: {runtime:.6f} s\n")
        f.write(f"# Acceptance ratios: {accept_ratios}\n")
        f.write("# Final lattice configuration:\n")
        np.savetxt(f, lattice, fmt="%d")

#=======================================================================
cdef double mc_move_parallel(np.ndarray[np.int8_t, ndim=2] lattice, int nmax, double J, double h, double inv_T):
    """
    Perform one Monte Carlo step using Metropolis-Hastings with OpenMP for parallel processing.
    """
    cdef int accept = 0
    cdef int i, j, ipp, jpp, inn, jnn, spin, neighbors
    cdef double deltaE, rand_val

    # Parallelized Monte Carlo loop
    for i in prange(nmax, nogil=True, schedule='dynamic'):
        for j in range(nmax):
            ipp = (i + 1) % nmax
            jpp = (j + 1) % nmax
            inn = (i - 1) % nmax
            jnn = (j - 1) % nmax

            neighbors = (
                lattice[ipp, j] +
                lattice[inn, j] +
                lattice[i, jpp] +
                lattice[i, jnn]
            )

            spin = lattice[i, j]
            deltaE = 2 * spin * (J * neighbors + h)
            rand_val = rand() / <double>RAND_MAX

            if deltaE <= 0 or rand_val < exp(-deltaE * inv_T):
                lattice[i, j] *= -1
                accept += 1

    return accept / (nmax * nmax)

#=======================================================================
def run_simulation(int nsteps, int nmax, double T, double J, double h, int pflag, int num_threads=4):
    """
    Run the Ising model simulation using OpenMP for parallelism.
    
    Parameters:
        nsteps: Number of Monte Carlo steps.
        nmax: Size of the lattice.
        T: Temperature.
        J: Interaction constant.
        h: External field.
        pflag: Plot flag.
        num_threads: Number of threads to use for OpenMP.
    """
    import tracemalloc
    tracemalloc.start()  # Start tracking memory usage

    # Set the number of threads for OpenMP
    omp_set_num_threads(num_threads)

    # Debug the number of threads
    print(f"Using {num_threads} threads for simulation (max available: {omp_get_max_threads()}).")

    srand(<unsigned int> clock())  # Seed RNG
    cdef np.ndarray[np.int8_t, ndim=2] lattice = initdat(nmax)
    cdef double inv_T = 1.0 / T
    cdef list accept_ratios = []
    cdef double start_time, end_time, runtime
    
    # Variables for Binder cumulant
    cdef double m2_sum = 0.0
    cdef double m4_sum = 0.0
    cdef double m, m2, m4
    cdef int total_spins = nmax * nmax

    # Plot initial lattice
    plotdat(lattice, pflag, nmax)

    # Measure high-resolution time
    start_time = clock()
    for step in range(nsteps):
        accept_ratio = mc_move_parallel(lattice, nmax, J, h, inv_T)
        accept_ratios.append(accept_ratio)
        
        # Calculate magnetization and update sums for Binder cumulant
        m = np.sum(lattice) / float(total_spins)  # Magnetization per spin
        m2 = m * m
        m4 = m2 * m2
        m2_sum += m2
        m4_sum += m4
        
    end_time = clock()

    runtime = (end_time - start_time) / CLOCKS_PER_SEC  # Convert to seconds
    
    # Calculate flips per nanosecond
    cdef double flips_per_ns = (total_spins * nsteps / runtime) * 1e-9
    
    # Calculate Binder cumulant
    cdef double m2_avg = m2_sum / nsteps
    cdef double m4_avg = m4_sum / nsteps
    cdef double binder_cumulant = 1.0 - m4_avg / (3.0 * m2_avg * m2_avg) if m2_avg != 0 else 0.0

    # Debug memory usage
    current, peak = tracemalloc.get_traced_memory()
    print(f"Current memory usage: {current / 1e6:.2f} MB; Peak memory usage: {peak / 1e6:.2f} MB")
    tracemalloc.stop()

    # Save final lattice
    savedat(lattice, nsteps, T, runtime, accept_ratios, nmax)

    # Plot final lattice
    plotdat(lattice, pflag, nmax)
    print(f"Simulation complete. Runtime: {runtime:.6f} seconds")
    print(f"Performance: {flips_per_ns:.2f} flips/ns")
    print(f"Binder Cumulant: {binder_cumulant:.6f}")
