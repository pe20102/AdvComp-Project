import sys
import time
import numpy as np
import matplotlib.pyplot as plt
from numba import njit

#=======================================================================
def init_checkerboard_lattice(nmax):
    """Initialize two sub-lattices (black and white) for a checkerboard pattern."""
    black_lattice = np.random.choice([-1, 1], size=(nmax, nmax // 2))
    white_lattice = np.random.choice([-1, 1], size=(nmax, nmax // 2))
    return black_lattice, white_lattice

#=======================================================================
@njit
def combine_checkerboard_lattice(black, white, nmax):
    """Combines black and white sub-lattices into a full lattice."""
    full_lattice = np.zeros((nmax, nmax), dtype=np.int8)
    for i in range(nmax):
        for j in range(nmax // 2):
            if i % 2 == 0:
                full_lattice[i, 2*j] = black[i, j]
                full_lattice[i, 2*j + 1] = white[i, j]
            else:
                full_lattice[i, 2*j] = white[i, j]
                full_lattice[i, 2*j + 1] = black[i, j]
    return full_lattice

#=======================================================================
@njit
def mc_move_checkerboard(black, white, nmax, J, h, inv_T, is_black):
    """Performs a single Monte Carlo step on one sub-lattice."""
    target = black if is_black else white
    opposite = white if is_black else black
    accept = 0
    
    for i in range(nmax):
        for j in range(nmax // 2):
            # Calculate neighbors with periodic boundary conditions
            ipp = (i + 1) % nmax
            inn = (i - 1) % nmax
            jpp = (j + 1) % (nmax // 2)
            jnn = (j - 1) % (nmax // 2)
            
            nn_sum = (
                opposite[inn, j] +
                opposite[ipp, j] +
                opposite[i, jnn] +
                opposite[i, jpp]
            )

            spin = target[i, j]
            deltaE = 2 * spin * (J * nn_sum + h)
            
            if deltaE <= 0 or np.random.random() < np.exp(-deltaE * inv_T):
                target[i, j] *= -1
                accept += 1

    return target, accept / (nmax * (nmax // 2))

#=======================================================================
def plot_checkerboard(black, white, nmax):
    """Plots the combined lattice."""
    full_lattice = combine_checkerboard_lattice(black, white, nmax)
    plt.imshow(full_lattice, cmap='gray', vmin=-1, vmax=1)
    plt.colorbar()
    plt.title("Checkerboard Lattice")
    plt.show()

#=======================================================================
@njit
def calculate_magnetization(black, white, nmax):
    """Calculate magnetization for Binder cumulant."""
    total_spins = nmax * nmax
    total_magnetization = np.sum(black) + np.sum(white)
    return total_magnetization / total_spins

#=======================================================================
def main(nsteps, nmax, T, J, h):
    """Main function for the checkerboard Ising model simulation."""
    black, white = init_checkerboard_lattice(nmax)
    inv_T = 1.0 / T
    
    # For Binder cumulant
    m2_sum = 0.0
    m4_sum = 0.0

    plot_checkerboard(black, white, nmax)
    start_time = time.time()

    for step in range(nsteps):
        black, _ = mc_move_checkerboard(black, white, nmax, J, h, inv_T, True)
        white, _ = mc_move_checkerboard(white, black, nmax, J, h, inv_T, False)
        
        # Calculate magnetization for Binder cumulant
        m = calculate_magnetization(black, white, nmax)
        m2_sum += m * m
        m4_sum += m * m * m * m

    end_time = time.time()
    runtime = end_time - start_time
    flips_per_ns = ((nmax * nmax * nsteps) / runtime) * 1e-9
    
    # Calculate Binder cumulant
    m2_avg = m2_sum / nsteps
    m4_avg = m4_sum / nsteps
    binder_cumulant = 1.0 - m4_avg / (3.0 * m2_avg * m2_avg) if m2_avg != 0 else 0

    plot_checkerboard(black, white, nmax)
    print(f"Simulation complete. Runtime: {runtime:.4f} s")
    print(f"Performance: {flips_per_ns:.9f} flips/ns")
    print(f"Binder Cumulant: {binder_cumulant:.6f}")

#=======================================================================
if __name__ == "__main__":
    if len(sys.argv) == 6:
        NSTEPS = int(sys.argv[1])
        SIZE = int(sys.argv[2])
        TEMPERATURE = float(sys.argv[3])
        J = float(sys.argv[4])
        H = float(sys.argv[5])
        main(NSTEPS, SIZE, TEMPERATURE, J, H)
    else:
        print("Usage: python CheckerboardModel.py <NSTEPS> <SIZE> <TEMPERATURE> <J> <H>")
