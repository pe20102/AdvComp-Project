import sys
import time
import numpy as np
import matplotlib.pyplot as plt

#=======================================================================
def init_checkerboard_lattice(nmax):
    """
    Arguments:
      nmax (int) = size of lattice (nmax x nmax).
    Description:
      Initializes two sub-lattices (black and white) for a checkerboard pattern.
    Returns:
      black_lattice (int(nmax, nmax // 2)),
      white_lattice (int(nmax, nmax // 2)).
    """
    black_lattice = np.random.choice([-1, 1], size=(nmax, nmax // 2))
    white_lattice = np.random.choice([-1, 1], size=(nmax, nmax // 2))
    return black_lattice, white_lattice
#=======================================================================
def combine_checkerboard_lattice(black, white, nmax):
    """
    Arguments:
      black (int(nmax, nmax // 2)) = black sub-lattice;
      white (int(nmax, nmax // 2)) = white sub-lattice;
      nmax (int) = lattice size.
    Description:
      Combines black and white sub-lattices into a full lattice.
    Returns:
      full_lattice (int(nmax, nmax)).
    """
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
def mc_move_checkerboard(black, white, nmax, J, h, inv_T, is_black):
    """
    Arguments:
      black (int(nmax, nmax // 2)) = black sub-lattice;
      white (int(nmax, nmax // 2)) = white sub-lattice;
      nmax (int) = lattice size;
      J (float) = interaction constant;
      h (float) = external magnetic field;
      inv_T (float) = inverse temperature (1/T);
      is_black (bool) = True if updating black lattice, False otherwise.
    Description:
      Performs a single Monte Carlo step on one sub-lattice.
    Returns:
      updated_lattice (int(nmax, nmax // 2)).
    """
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
            
            if is_black:
                nn_sum = (
                    opposite[inn, j] +
                    opposite[ipp, j] +
                    opposite[i, jnn] +
                    opposite[i, jpp]
                )
            else:
                nn_sum = (
                    opposite[inn, j] +
                    opposite[ipp, j] +
                    opposite[i, jnn] +
                    opposite[i, jpp]
                )

            spin = target[i, j]
            deltaE = 2 * spin * (J * nn_sum + h)
            if deltaE <= 0 or np.random.rand() < np.exp(-deltaE * inv_T):
                target[i, j] *= -1
                accept += 1

    return target, accept / (nmax * (nmax // 2))
#=======================================================================
def plot_checkerboard(black, white, nmax):
    """
    Arguments:
      black (int(nmax, nmax // 2)) = black sub-lattice;
      white (int(nmax, nmax // 2)) = white sub-lattice;
      nmax (int) = lattice size.
    Description:
      Plots the combined lattice.
    Returns:
      NULL
    """
    full_lattice = combine_checkerboard_lattice(black, white, nmax)
    plt.imshow(full_lattice, cmap='gray', vmin=-1, vmax=1)
    plt.colorbar()
    plt.title("Checkerboard Lattice")
    plt.show()
#=======================================================================
def main(nsteps, nmax, T, J, h):
    """
    Arguments:
      nsteps (int) = number of Monte Carlo steps;
      nmax (int) = lattice size;
      T (float) = temperature;
      J (float) = interaction constant;
      h (float) = external magnetic field.
    Description:
      Main function for the checkerboard Ising model simulation.
    Returns:
      NULL
    """
    black, white = init_checkerboard_lattice(nmax)
    inv_T = 1.0 / T

    plot_checkerboard(black, white, nmax)
    start_time = time.time()

    for step in range(nsteps):
        black, _ = mc_move_checkerboard(black, white, nmax, J, h, inv_T, True)
        white, _ = mc_move_checkerboard(white, black, nmax, J, h, inv_T, False)

    end_time = time.time()
    runtime = end_time - start_time
    flips_per_ns = ((nmax * nmax * nsteps) / runtime) * 1e-9

    plot_checkerboard(black, white, nmax)
    print(f"Simulation complete. Runtime: {runtime:.4f} s")
    print(f"Flips per nanosecond: {flips_per_ns:.9f}")
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
