import sys
import time
import datetime
import numpy as np
import matplotlib.pyplot as plt

#=======================================================================
def initdat(nmax):
    """
    Arguments:
      nmax (int) = size of lattice to create (nmax,nmax).
    Description:
      Initialize a square lattice (size nmax x nmax) with spins -1 or +1.
    Returns:
      arr (int(nmax,nmax)) = lattice array.
    """
    arr = np.random.choice([-1, 1], size=(nmax, nmax))
    return arr
#=======================================================================
def plotdat(arr, pflag, nmax):
    """
    Arguments:
      arr (int(nmax,nmax)) = lattice array;
      pflag (int) = parameter to control plotting;
      nmax (int) = size of lattice.
    Description:
      Function to visualize the lattice. Use pflag to control style:
        pflag = 0 for no plot;
        pflag = 1 for grayscale plot.
    Returns:
      NULL
    """
    if pflag == 0:
        return
    plt.imshow(arr, cmap='gray', vmin=-1, vmax=1)
    plt.colorbar()
    plt.title('Ising Model Lattice')
    plt.show()
#=======================================================================
def mc_move_vectorized(lattice, nmax, J, h, inv_T):
    """
    Arguments:
      lattice (int(nmax, nmax)) = lattice array;
      nmax (int) = size of lattice;
      J (float) = interaction constant;
      h (float) = external magnetic field;
      inv_T (float) = inverse temperature (1/T).
    Description:
      Perform one Monte Carlo step using a vectorized checkerboard update.
    Returns:
      accept_ratio (float) = fraction of accepted spin flips.
    """
    # Checkerboard pattern masks
    checkerboard1 = np.indices((nmax, nmax)).sum(axis=0) % 2 == 0
    checkerboard2 = ~checkerboard1

    accept = 0
    for mask in [checkerboard1, checkerboard2]:
        # Extract neighbors with periodic boundary conditions
        neighbors = (
            np.roll(lattice, 1, axis=0) +
            np.roll(lattice, -1, axis=0) +
            np.roll(lattice, 1, axis=1) +
            np.roll(lattice, -1, axis=1)
        )
        
        # Calculate energy change for the masked spins
        deltaE = 2 * lattice[mask] * (J * neighbors[mask] + h)
        
        # Decide which spins to flip
        flip_mask = (deltaE <= 0) | (np.random.rand(*deltaE.shape) < np.exp(-deltaE * inv_T))
        
        # Flip spins
        lattice[mask] *= np.where(flip_mask, -1, 1)
        accept += flip_mask.sum()
    
    return accept / (nmax**2)
#=======================================================================
def savedat(arr, nsteps, T, runtime, accept_ratios, nmax):
    """
    Arguments:
      arr (int(nmax,nmax)) = lattice array;
      nsteps (int) = number of Monte Carlo steps;
      T (float) = temperature;
      runtime (float) = simulation time;
      accept_ratios (list) = acceptance ratios per step;
      nmax (int) = lattice size.
    Description:
      Save the lattice and metadata to a file.
    Returns:
      NULL
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
        np.savetxt(f, arr, fmt="%d")
#=======================================================================
def calculate_binder_cumulant(lattice, nmax):
    """
    Arguments:
      lattice (int(nmax,nmax)) = lattice array;
      nmax (int) = size of lattice.
    Description:
      Calculate the Binder cumulant U = 1 - <m^4>/(3<m^2>^2)
      where m is the magnetization per spin.
    Returns:
      binder (float) = Binder cumulant value.
    """
    m = np.sum(lattice) / (nmax * nmax)  # Magnetization per spin
    m2 = m * m
    m4 = m2 * m2
    return 1.0 - m4 / (3.0 * m2 * m2) if m2 != 0 else 0

def main(program, nsteps, nmax, T, J, h, pflag):
    """
    Arguments:
      program (string) = name of program;
      nsteps (int) = number of Monte Carlo steps;
      nmax (int) = lattice size;
      T (float) = temperature;
      J (float) = interaction constant;
      h (float) = external field;
      pflag (int) = plot flag.
    Description:
      Main function for the Ising model simulation.
    Returns:
      NULL
    """
    # Initialize lattice
    lattice = initdat(nmax)
    plotdat(lattice, pflag, nmax)
    
    accept_ratios = []
    inv_T = 1.0 / T  # Precompute inverse temperature
    start_time = time.time()
    
    # Arrays to store magnetization values for Binder cumulant
    m2_sum = 0.0
    m4_sum = 0.0
    
    for step in range(nsteps):
        accept_ratio = mc_move_vectorized(lattice, nmax, J, h, inv_T)
        accept_ratios.append(accept_ratio)
        
        # Calculate magnetization and update sums for Binder cumulant
        m = np.sum(lattice) / (nmax * nmax)
        m2_sum += m * m
        m4_sum += m * m * m * m
    
    end_time = time.time()
    runtime = end_time - start_time
    
    # Calculate flips per nanosecond and Binder cumulant
    flips_per_ns = ((nmax * nmax * nsteps) / runtime) * 1e-9
    
    # Calculate final Binder cumulant
    m2_avg = m2_sum / nsteps
    m4_avg = m4_sum / nsteps
    binder_cumulant = 1.0 - m4_avg / (3.0 * m2_avg * m2_avg) if m2_avg != 0 else 0

    # Save data and display results
    savedat(lattice, nsteps, T, runtime, accept_ratios, nmax)
    plotdat(lattice, pflag, nmax)
    print(f"Simulation complete. Runtime: {runtime:.4f} s")
    print(f"Performance: {flips_per_ns:.9f} flips/ns")
    print(f"Binder Cumulant: {binder_cumulant:.6f}")
#=======================================================================
if __name__ == "__main__":
    if len(sys.argv) == 7:
        PROGNAME = sys.argv[0]
        NSTEPS = int(sys.argv[1])
        SIZE = int(sys.argv[2])
        TEMPERATURE = float(sys.argv[3])
        J = float(sys.argv[4])
        H = float(sys.argv[5])
        PLOTFLAG = int(sys.argv[6])
        main(PROGNAME, NSTEPS, SIZE, TEMPERATURE, J, H, PLOTFLAG)
    else:
        print("Usage: python IsingModel.py <NSTEPS> <SIZE> <TEMPERATURE> <J> <H> <PLOTFLAG>")
