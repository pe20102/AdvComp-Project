import sys
import time
import datetime
import numpy as np
import matplotlib.pyplot as plt
import os
os.environ["MKL_NUM_THREADS"] = "1"
os.environ["NUMEXPR_NUM_THREADS"] = "1"
os.environ["OMP_NUM_THREADS"] = "1"
from mpi4py import MPI

MAXWORKER = 15         # maximum number of worker tasks
MINWORKER = 1          # minimum number of worker tasks
BEGIN = 1              # message tag
DONE = 2              # message tag
ATAG = 3              # message tag
BTAG = 4              # message tag
NONE = 0              # indicates no neighbour
MASTER = 0            # taskid of first process

def initdat(lattice_size):
    """
    Arguments:
      lattice_size (int) = size of lattice to create (lattice_size,lattice_size).
    Description:
      Initialize a square lattice with spins -1 or +1.
    Returns:
      lattice (int(lattice_size,lattice_size)) = array to hold lattice.
    """
    return np.random.choice([-1, 1], size=(lattice_size, lattice_size)).astype(np.int32)

def plotdat(spins, energies, plot_flag, lattice_size, filename=None):
    """
    Arguments:
      spins (int(lattice_size,lattice_size)) = array that contains lattice spins;
      energies (float(lattice_size,lattice_size)) = array that contains lattice energies;
      plot_flag (int) = parameter to control plotting;
      lattice_size (int) = side length of square lattice;
      filename (str) = if provided, save plot to this filename.
    Description:
      Function to plot the Ising lattice state.
    """
    if plot_flag == 0:
        return

    plt.figure(figsize=(8, 8))
    plt.imshow(spins, cmap='binary', interpolation='nearest')
    plt.colorbar(label='Spin')
    plt.title('Ising Model Lattice')
    plt.xlabel('x')
    plt.ylabel('y')
    
    if filename:
        plt.savefig(filename, dpi=300, bbox_inches='tight')
        plt.close()
    else:
        plt.show()
    plt.close()

def block_energy(num_rows, spins, energies, J, h):
    """Calculate block energy for the Ising lattice"""
    for ix in range(1, num_rows+1):
        for jx in range(lattice_size):
            spin = spins[ix, jx]
            # Calculate neighbors with periodic boundary conditions
            neighbors = (
                spins[ix+1, jx] +
                spins[ix-1, jx] +
                spins[ix, (jx+1)%lattice_size] +
                spins[ix, (jx-1)%lattice_size]
            )
            energies[ix-1, jx] = -J * spin * neighbors - h * spin

def MC_substep(comm, spins, energies, temperature, lattice_size, num_rows, 
               above_rank, below_rank, acceptance_ratio, step, J, h):
    """Perform a Monte Carlo substep for Ising model"""
    
    # Communicate border rows with neighbours
    req = comm.Isend([spins[1,:], lattice_size, MPI.INT], dest=above_rank, tag=ATAG)
    req = comm.Isend([spins[num_rows,:], lattice_size, MPI.INT], dest=below_rank, tag=BTAG)
    comm.Recv([spins[0,:], lattice_size, MPI.INT], source=above_rank, tag=BTAG)
    comm.Recv([spins[num_rows+1,:], lattice_size, MPI.INT], source=below_rank, tag=ATAG)
    
    # Calculate initial energies
    block_energy(num_rows, spins, energies[0], J, h)
    
    # Attempt spin flips at random sites
    num_attempts = num_rows * lattice_size  # Try to flip each spin once on average
    for _ in range(num_attempts):
        # Randomly select a site in this block
        ix = np.random.randint(1, num_rows + 1)
        jx = np.random.randint(0, lattice_size)
        
        # Calculate energy change for flip
        old_energy = energies[0][ix-1, jx]
        spins[ix, jx] *= -1  # Flip spin
        
        # Recalculate energy with flipped spin
        spin = spins[ix, jx]
        neighbors = (
            spins[ix+1, jx] +
            spins[ix-1, jx] +
            spins[ix, (jx+1)%lattice_size] +
            spins[ix, (jx-1)%lattice_size]
        )
        new_energy = -J * spin * neighbors - h * spin
        
        # Accept or reject flip using Metropolis criterion
        deltaE = new_energy - old_energy
        if deltaE <= 0 or np.random.random() < np.exp(-deltaE / temperature):
            energies[1][ix-1, jx] = new_energy
            acceptance_ratio[step] += 1.0/(num_attempts)
        else:
            spins[ix, jx] *= -1  # Undo flip
            energies[1][ix-1, jx] = old_energy

def MC_step(comm, spins, energies, temperature, lattice_size, num_rows, offset, 
            above_rank, below_rank, total_energy, magnetization, acceptance_ratio, step, J, h):
    """Perform a full Monte Carlo step"""
    
    energies = np.zeros((2, num_rows, lattice_size))
    
    # Perform Monte Carlo updates
    MC_substep(comm, spins, energies, temperature, lattice_size, num_rows, 
               above_rank, below_rank, acceptance_ratio, step, J, h)
    
    # Update system properties
    total_energy[step] = np.sum(energies[1])
    magnetization[step] = np.sum(spins[1:num_rows+1, :])/(lattice_size**2)
    
    return spins, energies, total_energy, magnetization, acceptance_ratio

def calculate_binder_cumulant(magnetization):
    """
    Calculate the Binder cumulant from magnetization values
    U = 1 - <m^4>/(3<m^2>^2)
    """
    m2_avg = np.mean(magnetization**2)
    m4_avg = np.mean(magnetization**4)
    if m2_avg == 0:  # Avoid division by zero
        return 0
    return 1.0 - m4_avg/(3.0 * m2_avg**2)

def main(program_name, num_steps, lattice_size, temperature, J, h, plot_flag):
    """Main function for the Ising model simulation"""
    
    # Initialize MPI
    comm = MPI.COMM_WORLD
    taskid = comm.Get_rank()
    numtasks = comm.Get_size()
    numworkers = numtasks-1

    # Create arrays for system properties
    total_energy = np.zeros(num_steps)
    magnetization = np.zeros(num_steps)
    acceptance_ratio = np.zeros(num_steps)

    if taskid == MASTER:
        # Check if numworkers is within range
        if (numworkers > MAXWORKER) or (numworkers < MINWORKER):
            print("ERROR: the number of tasks must be between %d and %d." % (MINWORKER+1,MAXWORKER+1))
            print("Quitting...")
            comm.Abort()

        # Initialize grid and energies
        spins = initdat(lattice_size)
        energies = np.zeros((lattice_size,lattice_size), dtype=np.float64)

        # Plot and save initial state
        current_datetime = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
        initial_filename = f"ising_initial.png"
        plotdat(spins, energies, plot_flag, lattice_size, initial_filename)
        print(f"Initial lattice saved as '{initial_filename}'")

        # Distribute work to workers
        averow = lattice_size//numworkers
        extra = lattice_size%numworkers
        offset = 0

        initial_time = MPI.Wtime()
        for i in range(1, numworkers+1):
            rows = averow + (1 if i <= extra else 0)
            above_rank = numworkers if i == 1 else i - 1
            below_rank = 1 if i == numworkers else i + 1

            comm.send(offset, dest=i, tag=BEGIN)
            comm.send(rows, dest=i, tag=BEGIN)
            comm.send(above_rank, dest=i, tag=BEGIN)
            comm.send(below_rank, dest=i, tag=BEGIN)
            comm.Send([spins[offset:offset+rows,:], rows*lattice_size, MPI.INT], dest=i, tag=BEGIN)
            offset += rows

        # Temporary arrays for collecting results
        temp_energy = np.zeros(num_steps)
        temp_mag = np.zeros(num_steps)
        temp_acceptance = np.zeros(num_steps)
        
        # Collect results from workers
        for i in range(1, numworkers+1):
            offset = comm.recv(source=i, tag=DONE)
            rows = comm.recv(source=i, tag=DONE)
            temp_spins = np.empty((rows, lattice_size), dtype=np.int32)
            comm.Recv([temp_spins, rows*lattice_size, MPI.INT], source=i, tag=DONE)
            spins[offset:offset+rows,:] = temp_spins
            
            temp_energies = np.empty((rows, lattice_size), dtype=np.float64)
            comm.Recv([temp_energies, rows*lattice_size, MPI.DOUBLE], source=i, tag=DONE)
            energies[offset:offset+rows,:] = temp_energies
            
            comm.Recv([temp_energy, num_steps, MPI.DOUBLE], source=i, tag=DONE)
            comm.Recv([temp_mag, num_steps, MPI.DOUBLE], source=i, tag=DONE)
            comm.Recv([temp_acceptance, num_steps, MPI.DOUBLE], source=i, tag=DONE)
            
            total_energy += temp_energy
            magnetization += temp_mag
            acceptance_ratio += temp_acceptance

        # Calculate timing and final output
        final_time = MPI.Wtime()
        runtime = final_time-initial_time
        
        # Calculate flips per nanosecond
        total_attempts = lattice_size * lattice_size * num_steps
        flips_per_ns = (total_attempts / runtime) * 1e-9

        # Calculate Binder cumulant
        binder_cumulant = calculate_binder_cumulant(magnetization)

        print(f"{program_name}: Size: {lattice_size}, Steps: {num_steps}, T*: {temperature:5.3f}, "
              f"J: {J:5.3f}, h: {h:5.3f}, M: {magnetization[-1]:5.3f}, Time: {runtime:8.6f} s")
        print(f"Performance: {flips_per_ns:.7f} flips/ns")
        print(f"Binder Cumulant: {binder_cumulant:.6f}")
        
        # Plot and save final state
        final_filename = "ising_final.png"
        plotdat(spins, energies, plot_flag, lattice_size, final_filename)
        print(f"Final lattice saved as '{final_filename}'")

    else:
        # Worker process code
        offset = comm.recv(source=MASTER, tag=BEGIN)
        num_rows = comm.recv(source=MASTER, tag=BEGIN)
        above_rank = comm.recv(source=MASTER, tag=BEGIN)
        below_rank = comm.recv(source=MASTER, tag=BEGIN)

        spins = np.zeros((num_rows+2, lattice_size), dtype=np.int32)
        energies = np.zeros((2, num_rows, lattice_size), dtype=np.float64)
        
        # Receive initial spins
        comm.Recv([spins[1:num_rows+1,:], num_rows*lattice_size, MPI.INT], source=MASTER, tag=BEGIN)

        for step in range(num_steps):
            spins, energies, total_energy, magnetization, acceptance_ratio = MC_step(
                comm, spins, energies, temperature, lattice_size, num_rows, 
                offset, above_rank, below_rank, total_energy, magnetization, 
                acceptance_ratio, step, J, h)
            
        # Send results back to master
        comm.send(offset, dest=MASTER, tag=DONE)
        comm.send(num_rows, dest=MASTER, tag=DONE)
        comm.Send([spins[1:num_rows+1,:].astype(np.int32), num_rows*lattice_size, MPI.INT], dest=MASTER, tag=DONE)
        comm.Send([energies[1,:,:].astype(np.float64), num_rows*lattice_size, MPI.DOUBLE], dest=MASTER, tag=DONE)
        comm.Send([total_energy, num_steps, MPI.DOUBLE], dest=MASTER, tag=DONE)
        comm.Send([magnetization, num_steps, MPI.DOUBLE], dest=MASTER, tag=DONE)
        comm.Send([acceptance_ratio, num_steps, MPI.DOUBLE], dest=MASTER, tag=DONE)

if __name__ == '__main__':
    if int(len(sys.argv)) == 7:
        program_name = sys.argv[0]
        num_iterations = int(sys.argv[1])
        lattice_size = int(sys.argv[2])
        temperature = float(sys.argv[3])
        J = float(sys.argv[4])
        h = float(sys.argv[5])
        plot_flag = int(sys.argv[6])
        main(program_name, num_iterations, lattice_size, temperature, J, h, plot_flag)
    else:
        print("Usage: python {} <ITERATIONS> <SIZE> <TEMPERATURE> <J> <H> <PLOTFLAG>".format(sys.argv[0]))