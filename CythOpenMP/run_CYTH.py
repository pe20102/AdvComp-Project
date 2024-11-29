"""
Run file for Ising model simulation using Cython-optimized code with multi-threading
"""

import sys
from Ising_CYTH import run_simulation

if __name__ == "__main__":
    if len(sys.argv) == 8:
        try:
            # Get command line arguments
            iterations = int(sys.argv[1])
            size = int(sys.argv[2])
            temperature = float(sys.argv[3])
            J = float(sys.argv[4])
            H = float(sys.argv[5])
            plotflag = int(sys.argv[6])
            num_threads = int(sys.argv[7])  # New argument for number of threads
            
            # Input validation
            if size <= 0 or iterations <= 0:
                raise ValueError("Size and iterations must be positive")
            if temperature <= 0:
                raise ValueError("Temperature must be positive")
            if plotflag not in [0, 1]:
                raise ValueError("Plot flag must be 0 or 1")
            if num_threads <= 0:
                raise ValueError("Number of threads must be positive")
            
            # Run simulation and measure time
            import time
            start_time = time.perf_counter()
            run_simulation(iterations, size, temperature, J, H, plotflag, num_threads)
            end_time = time.perf_counter()

            # Calculate total runtime
            total_time = end_time - start_time  # Total time in seconds
            
            # Calculate flips per nanosecond
            lattice_size = size * size
            flips_per_ns = ((lattice_size * iterations) / total_time) * 1e-9

            print(f"Flips per nanosecond: {flips_per_ns:.6f} flips/ns.")

        except ValueError as e:
            print(f"Error: {e}")
            print("\nUsage: python run_ising.py <ITERATIONS> <SIZE> <TEMPERATURE> <J> <H> <PLOTFLAG> <NUM_THREADS>")
            print("  ITERATIONS: positive integer (number of Monte Carlo steps)")
            print("  SIZE: positive integer (lattice size)")
            print("  TEMPERATURE: positive float (temperature in reduced units)")
            print("  J: float (interaction constant)")
            print("  H: float (external field)")
            print("  PLOTFLAG: 0 (no plot) or 1 (show plot)")
            print("  NUM_THREADS: positive integer (number of threads to use)")
            sys.exit(1)
    else:
        print("\nUsage: python run_ising.py <ITERATIONS> <SIZE> <TEMPERATURE> <J> <H> <PLOTFLAG> <NUM_THREADS>")
        print("  ITERATIONS: positive integer (number of Monte Carlo steps)")
        print("  SIZE: positive integer (lattice size)")
        print("  TEMPERATURE: positive float (temperature in reduced units)")
        print("  J: float (interaction constant)")
        print("  H: float (external field)")
        print("  PLOTFLAG: 0 (no plot) or 1 (show plot)")
        print("  NUM_THREADS: positive integer (number of threads to use)")
        sys.exit(1)
