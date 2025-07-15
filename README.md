DATA LOADING STAGE
Load decaps.mat: This file contains the admittance profiles for a set of decoupling capacitors. Each row represents one capacitor, and each column corresponds to a specific frequency point. The data is likely complex-valued, capturing both the real and imaginary parts of admittance. This information is crucial for building accurate models of the power distribution network's behavior at different frequencies.
Load y.mat: This file stores the base admittance matrix of the Power Distribution Network (PDN). The matrix dimensions suggest a multi-port network, where each port represents a possible connection point for decoupling capacitors. The third dimension indicates that the admittance matrix is frequency-dependent, reflecting the PDN's varying behavior across the frequency spectrum.
Load freq.mat: This file contains a vector of frequencies. The admittance data in 'y.mat' and 'decaps.mat' is organized according to these frequencies. This vector allows for frequency sweeps and analysis of the PDN's performance at specific frequencies of interest.
INITIALIZATION
Define PSO parameters:
Np (number of particles): This sets the population size for the Particle Swarm Optimization (PSO) algorithm. Each particle represents a potential solution (a configuration of decoupling capacitors).
maxiter (maximum iterations): This determines the maximum number of iterations the PSO algorithm will run.
Define optimization parameters:
ports: This specifies the number of ports available on the PDN for placing decoupling capacitors.
zt (target impedance): This is the desired target impedance for the PDN. The optimization aims to find a capacitor configuration that achieves an impedance close to this target.
wf (final inertia), wi (initial inertia): These parameters control the "inertia" of PSO particles, influencing how much their velocity is affected by their previous velocity and the best-known positions.
c1, c2 (PSO coefficients): These coefficients balance the influence of the particle's own best position and the swarm's global best position on the particle's velocity update.
e_th (RBFN threshold): This threshold is used in the Radial Basis Function Network (RBFN) surrogate model to determine when to bypass the actual fitness calculation and rely on the RBFN prediction instead.
Initialize global trackers:
best_values_per_noc: This array stores the best impedance value achieved for each number of capacitors (NOC) considered.
overall_best_impedance: This variable tracks the overall best impedance found across all NOC values.
overall_best_noc: This variable stores the NOC that resulted in the overall best impedance.
total_time: This variable records the total execution time of the optimization process.
OUTER LOOP: For noc = 1 to ports

This loop iterates over different numbers of capacitors (NOC) to find the optimal NOC that achieves the target impedance.

Initial RBFN Database Construction
Generate Id = 100 random valid configurations: For each NOC value, Id random configurations of capacitors are generated. Each configuration consists of:
noc random capacitor indices: These indices select specific capacitors from the 'decaps.mat' file.
noc random port indices: These indices specify the PDN ports where the selected capacitors will be placed. The indices must be unique to avoid placing multiple capacitors on the same port.
For each configuration:
Call calculate_fitness(X, noc): This function evaluates the fitness of a given configuration X.
Loop over f = 1 to fmax: For each frequency in the 'freq.mat' vector:
Build Yeq: Construct the equivalent admittance matrix by combining the base PDN admittance (from 'y.mat') with the admittance of the selected capacitors at their assigned ports.
Zeq = inv(Yeq): Calculate the equivalent impedance matrix by inverting the admittance matrix.
Z = abs(Zeq(1,1)): Extract the magnitude of the impedance at the input port.
zmax = max(Z) over all f: Find the maximum impedance magnitude across all frequencies. This zmax value represents the fitness of the configuration.
Sort all Id configs by zmax: The configurations are sorted based on their zmax values.
Select top Np configs → Initialize swarm: The top Np configurations (those with the lowest zmax values) are selected to initialize the PSO swarm.
Save them as RBFN_centers and RBFN_values: These configurations and their fitness values are stored to build the initial RBFN surrogate model.
Compute D_max and derive RBFN spread: D_max, the maximum distance between RBFN centers, is calculated. This value is then used to determine the RBFN spread, which controls the shape of the RBFN kernel functions.
Swarm Initialization
X = top Np configs; Xl = X (local bests): The particle positions X are initialized with the top Np configurations. The local best positions Xl are also initialized with the same values.
globalbestval = min(zmax): The global best fitness value is set to the minimum zmax among the initial swarm.
Xg = best config from X, repeated Np times: The global best position Xg is initialized with the best configuration from the initial swarm, repeated Np times.
vel = zeros(Np, 2*ports): The particle velocities are initialized to zero.
INNER LOOP: For iter = 1 to maxiter

This loop iterates the PSO algorithm for a maximum of 'maxiter' iterations.
w = wf + (wi - wf) * (maxiter - iter) / maxiter: The inertia weight w is updated at each iteration, decreasing linearly from wi to wf.
FOR i = 1 to Np: Loop over each particle in the swarm.
Velocity Update:
The velocity of each particle is updated based on its current velocity, its distance from its local best position, and its distance from the global best position. Random values r1 and r2 are used to introduce stochasticity.
Position Update:
The position of each particle is updated by adding its velocity.
The new position values are rounded to the nearest integers and clamped to ensure they represent valid capacitor and port indices.
Any port conflicts (multiple capacitors assigned to the same port) are resolved.
Surrogate Fitness Evaluation:
The fitness of the new position is predicted using the Gaussian RBFN surrogate model.
If the predicted fitness (plus the threshold e_th) is greater than or equal to the particle's local best fitness, the actual fitness calculation is skipped.
Otherwise:
The actual fitness is calculated using the 'calculate_fitness' function.
If the new fitness is better than the particle's local best, the local best position and fitness are updated.
If the new fitness is better than the global best, the global best position and fitness are updated.
The new configuration and fitness are added to the RBFN database.
D_max and the RBFN spread are recomputed.
Update globalbestarray(iter) = globalbestval: The global best fitness value at each iteration is stored.
IF globalbestval < zt: If the global best fitness meets the target impedance, the inner loop is terminated early.
After Iteration Loop for This NOC
The best impedance value for this NOC is stored in 'best_values_per_noc'.
The impedance vs. frequency curve for the best configuration is calculated and plotted.
RBFN statistics (number of RBFN uses and number of real fitness evaluations) are logged.
If the global best fitness meets the target impedance, the outer loop is terminated early, and the final solution (capacitor and port indices) is saved.
FINAL OUTPUT AND DISPLAY
Print:
Optimal NOC
Best impedance
Final configuration (capacitor indices, port indices)
Total RBFN vs. real evaluation count
Total execution time
Plot:
Overlay of impedance vs. frequency curves for all valid NOC runs
Global best progression curves per NOC

