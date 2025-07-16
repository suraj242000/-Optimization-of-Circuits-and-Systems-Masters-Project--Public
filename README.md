# ⚡ PDN Optimization Using PSO with RBFN Surrogates

This MATLAB project implements a Particle Swarm Optimization (PSO) algorithm with Radial Basis Function Network (RBFN) surrogate modeling to optimize the placement of decoupling capacitors in Power Distribution Networks (PDNs). The objective is to minimize the PDN’s impedance across a frequency range under given constraints.

---

## 📚 Theoretical Background

### 1. Power Delivery Network (PDN) Overview

- The **Power Delivery Network (PDN)** in integrated circuits (ICs) is the infrastructure that distributes power throughout the chip to ensure stable and reliable operation.
- PDNs include power supplies, power rails, decoupling capacitors, voltage regulators, and inductors.
- Their goal is to provide necessary power to all functional blocks, logic gates, memory cells, and other circuit elements within the chip.
- **Key Design Aspects:** Current density, impedance matching, voltage drop, power efficiency, and thermal management.
- <img width="471" height="336" alt="image" src="https://github.com/user-attachments/assets/ebae12e4-aaf2-426e-9bc2-542a75bed94f" />


#### Key Components

1. **Power Supplies:** Internal or external sources providing electrical power.
2. **Power Rails:** Conductive paths distributing different voltage levels across the chip.
3. **Decoupling Capacitors:** Placed throughout the chip to stabilize voltage and filter noise.
4. **Voltage Regulators/Inductors:** Regulate or filter voltage to specific chip regions.

---

### 2. Decoupling Capacitors: Intuitive Explanation

- **Role:** Act as electronic “shock absorbers,” smoothing sudden changes in power demand.
- **How They Work:** Provide a quick reservoir of energy during high demand and absorb excess during low demand.
- **In PDN:** Strategically placed to ensure all circuit parts get smooth, uninterrupted power, preventing glitches or malfunctions.

---

### 3. Surrogate Models

- A **surrogate model** (emulator/metamodel) is a simplified, computationally cheap approximation of a complex function or simulation.
- Used to speed up optimization and analysis by replacing expensive calculations with fast, learned predictions.
- **In this context:** Surrogate models approximate the PDN’s impedance calculation, accelerating the optimization process.

---

### 4. PDN Challenges and Optimization

#### 4.1 Transistor Scaling and Power Integrity

- **Moore’s Law:** Drives more transistors into smaller silicon areas, increasing current requirements and reducing noise margins.
- **Power Integrity:** Supplying stable, low-noise DC power to all components grows more challenging with density and speed.
- **Power Supply Noise (PSN):** Variations in supply voltage, notably from simultaneous switching of many transistors (Simultaneous Switching Noise, SSN), can degrade system performance.

#### 4.2 Impedance Control and Target Impedance

- **Target Impedance (Zt):** Derived from maximum allowable voltage noise and worst-case transient current.
- **PDN Ratio:** Ratio of actual PDN impedance to target impedance; values below 1 reduce risk of failure.
- **Cost Tradeoff:** Lower PDN impedance requires more/expensive components, more board layers, or premium materials.
- **Decoupling Capacitors:** Cost-effective means to reduce impedance, but optimal selection/placement is complex for large systems.

#### 4.3 Computational Intelligence for Decap Selection

- **Problem:** Many possible combinations of decap placements; intuition-based approaches don’t scale.
- **Solution:** Computational intelligence (machine learning, metaheuristics like PSO) tackles these combinatorial challenges.
- **State-of-the-Art:** Neural networks, reinforcement learning, and PSO have all been used—surrogate models further accelerate large-scale optimization.

---

### 5. Optimization Problem Formulation

- **Placement:** Decaps are placed on power/ground planes via layout-defined ports.
- **Objective:** Minimize the PDN’s maximum self-impedance across frequencies, ensuring it is below the target impedance.
- **Calculation:** 
  - Equivalent self-impedance: `Zeq = (Z_pdn^(-1) + Z_decap^(-1))^(-1)`
  - Optimization variable: Which decap to place at which port.
  - Goal: Use the minimum number of decaps for the required impedance.

---

### 6. Particle Swarm Optimization (PSO)

#### 6.1 Basic PSO Algorithm

- **Inspired by:** Swarm behavior (e.g., birds, fish).
- **Population:** Each “particle” represents a candidate solution (decap configuration).
- **Movement:** Particles update positions based on their own best-known solution and the global best found by the swarm.
- **Update Equation:** Combines inertia, cognitive (personal), and social (global) influences.
- <img width="370" height="525" alt="image" src="https://github.com/user-attachments/assets/7e47c1ea-0d30-47fc-8374-0c5996656fbf" />


#### 6.2 Inertia Weight and Variants

- **Inertia Weight (ω):** Controls exploration/exploitation; linearly decreasing inertia weight (LDIW) is commonly used.
- **Fitness Evaluation:** Objective function (PDN self-impedance) is computationally expensive—matrix inversions at each iteration.
- **Total Real Evaluations (TRE):** Can be a limiting factor for runtime.

  ## 6.2 Simulation Parameters, Results & Performance Plots

### Parameters Used for PSO in this Simulation

- **Dataset:**
  - Capacitors: `decaps.mat`
  - Y-matrix: `y.mat`
  - Frequency: `freq.mat`
- **PSO Parameters:**
  - Maximum frequency points (\(f_{max}\)): 1391
  - Number of particles (\(N_p\)): 50
  - Number of ports: 40
  - Maximum iterations: 50
  - Total number of capacitors: 3348
  - Impedance threshold (\(Z_t\)): 0.06 Ω
  - Inertia weights (\(\omega_i, \omega_f\)): 0.9, 0.4
  - Cognitive coefficient (\(c_1\)): 1.5
  - Social coefficient (\(c_2\)): 1.5

---

### 6.3 Result Plots

**Global Best Value Against Iterations**

- Plots illustrate the convergence of the global best value for each NOC (Number of Capacitors, 1–6).
- Each plot shows the reduction in global best value with iterations.
- **Final values achieved:**
  - NOC = 1: \(Z = 0.087365\)
  - NOC = 2: \(Z = 0.073147\)
  - NOC = 3: \(Z = 0.086186\)
  - NOC = 4: \(Z = 0.070801\)
  - NOC = 5: \(Z = 0.062367\)
  - NOC = 6: \(Z = 0.059266\)
<img width="483" height="307" alt="image" src="https://github.com/user-attachments/assets/b52f327b-35ac-497e-8ad7-ddb9dd57826f" />


*Figure 6.1: Simulation result showing number of decaps and capacitor IDs for the optimal arrangement.*

<img width="819" height="777" alt="image" src="https://github.com/user-attachments/assets/9512ebb9-76bc-4ba1-a5bf-4f5416190e27" />


*Figure 6.2: Performance plots for different NOCs (Number of Decoupling Capacitors) — Global Best Value vs Iterations.*

**Impedance vs. Frequency**

- The plot displays the impedance curve for the optimal configuration found by PSO.
- Shows minimal impedance at specific frequencies, with a peak at higher frequencies.

<img width="332" height="262" alt="image" src="https://github.com/user-attachments/assets/1665ae48-a3ca-4f16-b051-ac31299232f4" />


*Figure 6.3: Impedance vs Frequency for NOC = 6 (optimal).*

---

### 6.4 Simulation Summary

- **Findings:** The simulation found that NOC = 6 produced the ideal arrangement.
- **Minimum Impedance:** \(Z_{min} = 0.059226\,\Omega\)
- **Indexes of Capacitors:** [1377, 972, 1521, 1294, 1138, 1159]
- **Ports:** [16, 10, 15, 7, 8, 11]
- **Time Taken:** ~411.3 seconds

**This simulation effectively illustrates the usefulness of PSO for PDN optimization, achieving excellent accuracy in impedance minimization and computational efficiency.**

---

---

### 7. Surrogate-Assisted Optimization

#### 7.1 What is Surrogate-Assisted Optimization?

- Uses surrogate (metamodel) to approximate expensive objective functions.
- Reduces number of real (expensive) evaluations by first consulting the surrogate.

#### 7.2 Choice of Surrogate: Radial Basis Function Network (RBFN)

- **RBFN:** A type of artificial neural network, robust for nonlinear problems, effective with small training sets, and computationally efficient.
- **Why RBFN:** Handles nonlinear, high-dimensional problems well; less hyperparameter tuning than Gaussian Process surrogates.

#### 7.3 Algorithm Integration

- **Database Initialization:** Generate random configurations, evaluate real fitness, and use top performers to seed the initial PSO swarm and RBFN training.
- **Surrogate Update:** During PSO, surrogate predicts fitness; only promising candidates are evaluated with the real objective, further updating the surrogate.
- **Result:** Fewer expensive function evaluations, faster convergence, and robust optimization.
## 7.3 SuA-PSO Simulation Parameters, Results & Performance Plots

### Parameters Used for SuA-PSO in this Simulation

- **Dataset:**
  - Capacitors: `decaps.mat`
  - Y-matrix: `y.mat`
  - Frequency: `freq.mat`
- **PSO Parameters:**
  - Maximum frequency points (\(f_{max}\)): 1391
  - Number of particles (\(N_p\)): 50
  - Number of ports: 40
  - Maximum iterations: 50
  - Total number of capacitors: 3348
  - Impedance threshold (\(Z_t\)): 0.06 Ω
  - Inertia weights (\(\omega_i, \omega_f\)): 0.9, 0.4
  - Cognitive coefficient (\(c_1\)): 1.5
  - Social coefficient (\(c_2\)): 1.5

---

### 7.4 Result Plots for SuA-PSO Approach

**Global Best Value Against Iterations**

- Plots show the convergence of the global best value for each NOC (Number of Capacitors, 1–5).
- Each plot highlights the reduction in global best value with iterations.
- **Final values achieved:**
  - NOC = 1: \(Z = 0.0943\)
  - NOC = 2: \(Z = 0.0838\)
  - NOC = 3: \(Z = 0.0683\)
  - NOC = 4: \(Z = 0.0632\)
  - NOC = 5: \(Z = 0.0593\)

<img width="453" height="221" alt="image" src="https://github.com/user-attachments/assets/d1bba3a5-4e0c-42fd-bd5a-a26d8edea794" />


*Figure 7.1: Simulation result showing number of decaps and capacitor IDs for the optimal SuA-PSO arrangement.*

<img width="783" height="820" alt="image" src="https://github.com/user-attachments/assets/82f5f292-39af-43d2-97aa-a4ad225e9c22" />


*Figure 7.2: Performance plots for different NOCs (Number of Decoupling Capacitors) — Global Best Value vs Iterations for SuA-PSO.*

**Impedance vs. Frequency**

- The plot displays the impedance curve for the optimal configuration found by SuA-PSO for each NOC.
- Shows minimal impedance at specific frequencies, with a peak at higher frequencies for different NOC values.

<img width="518" height="395" alt="image" src="https://github.com/user-attachments/assets/b8d35367-e599-455a-b228-386705ec1bce" />


*Figure 7.3: Impedance vs Frequency for each NOC (SuA-PSO).*

---

### 7.5 Simulation Summary

- **Findings:** The simulation found that NOC = 5 produced the ideal arrangement.
- **Minimum Impedance:** \(Z_{min} = 0.059269\,\Omega\)
- **Indexes of Capacitors:** [611, 2252, 932, 1169, 2274]
- **Ports:** [16, 10, 8, 6, 11]
- **Time Taken:** ~124.68 seconds

**This simulation effectively illustrates the usefulness of SuA-PSO for PDN optimization, achieving excellent accuracy in impedance minimization and computational efficiency.**

---
---

## 🧮 MATLAB Implementation: PSO + RBFN Code

Below is the MATLAB implementation of the PSO algorithm combined with RBFN surrogate modeling for PDN decap optimization:

```matlab
% Load the datasets
decaps = load('decaps.mat');
y = load('y.mat');
freq = load('freq.mat');
% Start timer
tic;
disp('RBFN-PSO Method');
% PSO Parameters
fmax = 1391;  % Maximum frequency points
Np = 40;      % Number of particles
ports = 20;   % Number of ports
maxiter = 50; % Maximum iterations
Decapcount = 3348;  % Total number of capacitors
zt = 60e-3;   % Impedance threshold
wf = 0.4;    % Initial inertia weight
wi = 0.9;    % Initial inertia weight
c1 = 1.5;     % Cognitive coefficient
c2 = 1.5;     % Social coefficient
e_th = 1e-3;  % Error threshold
% Initialize random matrices for velocity updates
r1 = rand(Np, 2 * ports);
r2 = rand(Np, 2 * ports);
% Initialize X1 with valid ranges
X = [randi([1, Decapcount], Np, ports), randi([2, 21], Np, ports)];
% Initialize velocity, lower, and upper bounds
vel = zeros(Np, 2 * ports); % Initialize velocity to 0
lb = [ones(Np, ports), 2 * ones(Np, ports)];
ub = [Decapcount * ones(Np, ports), 21 * ones(Np, ports)];
Xcopy = X;
localbestval = ones(Np, 1) * Inf;
% Function to evaluate fitness
evaluate_fitness = @(X, noc) calculate_fitness(X, noc, decaps.decaps, y.y, fmax, ports);
% Initialize variables to track the overall best solution
overall_best_noc = 0;
overall_best_impedance = Inf;
overall_best_X = [];
% Initialize variables for tracking time and RBFN usage
total_time = 0;
overall_start_time = tic;
% ========================================================================
% Main PSO loop
% ========================================================================
best_values_per_noc = Inf(1, ports); % Initialize with Inf
for noc = 1:ports % Iterate from lowest to highest number of capacitors
   % RBFN Initialization (Moved inside the NOC loop)
   RBFN_centers = [];
   RBFN_values = [];
   % Re-initialize best values for this NOC
   globalbestval = Inf;
   globalbestX = zeros(1, 2 * ports);
   globalbestIter = 0;  % Keep track of the iteration when the global best was found
   % Build initial database with 'Id' random solutions
   Id = 200;
   initial_database_X = [randi([1, Decapcount], Id, ports), randi([2, 21], Id, ports)];
   initial_database_fitness = zeros(Id, 1);
   for i = 1:Id
       initial_database_fitness(i) = evaluate_fitness(initial_database_X(i,:), noc);
   end
   % Sort database based on fitness values
   [initial_database_fitness, sorted_indices] = sort(initial_database_fitness);
   initial_database_X = initial_database_X(sorted_indices, :);
   % Initialize RBFN with a subset of initial data
   RBFN_centers = initial_database_X(1:Np, :); % Use top Np solutions
   RBFN_values = initial_database_fitness(1:Np);
   % Calculate D_max
   if size(RBFN_centers, 1) > floor(Np/2)
       distances = pdist(RBFN_centers);
       D_max = max(distances);
       d = 2 * ports;
       RBFN_spread = D_max * (d * size(RBFN_centers, 1))^(-1/d);
   else
       RBFN_spread = 0.5;
   end
   % Define Radial Basis Function (Gaussian)
   rbf = @(x, c) exp(-sum((x - c).^2) / (2 * RBFN_spread^2));
   % Track number of RBFN uses (reset for each NOC)
   num_rbfn_uses = 0;
   % Track number of actual fitness evaluations (reset for each NOC)
   num_actual_fitness_evaluations = 0;
   X = Xcopy;
   localbestval = ones(Np, 1) * Inf;
   Xl = X;                      % Local best positions
   Xg = repmat(initial_database_X(1,:), Np, 1); % Global best positions (initialize with initial database)
   vel = zeros(Np, 2 * ports);
   disp(['NOC = ', num2str(noc)]);
   globalbestarray = Inf * ones(maxiter, 1);
   final_impedance_vs_freq = zeros(fmax, 1);
   noc_start_time = tic;  % Start time for this NOC
   % PSO Iterations
   for iter = 1:maxiter
       w = wf + (wi - wf) * (maxiter - iter) / maxiter;
       iteration_best_val = Inf; % Best value within this iteration
       iteration_best_X = [];
       % Iterate through particles
       for i = 1:Np
           % Update particle velocities and positions
           for n = 1:noc
               vel(i, n) = w * vel(i, n) + c1 * r1(i, n) * (Xl(i, n) - X(i, n)) + c2 * r2(i, n) * (Xg(i, n) - X(i, n)) ;
               X(i, n) = X(i, n) + vel(i, n);
               vel(i, ports + n) = w * vel(i, ports + n) + c1 * r1(i, ports + n) * (Xl(i, ports + n) - X(i, ports + n)) + c2 * r2(i, ports + n) * (Xg(i, ports + n) - X(i, ports + n));
               X(i, ports + n) = X(i, ports + n) + vel(i, ports + n);
           end
           % Rounding and boundary handling
           X(i, 1:noc) = round(X(i, 1:noc));
           X(i, ports+1:ports+noc) = round(X(i, ports+1:ports+noc));
           X(i, :) = max(min(X(i, :), ub(i, :)), lb(i, :));
           % Ensure unique port assignments
           check = ones(1, 21);
           k = 1;
           while k <= noc
               port_index = X(i, ports + k);
               % Validate the index before using it.
               if port_index >= 1 && port_index <= 21
                   if check(port_index) == 1  % Use port_index directly
                       check(port_index) = 0;
                       k = k + 1;
                   else
                       X(i, ports + k) = X(i, ports + k) + 1;
                       if X(i, ports + k) > 21
                            X(i, ports + k) = 2;
                          % If port assignment exceeds limit, try to wrap around. If it is still occupied, skip solution
                           X(i, ports+k) = find(check, 1);  % Assign first available port
                           if isempty(X(i, ports+k))
                               zmax = Inf; % invalidate solution
                               continue;
                           end
                       end
                   end
               else
                   % Handle case where port assignment is out of bounds
                   disp(['Warning: Port assignment out of bounds. Port was ', num2str(port_index)]);
                   X(i, ports + k) = 2;  % Assign a valid default port
               end
           end
           % RBFN prediction
           if size(RBFN_centers, 1) >= floor(Np/2)  % Sufficient RBFN Data (adjust if needed)
               rbf_outputs = zeros(size(RBFN_centers, 1), 1);
               for k = 1:size(RBFN_centers, 1)
                   rbf_outputs(k) = rbf(X(i,:), RBFN_centers(k,:));
               end
               %Check for NaN or Inf
               if any(isnan(rbf_outputs)) || any(isinf(rbf_outputs))
                   predicted_fitness = Inf;
               else
                   predicted_fitness = RBFN_values'*(rbf_outputs/sum(rbf_outputs));
               end
               % Use RBFN if prediction is worse than local best, to force exploration
               if isscalar(predicted_fitness) && ~isnan(predicted_fitness) && (predicted_fitness + e_th) >= localbestval(i)
                   num_rbfn_uses = num_rbfn_uses + 1;
                   continue; % Skip real evaluation
               end
           end
           % Evaluate the objective function (use the evaluate_fitness function)
           zmax = evaluate_fitness(X(i,:), noc);
           num_actual_fitness_evaluations = num_actual_fitness_evaluations + 1;
           % Update local and global best values
           if zmax < localbestval(i)
               localbestval(i) = zmax;
               Xl(i, :) = X(i, :);
           end
          % Keep track of the best particle *in this iteration*
          if zmax < iteration_best_val
              iteration_best_val = zmax;
              iteration_best_X = X(i,:);
          end
         %STOP if target impedance reached for this Configuration-Particle
         if (zmax < zt)
             disp('Target impedance reached for this Particle. Ending Search!');
             localbestval(i) = zmax; % Update local best
             Xl(i, :) = X(i, :);
         end
       end % End particle loop
       % AFTER evaluating all particles in the iteration, update the global best
       if iteration_best_val < globalbestval
           globalbestval = iteration_best_val;
           globalbestX = iteration_best_X;
           globalbestIter = iter; %Store current Iteration Number where a new best was found.
           disp(['NEW GLOBAL BEST = ', num2str(globalbestval), ' (Iteration: ', num2str(globalbestIter), ')']);
           Xg(1, :) = iteration_best_X;
           Xg = repmat(Xg(1, :), Np, 1);
             % Add evaluated data to RBFN training set (every iteration for faster adaptation)
             RBFN_centers = [RBFN_centers; iteration_best_X];
             RBFN_values = [RBFN_values; iteration_best_val];
             % Recompute D_max and RBFN_spread (optional, but good for adaptation)
            if size(RBFN_centers, 1) > 1
                distances = pdist(RBFN_centers);
                 D_max = max(distances);
                 d = 2 * ports;
                 RBFN_spread = D_max * (d * size(RBFN_centers, 1))^(-1/d);
             end
       end
       % Update globalbestarray with the best impedance value found in this iteration
       globalbestarray(iter) = globalbestval;
       % Early stopping criterion *REMOVED*
       % if iter > 10 && globalbestval == globalbestarray(iter-10)
       %    disp('Convergence reached. Stopping early.');
       %    break;
       %end
       %STOP if target impedance reached for this NOC
       if (globalbestval <= zt)
           disp('Target impedance reached for this NOC. Ending Search!');
           overall_best_impedance = globalbestval;
           overall_best_noc = noc;
           overall_best_X = globalbestX;
           break;  % Break the *inner* loop and move to the next NOC
       end
   end % End iteration loop
   noc_elapsed_time = toc(noc_start_time);  % Time taken for this NOC
   % Store the best value for this NOC
   best_values_per_noc(noc) = globalbestval;
   disp(['Best Value for NOC = ', num2str(noc), ': ', num2str(globalbestval)]);
   final_impedance_vs_freq = calculate_impedance_vs_freq(globalbestX, noc, decaps.decaps, y.y, fmax, ports);
   % Store impedance vs frequency for this NOC
   %impedance_vs_freq_all_noc(:, noc) = final_impedance_vs_freq;
   % Print RBFN Usage and Fitness Evaluations
   disp(['Number of RBFN Centers Used: ', num2str(size(RBFN_centers, 1))]);
   disp(['Number of RBFN Uses: ', num2str(num_rbfn_uses)]);
   disp(['Number of Actual Fitness Evaluations: ', num2str(num_actual_fitness_evaluations)]);
   % Print Global Best Value vs Iterations
   figure;
   plot(1:iter, globalbestarray(1:iter), '-b', 'LineWidth', 1.5);
   xlabel('Iterations');
   ylabel('Global Best Value');
   title(['Global Best Value for NOC = ', num2str(noc)]);
   grid on;
   % Plot Impedance vs Frequency
   figure;
   plot(freq.freq, final_impedance_vs_freq, '-r', 'LineWidth', 1.5); % Corrected x-axis to use frequency values
   xlabel('Frequency (Hz)');
   ylabel('Impedance (Ohms)');
   title(['Impedance vs Frequency for NOC = ', num2str(noc)]);
   grid on;
   % Print Time taken for this NOC
   disp(['Time taken for NOC = ', num2str(noc), ': ', num2str(noc_elapsed_time), ' seconds']);
   % If we find a good impedance at lower NOC, break entire loop
   if(overall_best_noc>0)
       break;
   end
end % End NOC loop
% Calculate overall time
total_time = toc(overall_start_time);
% Find NOC with the overall best impedance
[overall_best_impedance, overall_best_noc] = min(best_values_per_noc);
% Get final configuration based on best NOC
final_cap_assignment = overall_best_X(1:overall_best_noc);
final_port_assignment = overall_best_X(ports+1:ports+overall_best_noc);
% Plot all impedance vs frequency curves up to the required NOC
figure;
hold on;
colors = lines(ports); % Generate distinct colors for each NOC
for noc = 1:ports
   if(best_values_per_noc(noc)~=Inf)
       % Use THIS NOC's global best solution for plotting
       final_impedance_vs_freq = calculate_impedance_vs_freq(globalbestX, noc, decaps.decaps, y.y, fmax, ports);
       plot(freq.freq, final_impedance_vs_freq, 'LineWidth', 1.5, 'DisplayName', ['NOC = ', num2str(noc)]);
   end
end
hold off;
xlabel('Frequency (Hz)');
ylabel('Impedance (Ohms)');
title('Impedance vs Frequency for NOC Values');
legend show;
grid on;
disp('Best values for each NOC:');
% Replace Inf with NaN for cleaner output
best_values_per_noc(isinf(best_values_per_noc)) = NaN;
disp(best_values_per_noc);
% Final results based on the overall best
disp(['Optimum NOC: ', num2str(overall_best_noc)]);
disp(['Impedance for ',num2str(overall_best_noc),' de-caps: ', num2str(overall_best_impedance)]);
% Final output format
disp('Final Capacitor-Port Configuration:');
disp(['NOC: ', num2str(overall_best_noc)]);
disp(['Impedance: ', num2str(overall_best_impedance)]);
final_cap_assignment = overall_best_X(1:overall_best_noc);
final_port_assignment = overall_best_X(ports+1:ports+overall_best_noc);
disp('Capacitors:');
disp(final_cap_assignment);
disp('Ports:');
disp(final_port_assignment);
% Print Total Time taken
disp(['Total Time taken for the entire optimization: ', num2str(total_time), ' seconds']);
% Helper Functions (Separate for clarity)
function zmax = calculate_fitness(X, noc, decaps, y, fmax, ports)
   % Calculate the fitness (maximum impedance) for a given configuration
   zmax = 0;
   for f = 1:fmax
       Ydecap = zeros(21, 21);
       for j = 1:noc
           a = X(j);
           b = X(ports + j);
           c = decaps(a, f);
           Ydecap(b, b) = Ydecap(b, b) + c;
       end
       Yeq = y(:, :, f) + Ydecap;
       try
           Zeq = inv(Yeq);
       catch
           Zeq = eye(size(Yeq))*Inf;
           disp('Warning: Singular matrix encountered during inversion. Setting impedance to Inf.');
       end
       Z = abs(Zeq(1, 1));
       if Z > zmax
           zmax = Z;
       end
   end
end
function final_impedance_vs_freq = calculate_impedance_vs_freq(X, noc, decaps, y, fmax, ports)
   % Calculate the impedance vs frequency for a given configuration
   final_impedance_vs_freq = zeros(fmax, 1);
   for f = 1:fmax
       Ydecap = zeros(21, 21);
       for j = 1:noc
           a = X(j);
           b = X(ports + j);
           c = decaps(a, f);
           Ydecap(b, b) = Ydecap(b, b) + c;
       end
       Yeq = y(:, :, f) + Ydecap;
       try
           Zeq = inv(Yeq);
       catch
           Zeq = eye(size(Yeq))*Inf;
           disp('Warning: Singular matrix encountered during inversion. Setting impedance to Inf.');
       end
       Z = abs(Zeq(1, 1));
       final_impedance_vs_freq(f) = Z;
   end
end
```

---

## 📁 Data Loading Stage

The following `.mat` files are required:

### `decaps.mat`
- Matrix of decoupling capacitor admittance profiles.
- **Rows**: Each row represents one capacitor.
- **Columns**: Each column corresponds to a specific frequency.
- **Values**: Complex numbers (real + imaginary admittance).
- Used to model frequency-dependent responses.

### `y.mat`
- 3D matrix representing the base admittance of the PDN.
- Dimensions suggest a **multi-port** system.
- Frequency-dependent (third dimension).
- Each port is a potential location to place capacitors.

### `freq.mat`
- Frequency vector used to align frequency-specific data.
- Shared vector used in both `y.mat` and `decaps.mat`.

---

## ⚙️ Initialization

### PSO Parameters

- `Np` → Number of Particles
- `maxiter` → Maximum number of iterations
- `wi`, `wf` → Initial and Final inertia weights
- `c1`, `c2` → PSO coefficients (personal and social influences)
- `e_th` → RBFN surrogate threshold

### Optimization Parameters

- `ports` → Number of PDN ports for capacitor placement
- `zt` → Target impedance to achieve

### Global Trackers

- `best_values_per_noc` → Best impedance for each number of capacitors (NOC)
- `overall_best_impedance` → Global best found impedance
- `overall_best_noc` → Number of capacitors used in the best configuration
- `total_time` → Tracks total runtime of the algorithm

---

## 🔁 Optimization Workflow

### 🧭 Outer Loop: Varying NOC (Number of Capacitors)

> Iterate from NOC = 1 to total number of ports

#### 1. RBFN Initialization

- Generate 100 random valid configurations:
  - Each configuration has:
    - `noc` random capacitor indices
    - `noc` unique port indices
- For each configuration:
  - Evaluate fitness using `calculate_fitness(X, noc)`:
    - Construct equivalent admittance matrix `Yeq`
    - Calculate impedance matrix: `Zeq = inv(Yeq)`
    - Extract input impedance: `Z = abs(Zeq(1,1))` over all frequencies
    - Objective: Minimize `zmax = max(Z)`
- Select top `Np` configurations based on `zmax`
- Use them as:
  - Initial **PSO swarm**
  - **RBFN centers** and values
- Compute `D_max` and RBFN spread

#### 2. Swarm Initialization

- Initialize:
  - `X` and `Xl` with top `Np` configurations
  - `Xg`: Global best (repeated across particles)
  - `vel`: Particle velocity vectors (zeros)
- Set:
  - Global best value from initial configurations

---

### 🔄 Inner Loop: PSO Iteration Loop

> Iterate for `iter = 1` to `maxiter`

#### At each iteration:

- **Update inertia weight**:  
  `w = wf + (wi - wf) * (maxiter - iter) / maxiter`

- **For each particle**:
  - Update **velocity** using position, local/global best, and random perturbations
  - Update **position**:
    - Round/clamp to valid index ranges
    - Resolve port conflicts (no overlap)
  - **Surrogate Evaluation** using RBFN:
    - If predicted fitness (plus `e_th`) < local best, perform real evaluation
  - **Update bests**:
    - Update local/global bests if improvement
    - Update RBFN centers
    - Recompute RBFN spread
- Store current `global_bestval` per iteration
- **Break early** if `global_bestval < zt`

---

## ✅ Final Output

### 📋 Terminal Outputs
- ✅ Optimal NOC (number of capacitors)
- 📉 Best impedance achieved
- 📌 Final configuration (capacitor & port indices)
- 🔁 Total evaluations: RBFN vs real
- ⏱ Total run-time

### 📈 Plots
- Impedance vs Frequency (all NOC cases)
- Global best fitness per iteration (optimization progress)

---

## 🚀 Getting Started

1. 🔃 Load these files:
   - `decaps.mat`
   - `y.mat`
   - `freq.mat`

2. ✏️ Modify PSO and optimization parameters at the start of the main script.

3. ▶️ Run the script in MATLAB.

4. 📊 Review console output and plots.


- This implementation is MATLAB-based and designed for high-performance PDN analysis.
- RBFN surrogate model significantly reduces computational cost by approximating expensive fitness evaluations.
- The algorithm supports early stopping for faster convergence when target impedance is met.
- ## 8. Performance Comparison of Algorithm

### 8.1 Simulation Results for Ten Independent Runs

**Dimension:** 3348 (capacitors) × 1391 (frequency points)

**Table 8.1:** Performance Comparison between PSO and SuA-PSO (10 Runs)

|        | **PSO**                 |        |        | **SuA-PSO**               |        |
|--------|-------------------------|--------|--------|---------------------------|--------|
| Z (mΩ) | T (sec) | Nd (No. of decaps) | | Z (mΩ) | T (sec) | Nd (No. of decaps) |
| 59.6   | 381     | 5               |        | 57.4   | 146     | 6              |
| 59.2   | 411     | 6               |        | 59.5   | 164     | 7              |
| 59.8   | 426     | 6               |        | 56.1   | 153     | 6              |
| 56.7   | 512     | 7               |        | 59.4   | 166     | 7              |
| 55.4   | 353     | 5               |        | 59.2   | 124     | 5              |
| 59.5   | 338     | 6               |        | 59.2   | 204     | 8              |
| 58.4   | 385     | 6               |        | 59.4   | 171     | 7              |
| 59.4   | 408     | 6               |        | 59.4   | 153     | 6              |
| 59.9   | 418     | 7               |        | 59.1   | 134     | 5              |
| 58.5   | 360     | 7               |        | 58.9   | 169     | 6              |

**Table 8.2:** Summary of Averaged Performance Metrics

| Criterion     | PSO        | SuA-PSO  |
|---------------|------------|----------|
| \(N_{avg}\) (Average no. of decaps) | 6 | 6 |
| \(N_{min}\) (Minimum no. of decaps) | 5 | 5 |
| \(T\) (Average Time in sec)         | 399.2 | 158.4 |
| **Gain in CPU Time**                | –      | **60.3%** |



---

**Conclusion:**  
SuA-PSO significantly reduced computation time compared to traditional PSO, achieving a **60.3% gain in CPU time**. Both algorithms achieved similar impedance performance and decap requirements. This demonstrates that **surrogate-assisted optimization using RBFN can maintain solution quality while greatly improving performance**.

---
