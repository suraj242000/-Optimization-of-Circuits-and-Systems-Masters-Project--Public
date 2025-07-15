# ⚡ PDN Optimization Using PSO with RBFN Surrogates

This MATLAB project implements a Particle Swarm Optimization (PSO) algorithm with Radial Basis Function Network (RBFN) surrogate modeling to optimize the placement of decoupling capacitors in Power Distribution Networks (PDNs). The objective is to minimize the PDN’s impedance across a frequency range under given constraints.

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

---

## 📌 Notes

- This implementation is MATLAB-based and designed for high-performance PDN analysis.
- RBFN surrogate model significantly reduces computational cost by approximating expensive fitness evaluations.
- The algorithm supports early stopping for faster convergence when target impedance is met.



