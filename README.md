# Introduction

This document supplements the paper \[ADD CITATION\]. It provides more
details on the implementation and a short user guide to either reproduce
the figures of the paper or re-run the MATLAB scripts.

# Overview

The numerical result section is split into two parts. The first part is
to compared the proposed approach with other state-of-the-art nonlinear
control methods. The second part demonstrates only the proposed approach
in a more challenging scenario. After providing the basic *installation*
instructions and what is needed, a short user guide for the two
aforementioned parts is provided.

## Needed Software and Setup

- MATLAB 2021b or newer is required (newer versions should also work,
  but are not tested!)

- For the pre-computation step we make use of
  CaΣoS [1] v1.0.0-rc. The used version can be found
  in the repository. Add the main folder of Ca$\Sigma$oS to the Matlab
  path.

- MOSEK [2] v10.2.5 is used as the underlying SDP
  solver for the pre-computation step. Academic licenses can be obtained
  from <https://www.mosek.com/products/academic-licenses/>. Follow the
  installation instructions from MOSEK.

- For online optimizations, we make use of
  CasADi [3] v3.6.7 to setup the problems and can
  be obtained from <https://web.casadi.org/get/>. We used
  qrqp [4] for the (S)QP and
  IPOPT [5] to solve the discrete-time OCP.
  The solvers are included in CasADi so no additional installation
  required.

## Running Examples and Reproducing Results

In the following, it is assumed everything is installed and setup as
explained before.

## Part I: Comparison

### Overview Files

For the CBF-CLF-QP, infinetesimal MPC scheme the user finds a synthesis
to compute the terminal conditions. These use an initial guess, which is
loaded from a `.mat` file. Once the synthesis is done, the terminal
conditions are also stored in a `.mat` file. This `.mat` files are the
loaded into the workspace in the simulation scripts.

`.mex` functions for the simulation are generated to improve simulation
time. To compile the `.mex` functions a C/C++ compiler is required.

For the full-horizon NMPC formulations (IPOPT, RTI), the terminal set is
also pre-computed, i.e., the maximum stable level set, which can be
found in `full_MPC_IPOPT/termIngredient_full.m`. After the simulations
run, plots from all runs are generated and `.mat ` files with the
results are stored in the main folder.

### Folder Structure

```text
Comparison_singleAxis/
├── CBF_CLF_QP         # CBF-CLF synthesis and simulation
├── full_MPC_alpaqa    # Full-horizon NMPC simulation using alpaqa solver
├── full_MPC_IPOPT     # Full-horizon NMPC simulation using IPOPT solver
├── helperFunc         # Folder that contains helper functions (e.g. MRP→Euler)
├── inf_MPC            # Synthesis and simulation of inf.MPC
├── poly_controlLaw    # Simulation of the poly. control law from proposed approach
└── RTI_full_MPC       # Full-horizon NMPC simulation using custom RTI solver
```


### Reproduction

We provide `.mat` files for each individual approach in the main folder,
i.e.,\
Comparison_singleAxis. This includes the pre-computation results or the
actual simulation results. Run `comparison_MultipleRuns.m`, to reproduce
the table from the paper and to get the plot for the single axis
rotation. Due to the large amount of data, the full-horizon NMPC
formulations (RTI and IPOPT) have post-processing scripts in their
folders. Once the actual simulation ran, the post-processing script
reduces the data to the comparison minimum. The data of the complete
workspace for the full-horizon formulations is not provided.

### Re-running 

The user is welcome to run the scripts and functions. It should be noted
that, for example, the full-horizon NMPC formulation might take a
significant amount of time for the simulation in MATLAB.

## Part II: Three-Axis Constrained Problem

### Overview Files

The second part only considers the infinetesimal MPC scheme. The user
finds a synthesis to compute the terminal conditions. These use an
initial guess, which is loaded from a `.mat` file. Once the synthesis is
done, the terminal conditions are also stored in a `.mat` file. This
`.mat` files are the loaded into the workspace in the simulation
scripts.

`.mex` functions for the simulation are generated to improve simulation
time. To compile the `.mex` functions a C/C++ compiler is required.

### Folder Structure

Not all functions and `.mat` files are listed. Only the most important
functions.

### Reproduction

Run `evaluation.m` to get all plots from the paper and additional once.

### Re-running

The inner-approximation of the constraint set (i.e., the allowable set)
for this scenario can be pre-computed using `innerApprox_allowSet.m`.
The allowable set is manually inserted into the synthesis
script`synthesis_CBF_CLF.m`. Important to know is that the simulations
script `inf_MPC_simulation.m` computes a new uniform distribution if
re-run. Thus, different results to the paper are expected!


### References


[1] T. Cunis and J. Olucak, “CaoS: A nonlinear sum-of-squares optimization suite,” in 2025 Amer-
ican Control Conference, (Boulder, CO), pp. 1659–1666, 2025.

[2] E. D. Andersen and K. D. Andersen, “The Mosek Interior Point Optimizer for Linear Program-
ming: An Implementation of the Homogeneous Algorithm,” in High Performance Optimization
(H. Frenk, K. Roos, T. Terlaky, and S. Zhang, eds.), 2000.

[3] J. A. E. Andersson, J. Gillis, G. Horn, J. B. Rawlings, and M. Diehl, “CasADi: a software frame-
work for nonlinear optimization and optimal control,” Mathematical Programming Computation,
vol. 11, no. 1, pp. 1–36, 2019.

[4] J. A. Andersson and J. B. Rawlings, “Sensitivity analysis for nonlinear programming in casadi,”
IFAC-PapersOnLine, vol. 51, no. 20, pp. 331–336, 2018. 6th IFAC Conference on Nonlinear
Model Predictive Control NMPC 2018.

[5] A. Wächter and L. T. Biegler, “On the implementation of an interior-point filter line-search
algorithm for large-scale nonlinear programming,” Mathematical Programming, vol. 106, pp. 25–
57, Mar. 2006.

