# Optimization – CIMAT CIMAT (Spring 2024)

**Author:** Ezau Faridh Torres Torres  
**Advisor:** Dr. Joaquín Peña Acevedo  
**Course:** Optimization  
**Institution:** CIMAT – Centro de Investigación en Matemáticas  
**Term:** Spring 2024 

*description*

## 📄 Table of Contents

- [Repository Structure](#repository-structure)
- [Technical Stack](#technical-stack)
- [Overview of Assignments](#overview-of-assignments)
  - [Assignment 1 – LU and Cholesky Decomposition](#assignment-1--lu-and-cholesky-decomposition)
  - [Assignment 2 – QR Decomposition and Least Squares](#assignment-2--qr-decomposition-and-least-squares)
  - [Assignment 3 – Numerical Stability](#assignment-3--numerical-stability)
  - [Assignment 4 – Eigenvalue Computation](#assignment-4--eigenvalue-computation)
  - [Assignment 5 – Stochastic Simulation](#assignment-5--stochastic-simulation)
  - [Assignment 6 – MCMC: Metropolis-Hastings](#assignment-6--mcmc-metropolis-hastings)
  - [Assignment 7 – Metropolis-Hastings in Multivariate Settings](#assignment-7--metropolis-hastings-in-multivariate-settings)
  - [Assignment 8 – MCMC with Hybrid Kernels and Gibbs Sampling](#assignment-8--mcmc-with-hybrid-kernels-and-gibbs-sampling)
  - [Final Project – Bayesian Inference for Weibull Parameters](#final-project--bayesian-inference-for-weibull-parameters)
- [Contact](#-contact)

---

## Repository Structure

Each assignment comprises the following elements:

- Python scripts with modular implementations of the required models and methods.
- A `report.pdf` that explains the methodology and findings.
- A `results/` directory with visual representations of the results.  

---

## Technical Stack

This project was developed in Python 3.11 using:

- **Core libraries:** `numpy`, `scipy`, `matplotlib`, `pandas`
- **Symbolic computation:** `sympy`
- **Statistical modeling & distributions:** `scipy.stats`
- **Plotting & visualization:** `seaborn`, `matplotlib`
- **Jupyter Notebooks** (for prototyping)

> Note: Each assignment may include additional libraries specified in the corresponding script headers.

---

## Overview of Assignments

The following section presents a concise overview of each task, highlighting its primary objective:

### Assignment 1 – *Golden Section Search and Gradient-Based Optimization* 
Implementation of the Golden Section Search method to find minima of univariate functions. The task also includes computing directional derivatives and applying the method to optimize along descent directions for the Griewangk function. Finally, a gradient is inferred from directional derivatives using linear systems.

<div align="center">
  <img src="https://github.com/ezautorres/Optimization-CIMAT/blob/main/assignment1/output.png" alt="Golden Section Search on test functions" width="500"/>
</div>


### Assignment 2 – *Exact Line Search and Gradient Descent on Quadratics*
Exploration of gradient descent with exact line search on quadratic functions. First, the number of iterations required for convergence is estimated using spectral properties of the Hessian. Then, the performance of the gradient descent algorithm is evaluated across datasets of increasing dimensionality. Finally, the method is tested on nonlinear functions (Himmelblau, Beale, and Rosenbrock) using golden section search to determine optimal step sizes, including an analysis of stopping criteria and trajectory visualization.

<div align="center">
  <img src="https://github.com/ezautorres/Optimization-CIMAT/blob/main/assignment2/output.png" alt="Gradient descent trajectories on test functions" width="500"/>
</div>

### Assignment 3 – *Newton’s Method for Nonlinear Optimization*  
Implementation of Newton’s method for unconstrained optimization. The assignment includes symbolic and numerical computation of gradients and Hessians using sympy, application of Newton’s method on various benchmark functions (Beale, Himmelblau, Rosenbrock), and a comparative analysis of convergence against gradient descent. The behavior of the method is visualized through contour plots of trajectories and stopping conditions are critically assessed.

<div align="center">
  <img src="https://github.com/ezautorres/Optimization-CIMAT/blob/main/assignment3/output.png" alt="Newton's method trajectory on benchmark function" width="500"/>
</div>

### Assignment 4 – *Conjugate Gradient Method*  
Solving large symmetric positive-definite systems using the Conjugate Gradient (CG) method. Theoretical properties are verified numerically, including orthogonality of residuals and convergence in at most n steps. A performance comparison is conducted between CG and standard solvers (np.linalg.solve) on randomly generated matrices, and the effect of perturbations in the system is explored to understand stability and convergence degradation.

<div align="center">
  <img src="https://github.com/ezautorres/Optimization-CIMAT/blob/main/assignment4/output.png" alt="Conjugate Gradient convergence behavior" width="500"/>
</div>

### Assignment 5 – *Numerical Optimization in Practice*  
Explores nonlinear multivariate optimization using gradient-based and Newton methods. Begins by locating and classifying critical points of a multivariable function via symbolic differentiation. Then, both gradient descent with exact line search and Newton’s method are applied to functions with indefinite Hessians, adjusting step size strategies as needed. The final part demonstrates convexity properties of function compositions and identifies intervals containing minimizers.

<div align="center">
  <img src="https://github.com/ezautorres/Optimization-CIMAT/blob/main/assignment5/output.png" alt="Function landscape and optimization steps" width="500"/>
</div>

### Assignment 6 – *Trust Region and BFGS Methods*  
Explores second-order optimization strategies, including Newton’s method with trust regions and the BFGS quasi-Newton algorithm. Both methods are applied to Himmelblau and Rosenbrock functions, with performance compared through trajectory visualization and function value decay. Implementation includes the dogleg method for step selection inside trust regions, and an update strategy for the inverse Hessian approximation in BFGS.

### Assignment 7 – *Constrained Optimization and KKT Conditions*  
Explores constrained optimization using Karush-Kuhn-Tucker (KKT) conditions. Begins with classification of constraint types and verification of constraint qualifications. Implements an active set strategy and barrier methods to solve problems with both equality and inequality constraints. Solutions are compared to SciPy’s minimize under trust-constr and SLSQP methods, analyzing the path toward the constrained optimum.

### Assignment 8 – *Penalty Methods for Constrained Optimization*  
Explores the use of quadratic penalty and logarithmic barrier functions for solving constrained optimization problems. Both equality and inequality constraints are handled through external penalization, with convergence monitored as the penalty parameter increases. Includes a comparison between penalty trajectories and solutions from standard constrained solvers, emphasizing the trade-off between feasibility and optimality.

### Assignment 9 – *Logistic Regression and Gauss-Newton for Nonlinear Least Squares*  
Explores the implementation of logistic regression using gradient descent with backtracking line search, applied to a real-world binary classification problem on heart disease data. It involves deriving the gradient of the regularized logistic loss and coding the classifier from scratch, including evaluation using accuracy and confusion matrix. The second part focuses on nonlinear regression using the Gauss-Newton method to fit a sigmoidal model to noisy data, highlighting parameter estimation and convergence behavior. Both sections emphasize numerical optimization strategies and practical modeling considerations.

---

### Test - *Constrained Optimization and Nonlinear Systems*
Implements advanced optimization methods for constrained problems and nonlinear equation systems. Part I solves a constrained minimization using quadratic penalty and modified BFGS with backtracking. Part II applies Newton’s method to solve a nonlinear system, examining convergence under random initialization.

---

### Final Project – *Hybrid Conjugate Gradient Method with Restart*  
Explores a family of hybrid conjugate gradient methods with restart procedures designed for large-scale unconstrained optimization and image restoration. Implements the IJYHL algorithm from recent literature, which combines two hybrid strategies into a single-parameter conjugate direction formula. The method ensures sufficient descent without line search dependency and guarantees global convergence under weak Wolfe conditions. The algorithm is tested on standard benchmark functions and compared against classical CG methods like HS, FR, PRP, and DY, highlighting its effectiveness and numerical robustness.

---

## Learning Outcomes

Throughout the course, I gained practical experience in:

- Implementing numerical linear algebra algorithms from scratch
- Performing polynomial and spline interpolation
- Solving ordinary differential equations using numerical schemes
- Designing and evaluating stochastic simulation pipelines (e.g., ARS, MCMC)
- Analyzing convergence and stability in numerical methods
- Applying Bayesian inference via MCMC techniques to real data
- Writing clear scientific reports with integrated visualizations

---

## 📫 Contact

- 📧 Email: ezau.torres@cimat.mx  
- 💼 LinkedIn: [linkedin.com/in/ezautorres](https://linkedin.com/in/ezautorres)