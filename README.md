# ACF DGPs in Julia

This code runs extensions of the original ACF DGPs. In particular, it considers a non-deterministic investment process that introduces exogenous variation with which to identify the coefficient on capital.

The point of the code is to compare the performance of ACF to OLS and extensions to OLS. In particular, it tries to make the point that as the variance of the investment shock increases, OLS and friends achieve similar efficiency as ACF. Hence, the computational burden of the nonlinear search that ACF uses could be obviated given certain assumptions on investment.

# Running the code

The "main loop" is contained in main.jl. It will save estimates to CSV in long format, with a column for estimator used and the standard deviation of the investment shock, zeta. Once you have cloned this repository, simply run

```julia
using Pkg
Pkg.activate(".")
Pkg.instantiate()

include("main.jl")
```
