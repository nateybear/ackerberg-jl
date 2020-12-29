module Test
export beta
include("DGP.jl")
include("Estimators.jl")
using .DGP
using .Estimators
using Statistics: quantile
using Plots
using ForwardDiff: gradient, hessian

setDGP(dgp = 1, sigmaLogK = 0.01)

n = 25
beta = zeros(n)

for i in 1:n
    beta[i] = generateData() |> estimateOLS
end

plot(beta)

end
