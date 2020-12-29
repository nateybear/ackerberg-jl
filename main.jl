module Runner

using Dates
using UUIDs
using Printf
using DataFrames
using CSV

include("DGP.jl")
include("Estimators.jl")

using .DGP: globals, setDGP, generateData
using .Estimators

const σs = [.01:.01:.1; .15:.05:.5]
const n = globals.niterations

struct Estimator
    name::String
    f::Function
end

const estimators = [
    Estimator("OLS", estimateOLS),
    #Estimator("OLS w Unaltered K", x->estimateOLS(x, includeKNoShock=true)),
    #Estimator("ACF", last ∘ estimateACF),
    #Estimator("OLS w Newton Step", x->estimateOLS(x, newtonSteps = 1))
]

# batched write to CSV in loop below. need to instantiate
const headers = [:sd_zeta, :estimator, :estimate]
const coltypes = [Float16, String, Float64]
const OutputRow = NamedTuple{tuple(headers...), Tuple{coltypes...}}

const fname = "$(today())-$(uuid4()).csv"
CSV.write(fname, DataFrame(coltypes, headers, 0))

const output = Vector{OutputRow}(undef, n*length(estimators))

@time for σ ∈ σs
    setDGP(dgp = 2, sigmaLogK = σ)
    for i ∈ 1:globals.niterations
        data = generateData()
        for (ie,e) ∈ enumerate(estimators)
            output[ie + (i-1)*length(estimators)] = (
                sd_zeta = σ, estimator = e.name, estimate = e.f(data)
            )
        end
    end
    CSV.write(fname, DataFrame(output, copycols = false), append = true)
end

print("Wrote estimates to $fname\n")

end