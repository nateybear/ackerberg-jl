using Dates
using UUIDs
using Printf
using DataFrames
using CSV

include("DGP.jl")
include("Estimators.jl")

using .DGP: globals, setDGP, generateData
using .Estimators

# The sds of the investment shock zeta to iterate over
const σs = [.01:.01:.1; .15:.05:.5]
# The number MC samples
const n = globals.niterations

"""
    Modular representation of the estimators we will use. Primarily to associate names to functions, but allows for other functionality to be implemented.
"""
struct Estimator
    name::String
    f::Function
end
Base.show(io::IO, e::Estimator) = Base.show(io, e.name)

const estimators = [
    Estimator("OLS", estimateOLS),
    Estimator("OLS w Unaltered K", x->estimateOLS(x, includeKNoShock=true)),
    Estimator("ACF", last ∘ estimateACF),
    Estimator("OLS w Newton Step", x->estimateOLS(x, newtonSteps = 1))
]

# batched write to CSV in loop below. need to instantiate the CSV file
const headers = [:sd_zeta, :estimator, :estimate]
const coltypes = [Float16, String, Float64]
const OutputRow = NamedTuple{tuple(headers...), Tuple{coltypes...}}

const filename = "$(today())-$(uuid4()).csv"
CSV.write(filename, DataFrame(coltypes, headers, 0))

# output is stored as a vector of named tuples, which is a Tables.jl compliant data structure
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
    CSV.write(filename, DataFrame(output, copycols = false), append = true)
end

print("Wrote estimates to $fname\n")