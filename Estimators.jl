
module Estimators

using ..DGP: Data, globals
using Optim
using Statistics
using LinearAlgebra
using Base.Iterators
using ForwardDiff: gradient, hessian
using Intervals
using Pipe: @pipe

export acfObjective, estimateACF, estimateOLS

"""
    Given a data struct, create the objective function that we will minimize as part of the ACF procedure.

    This is separate from the estimateACF function so that a Newton step can be taken in the estimateOLS function.

"""
function acfObjective(data::Data)
    ϕ = let
        X = [ data.lnKapital[:] data.lnIntermedInput[:] ]
        X = [ ones(size(X, 1)) X ]
        y = data.lnOutput[:] - globals.betaL * data.lnLabor[:]
        β = (X'X)\(X'y)

        reshape(X * β, globals.nperiodsKeep, globals.nfirms)
    end

    lag(x::Matrix{T}) where {T<:Real}  = x[1:(end-1),:][:]
    lead(x::Matrix{T}) where {T<:Real}  = x[2:end,:][:]
    lag(x::Symbol) = getfield(data, x)[1:(end-1),:][:]
    lead(x::Symbol) = getfield(data, x)[2:end,:][:]
        
    ϕ₋₁ = lag(ϕ)
    ϕ₀ = lead(ϕ)
    k₋₁ = lag(:lnKapital)
    k₀ = lead(:lnKapital)
    l₋₁ = lag(:lnLabor)
    l₀ = lead(:lnLabor)

    return function(θ)
        # innovation term from ω process
        ξ = let
            ω₋₁ = ϕ₋₁ - θ[1] *  k₋₁
            ω₀ = ϕ₀ - θ[1] *  k₀

            X = [ ones(size(ω₋₁)) ω₋₁ ]
            β = (X'X) \ (X'ω₀)

            ω₀ - X * β
        end
        
        # now we have just one moment condition instead of two
        Ψ = ξ .* k₀
        
        μ = mean(Ψ)
        σ² = cov(Ψ)
        n = length(Ψ)

        return n * μ^2 / σ²
    end
end

estimateACF(data::Data) = @pipe data |> acfObjective |> optimize(_, 0, 1) |> Optim.minimizer(_)[1]

function makePolynomial(order::Integer, vecs::Vararg{Vector{T}})::Matrix{T} where {T <: Real}
    powers = fill(0:order, length(vecs))
    vector_multiply(vecs::Vararg{Vector{T}}) = broadcast(*, vecs...)
    
    # each term is a vector of the powers to which we want to raise each corresponding element of vecs
    mapreduce(hcat, product(powers...)) do term
        mapreduce(vector_multiply, enumerate(term)) do (i, power)
            vecs[i] .^ power
        end
    end
end

function estimateOLS(data::Data; includeKNoShock = false, newtonSteps = 0)
    lag(x::Symbol) = getfield(data, x)[1:(end-1),:][:]
    lead(x::Symbol) = getfield(data, x)[2:end,:][:]

    y₀ = lead(:lnOutput)
    l₀ = lead(:lnLabor)
    k₋₁ = lag(:lnKapital)
    k₀ = lead(:lnKapital)
    m₋₁ = lag(:lnIntermedInput)
    kNoShock = lead(:lnKapitalNoShock)

    X = [ k₀ makePolynomial(2, k₋₁, m₋₁) ]

    if includeKNoShock
        X = [X kNoShock]
    end

    y = y₀ - globals.betaL * l₀

    # store intial β₀ in case our optimization goes awry
    β = ((X'X) \ (X'y))[1]


    step = let f = acfObjective(data)
        α -> ( α - (gradient(f, [α])/hessian(f, [α]))[1] )
    end
    
    # do the prescribed number of Newton steps, but stop
    # early if it looks like we are heading to a minimum
    # in the negative limit
    for _ in 1:newtonSteps
        βʼ = step(β)
        if ~( βʼ ∈ 0..1 )
            return β
        else
            β = βʼ
        end
    end
    
    return β
end

end