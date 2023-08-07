
module Estimators

using ..DGP: Data, globals
using Optim
using Statistics
using LinearAlgebra
using Base.Iterators
using ForwardDiff: gradient, hessian
using Intervals
using Pipe: @pipe
using DecisionTree
using ScikitLearn
using UUIDs

fit_np(X, y) = fit_predict!(RandomForestRegressor(impurity_importance = false, n_trees=100, max_depth=10, partial_sampling = 0.5, n_subfeatures = size(X, 2)), X, y)

fit_lm(X, y) =
    let X = [ones(size(X, 1)) X]
        X * (X \ y)
    end

export acfObjective, estimateACF, estimateOLS

const cache = Dict{UUID, Function}()

"""
    Given a data struct, create the objective function that we will minimize as part of the ACF procedure.

    This is separate from the estimateACF function so that a Newton step can be taken in the estimateOLS function.

"""
function acfObjective(data::Data)
    if haskey(cache, data.id)
        return cache[data.id]
    end


    ϕ = let X = [data.lnKapital[:] data.lnIntermedInput[:] data.lnLabor[:]]
        reshape(fit_np(X, data.lnOutput[:]), globals.nperiodsKeep, globals.nfirms)
    end

    lag(x::Matrix{T}) where {T<:Real} = x[1:(end-1), :][:]
    lead(x::Matrix{T}) where {T<:Real} = x[2:end, :][:]
    lag(x::Symbol) = getfield(data, x)[1:(end-1), :][:]
    lead(x::Symbol) = getfield(data, x)[2:end, :][:]

    ϕ₋₁ = lag(ϕ)
    ϕ₀ = lead(ϕ)
    k₋₁ = lag(:lnKapital)
    k₀ = lead(:lnKapital)
    l₋₁ = lag(:lnLabor)
    l₀ = lead(:lnLabor)

    out = function (θ)
        # innovation term from ω process
        ξ = let
            ω₋₁ = ϕ₋₁ - θ[1] * k₋₁ - θ[2] * l₋₁
            ω₀ = ϕ₀ - θ[1] * k₀ - θ[2] * l₀

            ω₀ - fit_lm(ω₋₁, ω₀)
        end

        moments = ξ .* [k₀ l₋₁]

        μ = mean(moments, dims=1)
        Σ = cov(moments)
        n = size(moments, 1)

        return n * only(μ * (Σ \ μ'))
    end

    empty!(cache)
    cache[data.id] = out

    out
end

estimateACF(data::Data) = @pipe data |> acfObjective |> optimize(_, [0.0, 0.0]) |> Optim.minimizer(_)

function makePolynomial(order::Integer, vecs::Vararg{Vector{T}})::Matrix{T} where {T<:Real}
    powers = fill(0:order, length(vecs))
    vector_multiply(vecs::Vararg{Vector{T}}) = broadcast(*, vecs...)

    # each term is a vector of the powers to which we want to raise each corresponding element of vecs
    mapreduce(hcat, product(powers...)) do term
        mapreduce(vector_multiply, enumerate(term)) do (i, power)
            vecs[i] .^ power
        end
    end
end

function estimateOLS(data::Data; includeKNoShock=false, newtonSteps=0)
    lag(x::Symbol) = getfield(data, x)[1:(end-1), :][:]
    lead(x::Symbol) = getfield(data, x)[2:end, :][:]

    y₀ = lead(:lnOutput)
    l₀ = lead(:lnLabor)
    k₋₁ = lag(:lnKapital)
    k₀ = lead(:lnKapital)
    m₋₁ = lag(:lnIntermedInput)
    kNoShock = lead(:lnKapitalNoShock)

    X = [k₀ l₀ makePolynomial(2, k₋₁, m₋₁)]

    if includeKNoShock
        X = [X kNoShock]
    end

    # store intial β₀ in case our optimization goes awry
    β = (X\y₀)[1:2]

    step = let f = acfObjective(data)
        α -> (α - (hessian(f, α) \ gradient(f, α)))
    end

    # do the prescribed number of Newton steps, but stop
    # early if it looks like we are heading to a minimum
    # in the negative limit
    for _ in 1:newtonSteps
        βʼ = step(β)
        if ~(βʼ[1] ∈ 0 .. 1)
            return β
        else
            β = βʼ
        end
    end

    return β
end

end
