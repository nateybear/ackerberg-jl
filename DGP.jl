module DGP

using Random: randn


export globals, Data, generateData, setDGP

############## GLOBALS STRUCTS DEFINITION ###################

Base.@kwdef mutable struct Globals
    nfirms::Integer = 1000
    nperiodsTotal::Integer = 111
    nperiodsKeep::Integer = 11
    niterations::Integer = 10

    # the true coefficient values
    beta0::Float64 = 1
    betaK::Float64 = 0.4
    betaL::Float64 = 0.6
    betaM::Float64 = 1
    
    # setup for investment optimization. sigmaPhi is across-firm
    # variation in adjustment costs
    sigmaPhi::Float64 = 0.6
    discountRate::Float64 = 0.95
    deprecRate::Float64 = 0.2
    
    # set up for wage process. variance of the innovation term must be
    # backed out from the desired overall process variance.
    rhoWage::Float64 = 0.7
    
    # set up for omega processes. this time have to decompose omega into
    # the partially observed omega_{t-b} where firms make their investment
    # decision
    sigmaOmega::Float64 = 0.3
    rhoOmega::Float64 = 0.7

    # the things below are set by the setDGP function
    dgp::Float64 = 1
    measureError::Float64 = 0
    sigmaLogK::Float64 = 0

    sigmaLogWage::Float64 = 0
    timeB::Float64 = 0
    sigmaOptimErrL::Float64 = 0
    sigmaXiWage::Float64 = 0
    rhoOmega1::Float64 = 0
    rhoOmega2::Float64 = 0
    sigmaXiOmega::Float64 = 0
    sigmaXiOmega1::Float64 = 0
    sigmaXiOmega2::Float64 = 0
end

const globals = Globals()

function setDGP(;
    dgp=globals.dgp, 
    measureError=globals.measureError, 
    sigmaLogK=globals.sigmaLogK
)

    # set DGP-specific vars
    globals.dgp = dgp
    if dgp == 1
        globals.sigmaLogWage = 0.1
        globals.timeB = 0.5
        globals.sigmaOptimErrL = 0.
    elseif dgp == 2
        globals.sigmaLogWage = 0.
        globals.timeB = 0.
        globals.sigmaOptimErrL = 0.37
    else
        globals.sigmaLogWage = 0.1
        globals.timeB = 0.5
        globals.sigmaOptimErrL = 0.37
    end
    
    globals.measureError = measureError
    globals.sigmaLogK = sigmaLogK

    globals.sigmaXiWage = sqrt((1-globals.rhoWage^2) * globals.sigmaLogWage^2)
    globals.rhoOmega1 = globals.rhoOmega^(1-globals.timeB)
    globals.rhoOmega2 = globals.rhoOmega^globals.timeB
    globals.sigmaXiOmega = sqrt((1-globals.rhoOmega^2)*globals.sigmaOmega^2)
    globals.sigmaXiOmega1 = sqrt((1-globals.rhoOmega1^2)*globals.sigmaOmega^2)
    globals.sigmaXiOmega2 = sqrt((1-globals.rhoOmega2^2)*globals.sigmaOmega^2)

    return(nothing)
end

setDGP()


################### DATA STRUCT DEFINITION #######################

mutable struct Data
    lnLabor::Matrix{Float64} 
    lnKapital::Matrix{Float64} 
    lnKapitalNoShock::Matrix{Float64} 
    lnOutput::Matrix{Float64}
    lnInvestment::Matrix{Float64} 
    lnOutputPrice::Matrix{Float64} 
    lnWage::Matrix{Float64} 
    lnIntermedInput::Matrix{Float64} 
    omegaT::Matrix{Float64} 
    omegaTminusB::Matrix{Float64} 
    epsilon::Matrix{Float64}
    Data() = begin
        init = ( zeros(globals.nperiodsTotal, globals.nfirms) for _ in fieldnames(Data) )
        new(init...)
    end
end

function generateExogenousShocks(data::Data)
    sigmaEpsilon = 0.1
    data.epsilon = randn(globals.nperiodsTotal, globals.nfirms) * sigmaEpsilon

    Xi1 = globals.sigmaXiOmega1 * randn(globals.nperiodsTotal, globals.nfirms)
    Xi2 = globals.sigmaXiOmega2 * randn(globals.nperiodsTotal, globals.nfirms)

    data.omegaTminusB[1,:] = Xi1[1,:]
    data.omegaT[1,:] = Xi2[1,:]
    for period = 2:globals.nperiodsTotal
        data.omegaTminusB[period,:] = globals.rhoOmega1 * data.omegaT[period-1,:] + Xi1[period,:]
        data.omegaT[period, :] = globals.rhoOmega2 * data.omegaTminusB[period, :] + Xi2[period,:]
    end

    data
end

function generateWages(data::Data)
    Xi = globals.sigmaXiWage * randn(globals.nperiodsTotal, globals.nfirms)
    
    data.lnWage[1,:] = Xi[1,:]
    
    for period = 2:globals.nperiodsTotal
        data.lnWage[period,:] = globals.rhoWage * data.lnWage[period-1,:] + Xi[period,:]
    end

    data
end

function calculateInvestmentDemand(data::Data)
    # these are 1/phi_i for each firm
    adjustmentTerm = exp.(globals.sigmaPhi * randn(1, globals.nfirms));
    
    
    # different parts of the analytical solution to investment problem
    squareBracketTerm = (globals.betaL^(globals.betaL/(1-globals.betaL))) *
        exp(0.5 * globals.betaL^2 * globals.sigmaOptimErrL^2) -
        (globals.betaL^(1/(1-globals.betaL))) * exp(0.5 * globals.sigmaOptimErrL^2)
    
    const1 = globals.discountRate * (globals.betaK/(1-globals.betaL)) * 
        exp(globals.beta0)^(1/(1-globals.betaL)) * squareBracketTerm
    
    vec1 = (globals.discountRate * (1-globals.deprecRate)).^(0:100)
    vec2 = cumsum(globals.rhoWage.^(2 * (0:100)))
    vec3 = globals.sigmaXiOmega^2 * [ 0; cumsum(globals.rhoOmega.^(2 * (0:99)))]
    
    expterm3 = exp.(0.5 * (-globals.betaL/(1-globals.betaL))^2 * globals.sigmaXiWage^2 * vec2)
    expterm4 = exp.(0.5 * (1/(1-globals.betaL))^2 * globals.rhoOmega2^2 *
        (globals.sigmaXiOmega1^2 * (vec3 .+ globals.rhoOmega.^(2 * (0:100)))))
    expterm5 = exp((1/(1-globals.betaL)) * globals.sigmaXiOmega2^2 / 2)
    
    
    
    # now for the actual loop step
    # start with a small initial amount of capital
    data.lnKapital[1,:] = -100 * ones(1, globals.nfirms)
    
    capitalShock = globals.sigmaLogK * randn(globals.nperiodsTotal, globals.nfirms)
    
    for period=2:globals.nperiodsTotal
        # increment capital (and introduce some noise depending on the scenario)
        data.lnKapitalNoShock[period, :] = log.((1-globals.deprecRate) *
            exp.(data.lnKapital[period-1, :]) + exp.(data.lnInvestment[period-1, :]))
        
        data.lnKapital[period, :] = data.lnKapitalNoShock[period, :] + capitalShock[period, :]
        
        
        expterm1 = exp.((1/(1-globals.betaL)) * globals.rhoOmega.^(1:101) .* data.omegaT[period, :]')
        expterm2 = exp.(-globals.betaL/(1-globals.betaL) * globals.rhoWage.^(1:101) * data.lnWage[period,:]')
        
        data.lnInvestment[period, :] = log.(adjustmentTerm * const1 * expterm5 .* 
            sum(vec1 .* expterm1 .* expterm2 .* expterm3 .* expterm4, dims = 1))
    end

    data
end

function calculateLaborDemand(data::Data)
    @. data.lnLabor = ((globals.sigmaXiOmega2^2)/2 + 
        log(globals.betaL) + globals.beta0 + globals.rhoOmega2 *
        data.omegaTminusB - data.lnWage + 
        globals.betaK * data.lnKapital)/(1-globals.betaL)

    data
end

function generateIntermediateInputDemand(data::Data)
    trueLabor = data.lnLabor
    optimError = globals.sigmaOptimErrL * randn(globals.nperiodsTotal, globals.nfirms)
    data.lnLabor = trueLabor + optimError;
    
    @. data.lnIntermedInput = globals.beta0 + globals.betaL * trueLabor +
        globals.betaK * data.lnKapital + data.omegaT

    data
end

function calculateFirmOutput(data::Data)
    @. data.lnOutput = globals.beta0 + globals.betaL * data.lnLabor +
        globals.betaK * data.lnKapital + data.omegaT + data.epsilon

    data
end

function keepLastN(data::Data)
    for field in fieldnames(Data)
        lastN = getfield(data, field)[(end - globals.nperiodsKeep + 1):end, :]
        setfield!(data, field, lastN)
    end

    data
end

const pipeline = reduce(∘, reverse([ 
    generateExogenousShocks, 
    generateWages,
    calculateInvestmentDemand,
    calculateLaborDemand, 
    generateIntermediateInputDemand,
    calculateFirmOutput,
    keepLastN
]))

generateData() = Data() |> pipeline

end