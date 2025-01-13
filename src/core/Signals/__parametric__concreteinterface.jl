using .Signals: ParametricSignal
using .Signals: parameters

import ..CtrlVQE: Parameters

Parameters.count(signal::S) where {S<:ParametricSignal} = length(parameters(signal))

function Parameters.names(signal::S) where {S<:ParametricSignal}
    return [string(field) for field in parameters(signal)]
end

function Parameters.values(signal::S) where {P,R,S<:ParametricSignal{P,R}}
    return P[getfield(signal, field) for field in parameters(signal)]
end

function Parameters.bind!(
    signal::S,
    x̄::AbstractVector{P}
) where {P,R,S<:ParametricSignal{P,R}}
    for (k, field) in enumerate(parameters(signal))
        setfield!(signal, field, x̄[k])
    end
    return signal
end