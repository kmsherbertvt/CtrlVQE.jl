import .Signals: SignalType, ParametricSignal

import ..CtrlVQE: Parameters
import ..CtrlVQE: Signals

"""
    ConstrainedSignal(template::<:ParametricSignal, constraints::Vector{Symbol})

The parametric signal `template`, freezing all fields specified by `constraints`.

Frozen parameters are omitted from the `Parameters` interface.
In other words, they do not appear in `Parameters.names` or `Parameters.values`,
    and they are not mutated by `Parameters.bind!`.

# Example

Say you have a `Trigonometric` signal sub-typing `ParametricSignal`,
    where all fields amplitude `A`, phase `ϕ`, and frequency `ν`
    are registered as variational parameters.
You want to run an optimization where the frequency `ν` is fixed to 4.608.
Rather than implementing a whole new ParametricSignal
    identical except for a different implementation of the `parameters` function,
    use a ConstrainedSignal:

    template = Trigonometric(0.0, 0.0, 4.608)   # Initialize A and ϕ to 0.0
    signal = Constrained(template, :ν)

Note that the `Constrained` constructor above is just syntactic sugar for:

    signal = ConstrainedSignal(template, [:ν])

"""
struct ConstrainedSignal{P,R,S<:ParametricSignal{P,R}} <: SignalType{P,R}
    template::S
    constraints::Vector{Symbol}
    _map::Vector{Int}

    function ConstrainedSignal(
        template::S,
        constraints::Vector{Symbol}
    ) where {P,R,S<:ParametricSignal{P,R}}
        fields = Signals.parameters(template)
        _map = [j for (j, field) in enumerate(fields) if field ∉ constraints]
        return new{P,R,S}(template, constraints, _map)
    end
end

#= CONSTRUCTOR ALIAS =#

"""
    Constrained(template::ParametricSignal, constraints::Symbol...)

Construct a `ConstrainedSignal` from a `ParametricSignal` and the fields to freeze.

"""
function Constrained(template::ParametricSignal, constraints::Symbol...)
    return ConstrainedSignal(template, collect(constraints))
end

#= `Parameters` INTERFACE =#

function Parameters.count(signal::ConstrainedSignal)
    return Parameters.count(signal.template) - length(signal.constraints)
end

function Parameters.names(signal::ConstrainedSignal)
    return Parameters.names(signal.template)[collect(signal._map)]
end

function Parameters.values(signal::ConstrainedSignal)
    return Parameters.values(signal.template)[collect(signal._map)]
end

function Parameters.bind!(
    signal::ConstrainedSignal{P,R,S},
    x̄::AbstractVector{P}
) where {P,R,S}
    fields = Signals.parameters(signal.template)
    for k in eachindex(x̄)
        setfield!(signal.template, fields[signal._map[k]], x̄[k])
    end
    return signal
end

#= `Signals` INTERFACE =#

function Signals.valueat(signal::ConstrainedSignal, t::Real)
    return Signals.valueat(signal.template, t)
end

function Signals.partial(k::Int, signal::ConstrainedSignal, t::Real)
    return Signals.partial(signal._map[k], signal.template, t)
end

function Base.string(signal::ConstrainedSignal, names::AbstractVector{String})
    newnames = string.(Parameters.values(signal.template))
    for k in eachindex(names)
        newnames[signal._map[k]] = names[k]
    end
    return Base.string(signal.template, newnames)
end