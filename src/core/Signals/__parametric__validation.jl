using .Signals: SignalType, ParametricSignal

import ..CtrlVQE: Validation

function Validation.validate(signal::ParametricSignal{P,R}) where {P,R}
    invoke(Validation.validate, Tuple{SignalType}, signal)

    # CHECK PARAMETER TYPINGS
    fields = parameters(signal)
    for field in fields
        @assert getfield(signal, field) isa P
    end
end