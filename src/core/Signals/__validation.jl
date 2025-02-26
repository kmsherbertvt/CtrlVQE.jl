using .Signals: SignalType

import ..CtrlVQE: Validation

function Validation.validate(signal::SignalType{P,R}; grid=nothing) where {P,R}
    # CHECK TYPING FUNCTIONS
    P_ = parametertype(signal);                 @assert P == P_
    R_ = returntype(signal);                    @assert R == R_

    # `Parameters` INTERFACE
    L = Parameters.count(signal);               @assert L isa Int
    x = Parameters.values(signal);              @assert x isa Vector{P}
    names = Parameters.names(signal);           @assert names isa Vector{String}
    Parameters.bind!(signal, x);    # No check. Just make sure it doesn't error.

    # SCALARS
    t = zero(P)
    f = valueat(signal, t);                     @assert f isa R
    g = [partial(k, signal, t) for k in 1:L];   @assert g isa Vector{R}

    # STRING FUNCTIONS
    name = string(signal, names);               @assert name isa String
    name_ = string(signal);                     @assert name_ isa String

    if !isnothing(grid)
        # VECTORIZED `valueat`
        fs = valueat(signal, grid);             @assert fs isa Vector{R}
        fs_ = similar(fs)
        fs__ = valueat(signal, grid; result=fs_)
        fs___ = [valueat(signal, t) for t in grid]
            @assert fs_ == fs
            @assert fs_ === fs__
            @assert fs_ === fs___

        # VECTORIZED `partial`
        if L > 0
            gs = partial(1, signal, grid);      @assert gs isa Vector{R}
            gs_ = similar(gs)
            gs__ = partial(1, signal, grid; result=gs_)
            gs___ = [partial(1, signal, t) for t in grid]
                @assert gs_ == gs
                @assert gs_ === gs__
                @assert gs_ === gs___
        end
    end
end