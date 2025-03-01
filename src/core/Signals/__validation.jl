import ..CtrlVQE: Validation
import ..CtrlVQE.Validation: @withresult

import ..CtrlVQE: Parameters

import FiniteDifferences

function Validation.validate(
    signal::SignalType{P,R};
    grid=nothing,
    t=zero(P),
    rms=nothing,
) where {P,R}
    Parameters.validate(signal)
    L = Parameters.count(signal)

    # CHECK TYPING FUNCTIONS
    P_ = parametertype(signal);                 @assert P == P_
    R_ = returntype(signal);                    @assert R == R_

    # CHECK TYPINGS
    f0 = valueat(signal, t);                    @assert f0 isa R
    g0 = [partial(k, signal, t) for k in 1:L];  @assert g0 isa Vector{R}

    # STRING FUNCTIONS
    name = string(signal, names);               @assert name isa String
    name_ = string(signal);                     @assert name_ isa String

    # CHECK VECTORIZED FUNCTIONS
    if !isnothing(grid)
        fs = @withresult valueat(signal, grid);             @assert fs isa Vector{R}
        fs_ = [valueat(signal, grid[i]) for i in eachindex(grid)]
            @assert fs == fs_

        if L > 0
            gs = @withresult partial(1, signal, grid);      @assert gs isa Vector{R}
            gs_ = [partial(1, signal, grid[i]) for i in eachindex(grid)]
                @assert gs == gs_
        end
    end

    # CHECK FINITE DIFFERENCE
    if !isnothing(rms)
        cfd = FiniteDifferences.central_fdm(2,1)
        x0 = Parameters.values(signal)

        gΔℝ = FiniteDifferences.grad(cfd, x -> (
            Parameters.bind!(signal, x);
            fx = real(Signals.valueat(signal, t));
            Parameters.bind!(signal, x0);
            fx
        ), x0)[1]

        if R <: Complex
            gΔℂ = FiniteDifferences.grad(cfd, x -> (
                Parameters.bind!(signal, x);
                fx = imag(Signals.valueat(signal, t));
                Parameters.bind!(signal, x0);
                fx
            ), x0)[1]
        end
        gΔ = R <: Complex ? Complex.(gΔℝ, gΔℂ) : gΔℝ

        ε = √(sum(abs2.(g0.-gΔ)) ./ L)
        @assert ε < rms
    end

end