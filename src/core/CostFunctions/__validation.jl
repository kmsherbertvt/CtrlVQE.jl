import ..CtrlVQE: Validation

import FiniteDifferences

function Validation.validate(
    costfn::CostFunctionType{F};
    x=nothing, rms=nothing,
) where {F}
    # CHECK BASE OVERRIDES
    F_ = eltype(costfn);            @assert F == F_
    L = length(costfn);             @assert L isa Int

    # MAKE A DUMMY PARAMETER VECTOR
    isnothing(x) && (x = zeros(F, L))
    g0 = similar(x)

    # CHECK CONSISTENCY OF FUNCTION EVALUATION
    f = cost_function(costfn)
        @assert f(x) == costfn(x)

    # CHECK CONSISTENCY OF GRADIENT EVALUATION
    g! = grad!function(costfn); g0_ = g!(g0, x)
    g  = grad_function(costfn); g0__ = g(x)
        @assert g0 === g0_
        @assert g0 == g0__

    # CHECK ACCURACY OF GRADIENT EVALUATION
    if !isnothing(rms)
        cfd = FiniteDifferences.central_fdm(2,1)
        gΔ = FiniteDifferences.grad(cfd, f, x)[1]
        ε = √(sum(abs2.(g0.-gΔ)) ./ L)
        @assert ε < rms
    end
end
