"""
    Validation

Standardized interface for ensuring user-defined types are consistent with the interface.

"""
module Validation
    import ..CtrlVQE: Parameters

    """
        validate(type)

    Check that `type` is consistent with the interface defined by its supertype.

    Each abstract type defined in the core interface implements its own `validate`,
        running a suite of interface-compliance and self-consistency checks.

    Each type accepts a different collection of keyword arguments:

        validate(::IntegrationType)
        validate(::SignalType; grid::IntegrationType, t, rms)
        validate(::DeviceType; grid::IntegrationType, t)
        validate(::EvolutionType; grid::IntegrationType, device::DeviceType, skipgradient)
        validate(::CostFunctionType; x, rms)
        validate(::EnergyFunction; grid::IntegrationType, device::DeviceType, x, rms)

    - If `grid` and `device` are omitted, relevant tests will be skipped.
      - Note in particular that `validate(::EvolutionType)`
        requires both in order to validate any evolution at all!
      - Also note that `validate(::EnergyFunction)`
        requires both for any validations special to energy functions,
        and they must be consistent with those used to construct the energy function,
        but if omitted the usual cost function tests will still occur.
    - `t` (a single time) and `x` (a vector of parameters)
            determine the point at which accuracy checks are conducted.
        They default to zero when omitted.
    - `rms` is the maximum root mean square error permitted
            between analytical gradients and finite differences.
        It defaults to `nothing`, which skips the finite difference altogether.
    - `skipgradient` is a flag indicating whether to skip anlaytical gradient validation,
        useful for evolution types that don't bother to implement `gradientsignals`.
        TODO: Also for CostFunctions?


    """
    function validate end

    """
        result = @withresult fn(args...; kwargs...)

    Validate that a function accepting an optional `result` kwarg
        calculates the same value with and without `result`,
        and that, with `result`, the object returned is identical to the one passed.

    """
    macro withresult(thecall)
        # CREATE A VERSION OF THE CALL INCLUDING `result=result` AS A KEYWORD ARGUMENT
        resultform = deepcopy(thecall)
        args = resultform.args
        haskwargs = length(args) > 1 && args[2] isa Expr && args[2].head == :parameters
        kwargexpr = haskwargs ? args[2] : Expr(:parameters)
        haskwargs || insert!(args, 2, kwargexpr)
        push!(kwargexpr.args, Expr(:kw, :result, :result))

        return esc(quote
            answer = $thecall           # fn(args...; kwargs...)
            result = similar(answer)
            result_ = $resultform       # fn(args...; kwargs..., result=result)
            @assert pointer(result) == pointer(result_)
                                        # result kwarg is the data returned
            @assert result == answer    # result kwarg matches answer sans result
            answer
        end)
    end

end