import ..CtrlVQE: Validation
import ..CtrlVQE: Devices

function Validation.validate(
    costfn::EnergyFunction{F};
    x=nothing, rms=nothing,
    grid=nothing, device=nothing,
) where {F}
    invoke(Validation.validate, Tuple{CostFunctionType}, costfn; x=x, rms=rms)
    (isnothing(grid) || isnothing(device)) && return

    # REPRODUCE NAMESPACE FROM SUPER CALL
    L = length(costfn)
    isnothing(x) && (x = zeros(F, L))

    # FETCH SIZES
    N = Devices.nstates(device)
    nG = Devices.ngrades(device)
    nK = nobservables(costfn);                  @assert nK isa Int

    # SET UP CALLBACK TESTING
    counter = Ref(0)
    count! = (i,t,ψ) -> (counter[] += 1)

    # CHECK CALLBACK SIGNATURE AND CHAINING
    E = zeros(F, length(grid))
    callback = trajectory_callback(costfn, E; callback=count!)
    callback(0, zero(F), zeros(Complex{F}, N))
        @assert counter[] == 1

    # CHECK EXTENDED INTERFACE FOR COST FUNCTION
    counter[] = 0
    f = cost_function(costfn; callback=count!)
    f(x)
        @assert counter[] == length(grid)

    # CHECK EXTENDED INTERFACE FOR GRADIENT FUNCTION
    ϕ = fill(Inf, length(grid), nG, nK)
    g! = grad!function(costfn; ϕ=ϕ)
    g!(similar(x), x)
        @assert !any(isinf.(ϕ))

end