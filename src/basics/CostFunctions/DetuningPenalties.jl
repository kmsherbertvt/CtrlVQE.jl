module DetuningPenalties
    import CtrlVQE: Parameters
    import CtrlVQE: CostFunctions
    import CtrlVQE.Devices: DeviceType
    import CtrlVQE.CostFunctions: CostFunctionType

    """
        DetuningPenalty(device::DeviceType, penalties::AbstractVector{<:CostFunctionType})

    The sum of several penalty functions acting on a device's detuning signals.

    In order for this type to be usable,
        each penalty in `penalties` must act on parameters corresponding to
        the signals contained in `device.Δ`,
        and parameters in `device` must be disjoint,
        listing parameters for each `device.Ω` then each `device.Δ`.
    This interface is suitable for the basic `TransmonDevice`.

    This struct records the last values computed (value or gradient)
        for each cost function in its `values` and `gradients` fields.

    Beware that the `components` field is a vector with abstract eltype.
    This is a relatively mild form of type instability;
        certain compiled code could hypothetically be sub-optimal.
    But it does not appear to cause any noticeable disadvantage;
        see `Examples/CompositeCostFunctions`.

    ```jldoctests
    julia> grid = TemporalLattice(20.0, 400);

    julia> device = TransmonDevice{2}([4.82, 4.84], [0.00, 0.00], [0.02], [Quple(1,2)]; Ω=Constrained(Constant(zero(ComplexF64)), :B), Δ=Constant(zero(Float64)));

    julia> penalties = [SignalStrengthPenalty(grid, signal; A=0.8) for signal in device.Δ];

    julia> costfn = DetuningPenalty(device, penalties);

    julia> x = collect(range(0.0, 1.0, length(costfn)))
    4-element Vector{Float64}:
     0.0
     0.3333333333333333
     0.6666666666666666
     1.0

    julia> validate(costfn; x=x, rms=1e-6);

    julia> costfn(x)
    0.02351774585600897
    julia> grad_function(costfn)(x)
    4-element Vector{Float64}:
     0.0
     0.0
     0.0
     0.4997520994401937

    ```

    """
    struct DetuningPenalty{
        F,
        D<:DeviceType{F},
        Λ<:CostFunctionType{F},
    } <: CostFunctionType{F}
        device::D               # DEVICE TYPE
        penalties::Vector{Λ}    # PENALTY TYPE

        values::Vector{F}       # STORES VALUES OF EACH COMPONENT FROM THE LAST CALL
        gradients::Vector{Vector{F}}
                                # STORES GRADIENTS OF EACH COMPONENT FROM THE LAST CALL

        function DetuningPenalty(
            device::DeviceType{F},
            penalties::AbstractVector{<:CostFunctionType{F}},
        ) where {F}
            # THERE MUST BE ONE PENALTY FOR EACH DRIVE IN `device`
            # AND `device` MUST HAVE DRIVES ORGANIZED INTO `Ω` AND `Δ` VECTORS
            @assert length(penalties) == length(device.Δ)

            D = typeof(device)
            Λ = eltype(penalties)
            return new{F,D,Λ}(
                device,
                convert(Vector{Λ}, penalties),
                Vector{F}(undef, length(penalties)),
                Vector{Vector{F}}(undef, length(penalties)),
            )
        end
    end

    function CostFunctions.cost_function(costfn::DetuningPenalty)
        Λs = [CostFunctions.cost_function(penalty) for penalty in costfn.penalties]
        return (x) -> (
            total = 0;
            Δ = sum(Parameters.count, costfn.device.Ω); # PARAMETER OFFSET
            for (i, Λ) in enumerate(Λs);
                L = length(costfn.penalties[i]);
                xΛ = @view(x[Δ+1:Δ+L]);
                fx = Λ(xΛ);         # EVALUATE EACH COMPONENT
                costfn.values[i] = fx;
                total += fx;        # TALLY RESULTS FOR THE COMPONENT
                Δ += L;
            end;
            total
        )
    end

    function CostFunctions.grad!function(costfn::DetuningPenalty{F}) where {F}
        g!s = [CostFunctions.grad!function(penalty) for penalty in costfn.penalties]
        costfn.gradients .= [
            Vector{F}(undef, length(penalty)) for penalty in costfn.penalties
        ]

        # USE THIS AS THE GRADIENT VECTOR FOR EACH COMPONENT CALL
        return (∇f, x) -> (
            ∇f .= 0;
            Δ = sum(Parameters.count, costfn.device.Ω); # PARAMETER OFFSET
            for (i, g!) in enumerate(g!s);
                ∇f_ = costfn.gradients[i];
                L = length(costfn.penalties[i]);
                xΛ = @view(x[Δ+1:Δ+L]);
                g!(∇f_, xΛ);            # EVALUATE EACH COMPONENT. WRITES GRADIENT TO ∇f_
                ∇f[Δ+1:Δ+L] .+= ∇f_;    # TALLY RESULTS FOR THE COMPONENT
                Δ += L;
            end;
            ∇f
        )
    end

    function Base.length(costfn::DetuningPenalty)
        return Parameters.count(costfn.device)
    end
end