module DrivePenalties
    import ..ModularFramework as Modular

    import CtrlVQE: Parameters
    import CtrlVQE: CostFunctions
    import CtrlVQE.Devices: DeviceType
    import CtrlVQE.CostFunctions: CostFunctionType

    import LinearAlgebra: mul!

    """
        DrivePenalty(device, penalties::AbstractVector{<:CostFunctionType})

    The sum of several penalty functions acting on a device's drives.

    In order for this type to be usable,
        each penalty in `penalties` must act on parameters corresponding to
        the drives contained in `device.drives`,
        and device parameters must be mapped onto drives according to `device.pmap`.
    This interface is suitable for the `Device` types in `ModularFramework`.

    This struct records the last values computed (value or gradient)
        for each cost function in its `values` and `gradients` fields.

    Beware that the `components` field is a vector with abstract eltype.
    This is a relatively mild form of type instability;
        certain compiled code could hypothetically be sub-optimal.
    But it does not appear to cause any noticeable disadvantage;
        see `Examples/CompositeCostFunctions`.

    """
    struct DrivePenalty{
        F,
        D<:DeviceType{F},
        Λ<:CostFunctionType{F},
    } <: CostFunctionType{F}
        device::D               # DEVICE TYPE
        penalties::Vector{Λ}    # PENALTY TYPE

        values::Vector{F}       # STORES VALUES OF EACH COMPONENT FROM THE LAST CALL
        gradients::Vector{Vector{F}}
                                # STORES GRADIENTS OF EACH COMPONENT FROM THE LAST CALL

        function DrivePenalty(
            device::DeviceType{F},
            penalties::AbstractVector{<:CostFunctionType{F}},
        ) where {F}
            # THERE MUST BE ONE PENALTY FOR EACH DRIVE IN `device`
            # AND `device` MUST HAVE DRIVES ORGANIZED INTO A `drives` VECTOR
            @assert length(penalties) == length(device.drives)

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

    function CostFunctions.cost_function(costfn::DrivePenalty)
        Λs = [CostFunctions.cost_function(penalty) for penalty in costfn.penalties]

        L = maximum(Parameters.count, costfn.device.drives)
        y_ = Array{eltype(costfn)}(undef, L)

        return (x) -> (
            Parameters.bind!(costfn.device, x);
            total = 0;
            # Δ = 0;      # PARAMETER OFFSET
            for (i, Λ) in enumerate(Λs);
                drive = costfn.device.drives[i];
                y = @view(y_[1:Parameters.count(drive)]);
                Modular.map_values(costfn.device.pmap, costfn.device, i; result=y);
                fx = Λ(y);          # EVALUATE EACH COMPONENT
                costfn.values[i] = fx;
                total += fx;        # TALLY RESULTS FOR THE COMPONENT
            end;
            total
        )
    end

    function CostFunctions.grad!function(costfn::DrivePenalty{F}) where {F}
        g!s = [CostFunctions.grad!function(penalty) for penalty in costfn.penalties]
        costfn.gradients .= [
            Vector{F}(undef, length(penalty)) for penalty in costfn.penalties
        ]

        L = maximum(Parameters.count, costfn.device.drives)
        y_ = Array{eltype(costfn)}(undef, L)
        g_ = Array{eltype(costfn)}(undef, length(costfn.device.x), L)

        # USE THIS AS THE GRADIENT VECTOR FOR EACH COMPONENT CALL
        return (∇f, x) -> (
            Parameters.bind!(costfn.device, x);
            ∇f .= 0;
            for (i, g!) in enumerate(g!s);
                drive = costfn.device.drives[i];

                # MAP DEVICE PARAMETERS TO DRIVE PARAMETERS
                y = @view(y_[1:Parameters.count(drive)]);
                Modular.map_values(costfn.device.pmap, costfn.device, i; result=y);

                # COMPUTE GRADIENT WITH RESPECT TO DRIVE PARAMETERS
                ∂y = costfn.gradients[i];
                g!(∂y, y);             # EVALUATE EACH COMPONENT. WRITES GRADIENT TO ∂y

                # CONSTRUCT GRADIENT OF DRIVE PARAMETERS WITH RESPECT TO DEVICE PARAMETERS
                g = @view(g_[:,1:length(∂y)]);
                Modular.map_gradients(costfn.device.pmap, costfn.device, i; result=g);

                # APPLY CHAIN RULE
                mul!(∇f, g, ∂y, 1, 1);  # Computes matrix product 1g*1∂y, adds to ∇f.
            end;
            ∇f
        )
    end

    function Base.length(costfn::DrivePenalty)
        return Parameters.count(costfn.device)
    end

    #= TODO: We still need a ligand to connect each kind of drive
        to the basic signal penalties. =#
end