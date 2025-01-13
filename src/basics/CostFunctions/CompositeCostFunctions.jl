module CompositeCostFunctions
    import CtrlVQE: CostFunctions
    import CtrlVQE.CostFunctions: CostFunctionType

    """
        CompositeCostFunction(components::AbstractVector{CostFunctionType})

    The sum of several cost-functions, with matching `length` and float type.

    Use this eg. to combine an energy function with one or more penalty functions.

    This struct records the last values computed (value or gradient)
        for each cost function in its `values` and `gradients` fields.

    Beware that the `components` field is a vector with abstract eltype.
    This is a relatively mild form of type instability;
        certain compiled code could hypothetically be sub-optimal.
    But it does not appear to cause any noticeable disadvantage;
        see `Examples/CompositeCostFunctions`.

    """
    struct CompositeCostFunction{F} <: CostFunctionType{F}
        components::Vector{CostFunctionType{F}}

        values::Vector{F}       # STORES VALUES OF EACH COMPONENT FROM THE LAST CALL
        gradients::Vector{Vector{F}}
                                # STORES GRADIENTS OF EACH COMPONENT FROM THE LAST CALL

        function CompositeCostFunction(
            components::AbstractVector{<:CostFunctionType{F}},
        ) where {F}
            K = length(components)
            if K == 0
                return new{F}(CostFunctionType{F}[], F[], F[;;])
            end

            L = length(first(components))
            for component in components; @assert length(component) == L; end

            return new{F}(
                convert(Vector{CostFunctionType{F}}, components),
                Array{F}(undef, K),
                Vector{Vector{F}}(undef, K),
            )
        end
    end

    """
        CompositeCostFunction(components::CostFunctionType{F}...)

    Alternate constructor, letting each function be passed as its own argument.

    """
    function CompositeCostFunction(components::CostFunctionType{F}...) where {F}
        return CompositeCostFunction(
            CostFunctionType{F}[component for component in components]
        )
    end

    function CostFunctions.cost_function(costfn::CompositeCostFunction)
        fs = [CostFunctions.cost_function(component) for component in costfn.components]
        return (x) -> (
            total = 0;
            for (i, f) in enumerate(fs);
                fx = f(x);          # EVALUATE EACH COMPONENT
                costfn.values[i] = fx;
                total += fx;        # TALLY RESULTS FOR THE COMPONENT
            end;
            total
        )
    end

    function CostFunctions.grad!function(costfn::CompositeCostFunction{F}) where {F}
        g!s = [CostFunctions.grad!function(component) for component in costfn.components]
        costfn.gradients .= [
            Vector{F}(undef, length(component)) for component in costfn.components
        ]
        # USE THIS AS THE GRADIENT VECTOR FOR EACH COMPONENT CALL
        return (∇f, x) -> (
            ∇f .= 0;
            for (i, g!) in enumerate(g!s);
                ∇f_ = costfn.gradients[i]
                g!(∇f_, x);         # EVALUATE EACH COMPONENT. WRITES GRADIENT TO ∇f_
                ∇f .+= ∇f_;         # TALLY RESULTS FOR THE COMPONENT
            end;
            ∇f;
        )
    end

    function Base.length(costfn::CompositeCostFunction)
        isempty(costfn.components) && return 0
        L = length(first(costfn.components))
        # Confirm self-consistency.
        for component in costfn.components
            @assert length(component) == L
        end
        return L
    end
end