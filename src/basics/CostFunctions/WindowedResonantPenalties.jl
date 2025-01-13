module WindowedResonantPenalties
    import CtrlVQE: Parameters
    import CtrlVQE: Devices, CostFunctions

    import CtrlVQE.Devices: DeviceType
    import CtrlVQE.CostFunctions: CostFunctionType

    # NOTE: Implicitly use smooth bounding function.
    wall(u) = exp(u - 1/u)
    grad(u) = exp(u - 1/u) * (1 + 1/u^2)

    """
        WindowedResonantPenalty(device, A::Vector, σ::Vector, λ::Vector)

    Compute penalties for exceeding the maximum amplitude of a device.

    Let ``u=\frac{|Ω|-A}{σ}``.
    When ``u>0`` the penalty is computed as ``Λ = λ \exp(u - 1/u)``.

    # Parameters
    - `device`: a compatible device (see below).
    - `A`: the maximum modulus of any amplitude, on each drive.
    - `σ`: the steepness parameter of the penalty function, on each drive.
    - `λ`: the strength parameter of the penalty function, on each drive

    # Device Compatibility

    Intended for use with the `RWRTDevice` and `CWRTDevice` provided in the basics,
        this type should work if:
    - `device` has a field `Ω`
    - `Ω` is a matrix of (potentially complex) amplitudes, with `n` columns,
        where `n = Devices.nqubits(device)`.
    - `Parameters.values(x)` is a vectorization of `Ω` (and nothing else),
        reinterpreted as a vector of floats when `Ω` is complex.

    """
    struct WindowedResonantPenalty{F,D<:DeviceType{F}} <: CostFunctionType{F}
        device::D       # DEVICE TYPE
        A::Vector{F}    # MAXIMUM MODULUS
        σ::Vector{F}    # STEEPNESS OF PENALTY
        λ::Vector{F}    # STRENGTH OF PENALTY
    end

    """
        WindowedResonantPenalty(device; A=1.0, σ=A, λ=1.0)

    Alternate constructor, with kwargs and sensible defaults.

    Each parameter may be provided as a vector or as a scalar,
        in which case it is automatically expanded to a vector
        where the given value is assigned to each qubit.

    We have selected σ defaulting to A because heuristically it seems to work well.
    Don't feel too attached to that choice.

    """
    function WindowedResonantPenalty(
        device::DeviceType{F};
        A::Union{F,Vector{F}}=one(F),
        σ::Union{F,Vector{F}}=deepcopy(A),
        λ::Union{F,Vector{F}}=one(F),
    ) where {F}
        L = Parameters.count(device)
        A_ = A isa F ? fill(A,L) : A
        σ_ = σ isa F ? fill(A,L) : σ
        λ_ = λ isa F ? fill(λ,L) : λ
        return WindowedResonantPenalty(device, A_, σ_, λ_)
    end

    function CostFunctions.cost_function(costfn::WindowedResonantPenalty)
        return (x) -> (
            Parameters.bind!(costfn.device, x);
            Ω = costfn.device.Ω;
            Λ = 0;
            for w in axes(Ω,1);
            for q in axes(Ω,2);
                r = abs(Ω[w,q]);
                u = (r - costfn.A[q]) / costfn.σ[q];
                u > 0 && (Λ += costfn.λ[q] * wall(u));
            end; end;
            Λ
        )
    end

    function CostFunctions.grad!function(costfn::WindowedResonantPenalty)
        return (∇f, x) -> (
            Parameters.bind!(costfn.device, x);
            Ω = costfn.device.Ω;
            s = eltype(Ω) <: Complex ? 2 : 1;
            W = size(Ω,1);

            ∇f .= 0;
            for w in axes(Ω,1);
            for q in axes(Ω,2);
                r = abs(Ω[w,q]);
                u = (r - costfn.A[q]) / costfn.σ[q];
                if u > 0;
                    i = 1 + (q-1)*W*s + (w-1)*s;
                    ∇f[i]   += costfn.λ[q] * grad(u) / costfn.σ[q] * (real(Ω[w,q])/r);
                    s == 1 && continue;     # REAL-VALUED AMPLITUDES
                    ∇f[i+1] += costfn.λ[q] * grad(u) / costfn.σ[q] * (imag(Ω[w,q])/r);
                end;
            end; end;
            ∇f
        )
    end

    function Base.length(costfn::WindowedResonantPenalty)
        return Parameters.count(costfn.device)
    end

end