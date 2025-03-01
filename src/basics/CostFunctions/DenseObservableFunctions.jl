module DenseObservableFunctions
    export DenseObservable

    import ..CtrlVQE: LAT
    import ..CtrlVQE: Parameters, Operators, Bases
    import ..CtrlVQE: Integrations, Devices, Evolutions, CostFunctions

    """
        DenseObservable(observable, reference, device, basis, frame, grid, evolution)

    Calculate the expectation value of an observable after evolution of a reference state.

    # Parameters
    - `observable`: the matrix (defined on the whole Hilbert space) to measure.
    - `reference`: the initial statevector before evolution.
    - `device`: the device under which the state evolves.
    - `basis`: the basis that `observable` and `reference` are input as.
    - `frame`: the rotating frame in which `observable` is to be measured.
    - `grid`: the time grid on which the state evolves.
    - `evolution`: the algorithm to calculate the time evolution.

    """
    struct DenseObservable{
        F <: AbstractFloat,
        D <: Devices.DeviceType,
        B <: Bases.BasisType,
        R <: Operators.OperatorType,
        G <: Integrations.IntegrationType,
        A <: Evolutions.EvolutionType,
    } <: CostFunctions.EnergyFunction{F}
        observable::Matrix{Complex{F}}
        reference::Vector{Complex{F}}
        device::D
        basis::B
        frame::R
        grid::G
        evolution::A
    end

    Base.length(costfn::DenseObservable) = Parameters.count(costfn.device)

    function CostFunctions.cost_function(costfn::DenseObservable; callback=nothing)
        T = Integrations.endtime(costfn.grid)
        OT = copy(costfn.observable)
        Devices.evolve!(costfn.frame, costfn.device, costfn.basis, T, OT)
        ψ = similar(costfn.reference)

        return (x̄) -> (
            Parameters.bind!(costfn.device, x̄);
            Evolutions.evolve(
                costfn.evolution,
                costfn.device,
                costfn.basis,
                costfn.grid,
                costfn.reference;
                result=ψ,
                callback=callback,
            );
            real(LAT.expectation(OT, ψ))
        )
    end

    function CostFunctions.grad!function(costfn::DenseObservable{F}; ϕ=nothing) where {F}
        nG = Devices.ngrades(costfn.device)
        isnothing(ϕ) && (ϕ=Array{F}(undef, length(costfn.grid), nG, 1))
        ϕ = reshape(ϕ, length(costfn.grid), nG)

        T = Integrations.endtime(costfn.grid)
        OT = copy(costfn.observable)
        Devices.evolve!(costfn.frame, costfn.device, costfn.basis, T, OT)

        return (∇f, x) -> (
            Parameters.bind!(costfn.device, x);
            Evolutions.gradientsignals(
                costfn.evolution,
                costfn.device,
                costfn.basis,
                costfn.grid,
                costfn.reference,
                OT;
                result=ϕ,   # NOTE: This writes the gradient signal as needed.
            );
            Devices.gradient(costfn.device, costfn.grid, ϕ; result=∇f)
        )
    end

    CostFunctions.nobservables(::DenseObservable) = 1

    function CostFunctions.trajectory_callback(
        costfn::DenseObservable,
        E::AbstractVector;
        callback=nothing,
    )
        workbasis = Evolutions.workbasis(costfn.evolution)  # BASIS OF CALLBACK ψ
        U = Devices.basisrotation(costfn.basis, workbasis, costfn.device)
        ψ_ = similar(costfn.reference)

        return (i, t, ψ) -> (
            ψ_ .= ψ;
            LAT.rotate!(U, ψ_);     # ψ_ IS NOW IN MEASUREMENT BASIS
            Devices.evolve!(costfn.frame, costfn.device, costfn.basis, -t, ψ_);
                                    # ψ_ IS NOW IN MEASUREMENT FRAME
            E[1+i] = real(LAT.expectation(costfn.observable, ψ_));
            !isnothing(callback) && callback(i, t, ψ)
        )
    end

end