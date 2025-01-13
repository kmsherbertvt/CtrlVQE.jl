module DenseLeakageFunctions
    export DenseLeakage

    import ..CtrlVQE: LAT, QubitOperations
    import ..CtrlVQE: Parameters, Operators, Bases
    import ..CtrlVQE: Integrations, Devices, Evolutions, CostFunctions

    """
        DenseLeakage(reference, device, basis, frame, grid, evolution)

    Calculate leakage of a reference state after evolution.

    Leakage is the probability of finding any qubit outside the |0⟩,|1⟩ subspace.

    # Parameters
    - `reference`: the initial statevector before evolution.
    - `device`: the device under which the state evolves.
    - `basis`: the basis that `reference` is input as,
        and the basis which for which leakage is defined.
    - `frame`: the rotating frame for which leakage is defined.
    - `grid`: the time grid on which the state evolves.
    - `evolution`: the algorithm to calculate the time evolution.

    """
    struct DenseLeakage{
        F <: AbstractFloat,
        D <: Devices.DeviceType{F},
        B <: Bases.BasisType,
        R <: Operators.OperatorType,
        G <: Integrations.IntegrationType{F},
        A <: Evolutions.EvolutionType,
    } <: CostFunctions.EnergyFunction{F}
        reference::Vector{Complex{F}}
        device::D
        basis::B
        frame::R
        grid::G
        evolution::A
    end

    Base.length(costfn::DenseLeakage) = Parameters.count(costfn.device)

    function CostFunctions.cost_function(costfn::DenseLeakage; callback=nothing)
        m = Devices.nlevels(costfn.device)
        n = Devices.nqubits(costfn.device)
        π̄ = QubitOperations.localqubitprojectors(m, n)
        T = Evolutions.endtime(costfn.grid)
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
            Devices.evolve!(costfn.frame, costfn.device, costfn.basis, -T, ψ);
                                    # ψ IS NOW IN MEASUREMENT FRAME
            1 - real(LAT.expectation(π̄, ψ))
        )
    end

    function CostFunctions.grad!function(costfn::DenseLeakage{F}; ϕ=nothing) where {F}
        isnothing(ϕ) && (
            ϕ=Array{F}(undef, length(costfn.grid), Devices.ngrades(costfn.device))
        )

        # CONSTRUCT LEAKAGE OBSERVABLE 1-Π
        O = LAT.basisvectors(Devices.nstates(costfn.device))
        O -= QubitOperations.qubitprojector(fn.device)
        # ROTATE THE FRAME
        T = Integrations.endtime(costfn.grid)
        Devices.evolve!(costfn.frame, costfn.device, costfn.basis, T, O)
            #= NOTE: It feels awfully strange, rotating a projector matrix.
            But it is equivalent to rotating the state at the end of evolution,
                which is hard to do in the gradientsignals method.
            =#

        return (∇f, x) -> (
            Parameters.bind!(fn.device, x);
            Evolutions.gradientsignals(
                fn.evolution,
                fn.device,
                fn.basis,
                fn.grid,
                fn.reference,
                O;
                result=ϕ,   # NOTE: This writes the gradient signal as needed.
            );
            ∇f .= Devices.gradient(fn.device, fn.grid, ϕ)
        )
    end

    function CostFunctions.trajectory_callback(
        costfn::DenseLeakage,
        E::AbstractVector;
        callback=nothing,
    )
        m = Devices.nlevels(costfn.device)
        n = Devices.nqubits(costfn.device)
        workbasis = Evolutions.workbasis(costfn.evolution)  # BASIS OF CALLBACK ψ
        U = Devices.basisrotation(costfn.basis, workbasis, costfn.device)
        π̄ = QubitOperations.localqubitprojectors(m, n)
        ψ_ = similar(costfn.reference)

        return (i, t, ψ) -> (
            ψ_ .= ψ;
            LAT.rotate!(U, ψ_);  # ψ_ IS NOW IN MEASUREMENT BASIS
            Devices.evolve!(costfn.frame, costfn.device, costfn.basis, -t, ψ_);
                                    # ψ_ IS NOW IN MEASUREMENT FRAME
            E[1+i] = 1 - real(LAT.expectation(π̄, ψ_));
            !isnothing(callback) && callback(i, t, ψ)
        )
    end

end