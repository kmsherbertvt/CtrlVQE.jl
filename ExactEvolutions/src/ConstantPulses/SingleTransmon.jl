module SingleTransmon
    import DifferentialEquations

    import LinearAlgebra: norm, mul!


    """
        evolve_transmon(ψ0, ω, δ, ν, Ω, T)

    Calculate the state of a single transmon after applying a constant drive.

    # Parameters
    - ψ0: initial state of transmon
    - ω: resonance frequency (rad/ns)
    - δ: anharmonicity (rad/ns)
    - ν: drive frequency (rad/ns)
    - Ω: drive amplitude (rad/ns)
    - T: duration of drive (ns)

    """
    function evolve_transmon(
        ψ0,             # INITIAL WAVE FUNCTION
        ω,              # DEVICE RESONANCE FREQUENCY
        δ,              # DEVICE ANHARMONICITY
        ν,              # PULSE FREQUENCY
        Ω,              # PULSE AMPLITUDE
        T,              # PULSE DURATION (ns)
    )
        a = annihilator(length(ψ0))
        H0 = ω * (a'*a) - δ/2 * (a'*a'*a*a)
        V = similar(H0)
        H = similar(H0)
        p = (a, H0, V, H, ν, Ω)

        ψT = deepcopy(ψ0)
        schrodinger = DifferentialEquations.ODEProblem(hamiltonian!, ψT, (0.0, T), p)
        solution = DifferentialEquations.solve(
            schrodinger,
            reltol=1e-6,
            save_everystep=false,
        )
        ψT .= solution.u[end]

        # RE-NORMALIZE THIS STATE
        ψT ./= norm(ψT)

        return ψT
    end

    function hamiltonian!(du, u, p, t)
        a, H0, V, H, ν, Ω = p

        # BUILD UP HAMILTONIAN
        V .= Ω .* exp(im*ν*t) .* a
        V .+= V'
        H .= H0 .+ V

        # ∂ψ/∂t = -𝑖 H(t) ψ
        H .*= -im
        mul!(du, H, u)
    end

    function annihilator(m::Integer=2)
        a = zeros(ComplexF64, (m,m))
        for i ∈ 1:m-1
            a[i,i+1] = √i               # BOSONIC ANNIHILATION OPERATOR
        end
        return a
    end
end