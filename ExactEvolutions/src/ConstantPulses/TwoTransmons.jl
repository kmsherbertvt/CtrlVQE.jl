module TwoTransmons
    import DifferentialEquations

    import LinearAlgebra: norm, kron, mul!


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
        ω1, ω2,         # DEVICE RESONANCE FREQUENCY
        δ1, δ2,         # DEVICE ANHARMONICITY
        g,              # DEVICE COUPLING
        ν1, ν2,         # PULSE FREQUENCY
        Ω1, Ω2,         # PULSE AMPLITUDE
        T,              # PULSE DURATION (ns)
    )
        a1, a2 = twoannihilators(round(Int, sqrt(length(ψ0))))
        H0  = ω1 * (a1'*a1) - δ1/2 * (a1'*a1'*a1*a1)
        H0 += ω2 * (a2'*a2) - δ2/2 * (a2'*a2'*a2*a2)
        H0 +=  g * (a1'*a2 + a2'*a1)
        V1 = similar(H0)
        V2 = similar(H0)
        H = similar(H0)
        p = (a1, a2, H0, V1, V2, H, ν1, ν2, Ω1, Ω2)

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
        a1, a2, H0, V1, V2, H, ν1, ν2, Ω1, Ω2 = p

        # BUILD UP HAMILTONIAN
        V1 .= Ω1 .* exp(im*ν1*t) .* a1
        V1 .+= V1'
        V2 .= Ω2 .* exp(im*ν2*t) .* a2
        V2 .+= V2'
        H .= H0 .+ V1 .+ V2

        # ∂ψ/∂t = -𝑖 H(t) ψ
        H .*= -im
        mul!(du, H, u)
    end

    function twoannihilators(m::Integer=2)
        a = zeros(ComplexF64, (m,m))
        Im = zeros(ComplexF64, (m,m))
        for i ∈ 1:m-1
            a[i,i+1] = √i               # BOSONIC ANNIHILATION OPERATOR
            Im[i,i] = 1                 # IDENTITY
        end
        Im[m,m] = 1

        return kron(a,Im), kron(Im,a)
    end
end