module SingleQubit
    import LinearAlgebra: norm

    """
        evolve_transmon(ψ0, ω, δ, ν, Ω, T)

    Calculate the state of a single transmon after applying a constant drive.

    # Parameters
    - ψ0: initial state of transmon
    - ω: resonance frequency (rad/ns)
    - δ: anharmonicity (rad/ns) [not used; included for consistency with larger systems]
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
        # HELPFUL CONSTANTS
        Δ  = ν - ω              # DETUNING
        ξ  = 2Ω/Δ               # RELATIVE STRENGTH OF PULSE
        η  = √(1 + abs2(ξ))     # SCALING FACTOR

        # IMPOSE BOUNDARY CONSTRAINTS (these are coefficients for solutions to each diff eq)
        A₀ = ψ0[1] * (η-1)/2η - ψ0[2] * ξ /2η
        B₀ = ψ0[1] * (η+1)/2η + ψ0[2] * ξ /2η
        A₁ = ψ0[2] * (η-1)/2η + ψ0[1] * ξ'/2η
        B₁ = ψ0[2] * (η+1)/2η - ψ0[1] * ξ'/2η

        # WRITE OUT GENERAL SOLUTIONS TO THE DIFFERENTIAL EQUATIONS
        ψT = [
            A₀*exp( im*Δ * (η+1)/2 * T) + B₀*exp(-im*Δ * (η-1)/2 * T),
            A₁*exp(-im*Δ * (η+1)/2 * T) + B₁*exp( im*Δ * (η-1)/2 * T),
        ]

        # ROBUSTLY HANDLE lim Δ→0
        if abs(Δ) < eps(typeof(Δ))
            # Δη        → √(Δ² + |2Ω|²)
            # ξ/2η      → Ω / Δη
            # (η±1)/2η  → (1 ± 1/η)/2
            Δη = √(Δ^2 + abs2(2Ω))
            A₀ = ψ0[1] *(1 - 1/η)/2 + ψ0[2] * Ω/Δη
            B₀ = ψ0[1] *(1 + 1/η)/2 - ψ0[2] * Ω/Δη
            A₁ = ψ0[2] *(1 - 1/η)/2 - ψ0[1] * Ω'/Δη
            B₁ = ψ0[2] *(1 + 1/η)/2 + ψ0[1] * Ω'/Δη

            ψT = [
                A₀*exp(-im* (Δη-Δ)/2 * T) + B₀*exp( im* (Δη+Δ)/2 * T),
                A₁*exp( im* (Δη-Δ)/2 * T) + B₁*exp(-im* (Δη+Δ)/2 * T),
            ]
        end

        # ROTATE OUT OF INTERACTION FRAME
        ψT[2] *= exp(-im*T*ω)

        # RE-NORMALIZE THIS STATE
        ψT ./= norm(ψT)

        return ψT
    end
end