module SingleQutrit
    import Polynomials: Polynomial, roots
    import SpecialMatrices: Vandermonde

    import LinearAlgebra: norm


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
        #######################
        #= HELPFUL CONSTANTS =#
        Δ  = ν - ω      # DETUNING
        # DERIVED FROM INTERACTION HAMILTONIAN ELEMENT ⟨0|V|1⟩
        A₁ = Ω          # t=0 AMPLITUDE
        A₁²= abs2(A₁)   # SQUARE AMPLITUDE
        D₁ = im * Δ     # DERIVATIVE MULITPLE
        # DERIVED FROM INTERACTION HAMILTONIAN ELEMENT ⟨1|V|2⟩
        A₂ = Ω * √2     # t=0 AMPLITUDE
        A₂²= abs2(A₂)   # SQUARE AMPLITUDE
        D₂ = im *(Δ+δ)  # DERIVATIVE MULITPLE
        #######################

        ψT = [          # FINAL WAVEFUNCTION
            # |0⟩ COEFFICIENT
            _solve_diffeq([                 # CONSTANT COEFFICIENTS IN LINEAR DIFF EQ
                -A₁² * (D₁ + D₂),                               # c COEFFICIENT
                A₁² + A₂² + D₁*(D₁ + D₂),                       # ċ COEFFICIENT
                -(2D₁ + D₂),                                    # c̈ COEFFICIENT
            ],[                             # BOUNDARY CONDITIONS AT START OF THE PULSE
                ψ0[1],                                          # c(t=0)
                ψ0[2] * -im*A₁,                                 # ċ(t=0)
                -A₁*(A₁'*ψ0[1] + im*D₁*ψ0[2] + A₂*ψ0[3])        # c̈(t=0)
            ])(T),                          # EVALUATE SOLUTION AT END OF THE PULSE

            # |1⟩ COEFFICIENT
            _solve_diffeq([                 # CONSTANT COEFFICIENTS IN LINEAR DIFF EQ
                D₁*A₂² - D₂*A₁²,                                # c COEFFICIENT
                A₁² + A₂² - D₁*D₂,                              # ċ COEFFICIENT
                D₁ - D₂,                                        # c̈ COEFFICIENT
            ],[                             # BOUNDARY CONDITIONS AT START OF THE PULSE
                ψ0[2],                                              # c(t=0)
                -im*(A₁'*ψ0[1] + A₂*ψ0[3]),                         # ċ(t=0)
                im*D₁*A₁'*ψ0[1] - (A₁²+A₂²)*ψ0[2] - im*D₂*A₂*ψ0[3]  # c̈(t=0)
            ])(T),                          # EVALUATE SOLUTION AT END OF THE PULSE

            # |2⟩ COEFFICIENT
            _solve_diffeq([                 # CONSTANT COEFFICIENTS IN LINEAR DIFF EQ
                A₂² * (D₁ + D₂),                                # c COEFFICIENT
                A₁² + A₂² + D₂*(D₁ + D₂),                       # ċ COEFFICIENT
                D₁ + 2D₂,                                       # c̈ COEFFICIENT
            ],[                             # BOUNDARY CONDITIONS AT START OF THE PULSE
                ψ0[3],                                          # c(t=0)
                ψ0[2] * -im*A₂',                                # ċ(t=0)
                -A₂'*(A₁'*ψ0[1] - im*D₂*ψ0[2] + A₂*ψ0[3])       # c̈(t=0)
            ])(T),                          # EVALUATE SOLUTION AT END OF THE PULSE
        ]

        # ROTATE OUT OF INTERACTION FRAME
        ψT[2] *= exp(-im*T*ω)
        ψT[3] *= exp(-im*T*(2ω-δ))

        # RE-NORMALIZE THIS STATE
        ψT ./= norm(ψT)

        return ψT
    end

    function _solve_diffeq(a, b)
        # SOLVE THE AUXILIARY POLYNOMIAL EQUATION
        r = roots(Polynomial([a..., 1]))        # SOLUTIONS HAVE FORM exp(r⋅t)
        # SOLVE FOR RELATIVE WEIGHT OF EACH SOLUTION VIA BOUNDARY CONDITIONS
        C = transpose(Vandermonde(r)) \ b       # x = A \ b SOLVES MATRIX-VECTOR EQUATION Ax=b
        # RETURN A FUNCTION GIVING THE LINEAR COMBINATION OF ALL SOLUTIONS
        return t -> transpose(C) * exp.(r*t)    # THIS IS AN INNER PRODUCT OF TWO VECTORS!
    end
end