using Test

using FiniteDifferences: grad, central_fdm

using ..AnalyticSquarePulse
using CtrlVQE: Parameters, Signals, Devices, Evolutions
using CtrlVQE.TransmonDevices: TransmonDevice
using CtrlVQE.Bases: Occupation, Dressed

@testset "Devices" begin

    # TODO: Reformulate AnalyticSquarePulse interface, esp. I/O in lab frame.

    #= SINGLE QUBIT TESTS =#

    # ANALYTICAL SOLUTION
    ψ0 = [1,-1]/√2
    T = 3.5^2       # NOTE: Do NOT let T be a multiple of ω-ν!
    ω = 4.50 * 2π
    Ω = 0.020 * 2π
    ν = 4.30 * 2π
    ψT = AnalyticSquarePulse.onequbitsquarepulse(ψ0, T, ν, Ω, ω)

    # ROTATE OUT OF INTERACTION FRAME (TODO: OMIT)
    a = [0 1; 0 0]
    n̂ =    a'*a
    ψT = exp(-im*T*(ω*n̂)) * ψT

    # CONVERT STATEVECTORS TO DRESSED BASIS
    Ω̄ = [Signals.Constant(Ω)]
    device = TransmonDevice([ω], [0], Int[], Devices.Quple[], [1], [ν], Ω̄, 2)

    # VALIDATE `evolve!`: rotate/direct algorithms
    res = convert(Array{ComplexF64}, ψ0)
    res_ = Evolutions.evolve!(Evolutions.Rotate, device, T, res; r=1000)
    @test res === res_
    @test abs(1 - abs(res'*ψT)^2) < 1e-8

    res = convert(Array{ComplexF64}, ψ0)
    res_ = Evolutions.evolve!(Evolutions.Direct, device, T, res; r=1000)
    @test res === res_
    @test abs(1 - abs(res'*ψT)^2) < 1e-8




    #= SINGLE QUTRIT TESTS =#

    # ANALYTICAL SOLUTION
    ψ0 = [1,1,1]/√3
    T = 3.5^2       # NOTE: Do NOT let T be a multiple of ω-ν!
    ω = 4.50 * 2π
    δ = 0.34 * 2π
    Ω = 0.020 * 2π
    ν = 4.30 * 2π
    ψT = AnalyticSquarePulse.onequtritsquarepulse(ψ0, T, ν, Ω, ω, δ)

    # ROTATE OUT OF INTERACTION FRAME (TODO: OMIT)
    a = [0 1 0; 0 0 √2; 0 0 0]
    n̂ =    a'*a
    η̂ = a'*a'*a*a
    ψT = exp(-im*T*(ω*n̂ - δ/2*η̂)) * ψT

    # CONVERT STATEVECTORS TO DRESSED BASIS
    Ω̄ = [Signals.Constant(Ω)]
    device = TransmonDevice([ω], [δ], Int[], Devices.Quple[], [1], [ν], Ω̄, 3)

    # VALIDATE `evolve!`: rotate/direct algorithms
    res = convert(Array{ComplexF64}, ψ0)
    res_ = Evolutions.evolve!(Evolutions.Rotate, device, T, res; r=1000)
    @test res === res_
    @test abs(1 - abs(res'*ψT)^2) < 1e-8

    res = convert(Array{ComplexF64}, ψ0)
    res_ = Evolutions.evolve!(Evolutions.Direct, device, T, res; r=1000)
    @test res === res_
    @test abs(1 - abs(res'*ψT)^2) < 1e-8



    #= TWO-QUBIT TESTS =#

    Ω̄ = [
        Signals.Constant( 0.020),
        Signals.Constant(-0.020),
    ]

    device = TransmonDevice(
        2π * [4.50, 4.52],      # ω̄
        2π * [0.33, 0.34],      # δ̄
        2π * [0.020],           # ḡ
        [Devices.Quple(1, 2)],
        [1, 2],                 # q̄
        2π * [4.30, 4.80],      # ν̄
        Ω̄,
        3,                      # m
    )

    T = 9.6
    ψ0 = [0, 1, 0, 0, 0, 0, 0, 0 ,0]    # |01⟩

    # REFERENCE SOLUTION, BASED ON ALREADY-TESTED CODE
    res = convert(Array{ComplexF64}, ψ0)
    ψ = Evolutions.evolve!(Evolutions.Rotate, device, T, res; r=1000)

    # TEST NON-MUTATING `evolve` IN OCCUPATION BASIS
    @test Evolutions.evolve(Evolutions.Rotate, device, T, ψ0; r=1000) ≈ ψ
    @test Evolutions.evolve(Evolutions.Direct, device, Occupation, T, ψ0; r=1000) ≈ ψ

    # TEST NON-MUTATING `evolve` IN DRESSED BASIS
    U = Devices.basisrotation(Dressed, Occupation, device)
    @test Evolutions.evolve(Evolutions.Rotate, device, Dressed, T, U*ψ0; r=1000) ≈ U*ψ
    @test Evolutions.evolve(Evolutions.Direct, device, T, U*ψ0; r=1000) ≈ U*ψ

    # RUN THE GRADIENT CALCULATION
    O = Matrix([i*j for i in 1:Devices.nstates(device), j in 1:Devices.nstates(device)])
        # SOME SYMMETRIC MATRIX, SUITABLE TO PLAY THE ROLE OF AN OBSERVABLE

    r = 1000
    ϕ̄ = Evolutions.gradientsignals(device, T, ψ0, r, [O])
    τ = T / r
    τ̄ = fill(τ, r + 1)
    τ̄[[begin, end]] ./= 2
    t̄ = τ * (0:r)
    gx = Devices.gradient(device, τ̄, t̄, ϕ̄[:,:,1])

    # TEST AGAINST THE FINITE DIFFERENCE
    function f(x̄)
        Parameters.bind(device, x̄)
        ψ = Evolutions.evolve(device, T, ψ0; r=r)
        return real(ψ'*O*ψ)
    end
    x̄ = 2π*[0.020, -0.020, 4.30, 4.80]      # TODO: Replace with Parameters.values
    g0 = grad(central_fdm(5, 1), f, x̄)[1]
    @test g0 ≈ gx

    display(g0)
    display(gx)
    display(ϕ̄)

    #= TODO:

    We are so close, but this is going to be a pain to debug.

    I guess we need to craft a piece-wise constant pulse with steps every half-Trotter step,
        so that we can compare the finite difference directly to ϕ̄.

    You should write a script for this. >_>

    =#

end
