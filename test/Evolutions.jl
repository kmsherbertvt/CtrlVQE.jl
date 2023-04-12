using Test

import FiniteDifferences: grad, central_fdm

# TEMP: Julia package testing framework evidently can't handle url-packages in manifest.
import Pkg; Pkg.add(url="https://github.com/kmsherbertvt/AnalyticPulses.jl")
# TODO (lo): When fixed, replace "Pkg" dependency with "AnalyticPulses"...

import AnalyticPulses: OneQubitSquarePulses

import CtrlVQE: Parameters, Signals, Devices, Evolutions
import CtrlVQE.Bases: OCCUPATION, DRESSED

@testset "Evolutions" begin
    ROTATE = Evolutions.Rotate(1000)
    DIRECT = Evolutions.Direct(1000)

    #= SINGLE QUBIT TESTS =#

    # ANALYTICAL SOLUTION
    ψ0 = [1,-1]/√2
    T = 3.5^2       # NOTE: Do NOT let T be a multiple of ω-ν!
    ω = 4.50 * 2π
    δ = 0.34 * 2π
    Ω = 0.020 * 2π
    ν = 4.30 * 2π
    ψT = OneQubitSquarePulses.evolve_transmon(ω, δ, Ω, ν, T, ψ0)

    # TEST DEVICE
    Ω̄ = [Signals.Constant(Ω)]
    device = Devices.TransmonDevice([ω], [0], Int[], Devices.Quple[], [1], [ν], Ω̄, 2)

    # VALIDATE `evolve!`: rotate/direct algorithms
    res = convert(Array{ComplexF64}, copy(ψ0))
    res_ = Evolutions.evolve!(ROTATE, device, T, res)
    @test res === res_
    @test abs(1 - abs(res'*ψT)^2) < 1e-8

    U = Devices.basisrotation(DRESSED, OCCUPATION, device)
    res = convert(Array{ComplexF64}, copy(U*ψ0))
    res_ = Evolutions.evolve!(DIRECT, device, T, res)
    @test res === res_
    @test abs(1 - abs(res'*(U*ψT))^2) < 1e-8




    #= SINGLE QUTRIT TESTS =#

    # ANALYTICAL SOLUTION
    ψ0 = [1,1,1]/√3
    T = 3.5^2       # NOTE: Do NOT let T be a multiple of ω-ν!
    ω = 4.50 * 2π
    δ = 0.34 * 2π
    Ω = 0.020 * 2π
    ν = 4.30 * 2π
    ψT = OneQubitSquarePulses.evolve_transmon(ω, δ, Ω, ν, T, ψ0)

    # CONVERT STATEVECTORS TO DRESSED BASIS
    Ω̄ = [Signals.Constant(Ω)]
    device = Devices.TransmonDevice([ω], [δ], Int[], Devices.Quple[], [1], [ν], Ω̄, 3)

    # VALIDATE `evolve!`: rotate/direct algorithms
    res = convert(Array{ComplexF64}, copy(ψ0))
    res_ = Evolutions.evolve!(ROTATE, device, T, res)
    @test res === res_
    @test abs(1 - abs(res'*ψT)^2) < 1e-8

    U = Devices.basisrotation(DRESSED, OCCUPATION, device)
    res = convert(Array{ComplexF64}, copy(U*ψ0))
    res_ = Evolutions.evolve!(DIRECT, device, T, res)
    @test res === res_
    @test abs(1 - abs(res'*(U*ψT))^2) < 1e-8



    #= TWO-QUBIT TESTS =#

    Ω̄ = [
        Signals.Constant( 0.020*2π),
        Signals.Constant(-0.020*2π),
    ]

    device = Devices.TransmonDevice(
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
    res = convert(Array{ComplexF64}, copy(ψ0))
    ψ = Evolutions.evolve!(ROTATE, device, T, res)

    # TEST NON-MUTATING `evolve` IN OCCUPATION BASIS
    @test Evolutions.evolve(ROTATE, device, T, ψ0) ≈ ψ
    @test Evolutions.evolve(DIRECT, device, OCCUPATION, T, ψ0) ≈ ψ

    # TEST NON-MUTATING `evolve` IN DRESSED BASIS
    U = Devices.basisrotation(DRESSED, OCCUPATION, device)
    @test Evolutions.evolve(ROTATE, device, DRESSED, T, U*ψ0) ≈ U*ψ
    @test Evolutions.evolve(DIRECT, device, T, U*ψ0) ≈ U*ψ

    # RUN THE GRADIENT CALCULATION
    O = Matrix([i*j for i in 1:Devices.nstates(device), j in 1:Devices.nstates(device)])
        # SOME SYMMETRIC MATRIX, SUITABLE TO PLAY THE ROLE OF AN OBSERVABLE

    x̄ = Parameters.values(device)

    r = 1000
    ϕ̄ = Evolutions.gradientsignals(device, T, ψ0, r, [O])
    τ, τ̄, t̄ = Evolutions.trapezoidaltimegrid(T, r)
    g0 = Devices.gradient(device, τ̄, t̄, ϕ̄[:,:,1])

    # TEST AGAINST THE FINITE DIFFERENCE
    function f(x̄)
        Parameters.bind(device, x̄)
        ψ = Evolutions.evolve(ROTATE, device, T, ψ0)
        return real(ψ'*O*ψ)
    end
    gΔ = grad(central_fdm(5, 1), f, x̄)[1]

    @test √sum((g0 .- gΔ).^2) < 1e-3
end
