using Test
using LinearAlgebra: diagm, norm
using CtrlVQE: Parameters, Signals, Devices

using CtrlVQE.TransmonDevices: TransmonDevice

using CtrlVQE.Bases: Occupation, Dressed
using CtrlVQE.Operators: Qubit, Coupling, Channel, Gradient
using CtrlVQE.Operators: Uncoupled, Static, Drive, Hamiltonian

@testset "Devices" begin
    # DEFINE THE TEST DEVICE AND OTHER VARIABLES
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

    τ, t = 0.02, 1.0
    ψ = ones(Devices.nstates(device)); ψ ./= norm(ψ)
    φ = copy(ψ); φ[3:7] .*= -1; φ ./= norm(φ)

    # SCALAR TESTS
    @test Devices.nqubits(device) == 2
    @test Devices.nstates(device, 1) == 3
    @test Devices.nstates(device) == 9
    @test Devices.ndrives(device) == 2
    @test Devices.ngrades(device) == 4
    @test Devices.drivequbit(device, 2) == 2
    @test Devices.gradequbit(device, 2) == 1

    # PARAMETER METHODS
    τ̄ = fill(τ, 51)      # NOTE: The correct τ̄ would halve endpoints, but we are...
    ϕ̄ = zeros(length(τ̄), Devices.ngrades(device))   # not testing correctness right now.
    grad = Devices.gradient(device, τ̄, cumsum(τ̄), ϕ̄)
    @test length(grad) == Parameters.count(device) == 4

    Parameters.bind(device, [-0.020, 0.020, 2π * 4.35, 2π * 4.75])
    @test device.Ω̄[1](t) == -0.020
    @test device.Ω̄[2](t) == 0.020
    @test device.ν̄[1] == 2π * 4.35
    @test device.ν̄[2] == 2π * 4.75

    # BASIC MATRICES - not testing for correctness of particular model, just hermiticity
    a = Devices.localloweringoperator(device, 1)        # LOCAL
    a1 = kron(a, one(a))
    a2 = kron(one(a), a)

    h1 = Devices.qubithamiltonian(device, [a, a], 1)    # LOCAL
    @test h1 ≈ h1'
    h2 = Devices.qubithamiltonian(device, [a, a], 2)    # LOCAL
    @test h2 ≈ h2'
    h = kron(h1, one(h1)) + kron(one(h2), h2)

    G = Devices.staticcoupling(device, [a1, a2])
    @test G ≈ G'
    H0 = h + G

    v1 = Devices.driveoperator(device, [a, a], 1, t)    # LOCAL
    @test v1 ≈ v1'
    v2 = Devices.driveoperator(device, [a, a], 2, t)    # LOCAL
    @test v2 ≈ v2'
    V = kron(v1, one(v1)) + kron(one(v2), v2)
    H = H0 + V

    Aα1 = Devices.gradeoperator(device, [a, a], 1, t)   # LOCAL
    @test Aα1 ≈ Aα1'
    Aβ1 = Devices.gradeoperator(device, [a, a], 2, t)   # LOCAL
    @test Aβ1 ≈ Aβ1'
    Aα2 = Devices.gradeoperator(device, [a, a], 3, t)   # LOCAL
    @test Aα2 ≈ Aα2'
    Aβ2 = Devices.gradeoperator(device, [a, a], 4, t)   # LOCAL
    @test Aβ2 ≈ Aβ2'

    # BASIC UTILITIES
    @test Devices.globalize(device, a, 1) ≈ a1
    O = diagm(ones(4))
    ΠOΠ = diagm([1, 1, 0, 1, 1, 0, 0, 0, 0])
    @test Devices.project(device, O, [2,2]) ≈ ΠOΠ
    @test Devices.project(device, O, 2)     ≈ ΠOΠ
    @test Devices.project(device, O)        ≈ ΠOΠ

    # BASIS ROTATIONS - only bother with Occupation and Dressed for now
    Λ, U = Devices.diagonalize(Occupation, device)
    @test U*diagm(Λ)*U' ≈ diagm(ones(Devices.nstates(device)))
    Λ, U = Devices.diagonalize(Occupation, device, 1)
    @test U*diagm(Λ)*U' ≈ diagm(ones(Devices.nstates(device, 1)))
    Λ, U = Devices.diagonalize(Dressed, device)
    @test U*diagm(Λ)*U' ≈ H0

    U1 = Devices.basisrotation(Occupation, Dressed, device)
    U2 = Devices.basisrotation(Dressed, Occupation, device)
    @test U1 ≈ U2'  # NOTE: Not especially thorough...


    # ALGEBRA
    @test collect(Devices.algebra(device)) ≈ [a1, a2]
    @test collect(Devices.localalgebra(device)) ≈ [a, a]

    # OPERATORS
    @test Devices.operator(Qubit, device, 1) ≈ kron(h1, one(h1))
    @test Devices.operator(Coupling, device) ≈ G
    @test Devices.operator(Channel, device, 1, t) ≈ kron(v1, one(v1))
    @test Devices.operator(Gradient, device, 1, t) ≈ kron(Aα1, one(Aα1))
    @test Devices.operator(Gradient, device, 2, t) ≈ kron(Aβ1, one(Aβ1))

    @test Devices.operator(Uncoupled, device) ≈ h
    @test Devices.operator(Static, device) ≈ H0
    @test Devices.operator(Drive, device, t) ≈ V
    @test Devices.operator(Hamiltonian, device, t) ≈ H

    @test collect(Devices.localqubitoperators(device)) ≈ [h1, h2]

    # EVOLVERS - don't bother testing every single operator at this point
    @test Devices.propagator(Uncoupled, device, τ) ≈ exp(-im*τ*h)
    @test Devices.propagator(Static, device, τ) ≈ exp(-im*τ*H0)
    @test Devices.propagator(Drive, device, τ, t) ≈ exp(-im*τ*V)

    @test collect(Devices.localqubitpropagators(device, τ))≈ [exp(-im*τ*h1),exp(-im*τ*h2)]

    @test Devices.evolver(Uncoupled, device, t) ≈ exp(-im*t*h)
    @test Devices.evolver(Static, device, t) ≈ exp(-im*t*H0)
    @test_throws ErrorException Devices.evolver(Drive, device, τ, t)

    @test collect(Devices.localqubitevolvers(device, t)) ≈ [exp(-im*t*h1), exp(-im*t*h2)]

    # MUTATING EVOLUTION
    res = convert(Array{ComplexF64}, ψ)
    res_ = Devices.propagate!(Uncoupled, device, τ, res)
    @test res === res_
    @test res ≈ exp(-im*τ*h) * ψ

    res = convert(Array{ComplexF64}, ψ)
    res_ = Devices.propagate!(Static, device, τ, res)
    @test res === res_
    @test res ≈ exp(-im*τ*H0) * ψ

    res = convert(Array{ComplexF64}, ψ)
    res_ = Devices.propagate!(Drive, device, τ, res, t)
    @test res === res_
    @test res ≈ exp(-im*τ*V) * ψ

    res = convert(Array{ComplexF64}, ψ)
    res_ = Devices.evolve!(Uncoupled, device, t, res)
    @test res === res_
    @test res ≈ exp(-im*t*h) * ψ

    res = convert(Array{ComplexF64}, ψ)
    res_ = Devices.evolve!(Static, device, t, res)
    @test res === res_
    @test res ≈ exp(-im*t*H0) * ψ

    # BRAKET AND EXPECTATION
    @test Devices.braket(Uncoupled, device, ψ, φ) ≈ ψ' * h * φ
    @test Devices.braket(Static, device, ψ, φ) ≈ ψ' * H0 * φ
    @test Devices.braket(Drive, device, ψ, φ, t) ≈ ψ' * V * φ

    @test Devices.expectation(Uncoupled, device, ψ) ≈ ψ' * h * ψ
    @test Devices.expectation(Static, device, ψ) ≈ ψ' * H0 * ψ
    @test Devices.expectation(Drive, device, ψ, t) ≈ ψ' * V * ψ

end
