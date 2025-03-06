module EvolutionTests
    import CtrlVQE.ModularFramework: LocalDevice
    import CtrlVQE.ModularFramework: TruncatedBosonicAlgebra
    import CtrlVQE.ModularFramework: TransmonDrift, DipoleDrive
    import CtrlVQE.ModularFramework: DISJOINT

    import CtrlVQE
    import CtrlVQE.Quples: Quple
    import CtrlVQE.Devices: DeviceType
    import CtrlVQE: Constant

    import ExactEvolutions: ConstantPulses

    import Random;
    import LinearAlgebra: norm

    const ω = [4.82, 4.84]
    const δ = [0.30, 0.30]
    const g = [0.02]
    const T = 10.0

    """
        testinfidelity(::Type{D}, m::Int, Ω, ν) where {D <: DeviceType}

    Compute the infidelity between states evolved "analytically" and "experimentally".

    The values Ω and ν specify the drive parameters,
        since different device types can handle different sorts.
    Pulse duration and static device parameters are hard-coded for now.

    "Analytical" is rather loosely defined.
    When n > 2 or m > 3,
        it's a generic black-box ODE solver with very strict relative error.

    Currently this test is only defined for transmons,
        but the interface can in principle be extended if needed.

    """
    function testinfidelity(devicetype::Type{<:DeviceType}, m::Int, Ω, Δ)
        n = length(Ω)

        # CONSTRUCT ARBITRARY REFERENCE STATE
        Random.seed!(0)
        ψ0 = rand(Complex{real(eltype(Ω))}, m^n)
        ψ0 ./= norm(ψ0)

        # CONSTRUCT "ANALYTICAL" SOLUTION
        if n == 1
            ψ = (
                m == 2 ? ConstantPulses.SingleQubit :
                m == 3 ? ConstantPulses.SingleQutrit :
                         ConstantPulses.SingleTransmon
            ).evolve_transmon(ψ0, ω[1], δ[1], ω[1]+Δ[1], only(Ω), T)
        elseif n == 2
            ψ = ConstantPulses.TwoTransmons.evolve_transmon(ψ0,
                ω[1], ω[2], δ[1], δ[2], g[1],
                ω[1]+Δ[1], ω[2]+Δ[2], Ω[1], Ω[2], T,
            )
        else
            throw(ArgumentError("Qubit count $n > 2 not supported."))
        end

        # CONSTRUCT "EXPERIMENTAL" SOLUTION
        device = TestDevice(devicetype, m, Ω, Δ)
        grid = CtrlVQE.TemporalLattice(T, round(Int, 20T))
        evolution = CtrlVQE.QUBIT_FRAME
        ψ_ = CtrlVQE.evolve(evolution, device, grid, ψ0)

        return 1 - abs2(ψ' * ψ_)
    end

    """
        TestDevice(::Type{D}, m::Int, Ω, Δ) where {D <: DeviceType}

    Construct a device compatible with an `evolutiontest`.

    The device should be a transmon device with `m` levels per transmon,
        with each transmon subject to a single, constant pulse defined by `Ω`.
    Pulse duration and transmon parameters are defined by consts in `EvolutionTests`.

    If you define a new `DeviceType` compatible with these requirements,
        you can implement this method and call.
    Otherwise, it will assume the `devicetype` has a constructor
        similar to `ComplexWindowedResonantTransmonDevice`.

    """
    function TestDevice(devicetype::Type{<:LocalDevice}, m::Int, Ω, Δ)
        n = length(Ω)
        A = TruncatedBosonicAlgebra{m,n}
        quples = Quple[Quple(q,q+1) for q in 1:n-1]
        drift = TransmonDrift{A}(ω[1:n], δ[1:n], g[1:n-1], quples)
        drives = [DipoleDrive{A}(q, ω[q], Constant(Ω[q]), Constant(Δ[q])) for q in 1:n]
        return LocalDevice(eltype(Δ), A(), drift, drives, DISJOINT)
    end

end
# import .EvolutionTests: testinfidelity
testinfidelity = EvolutionTests.testinfidelity
import CtrlVQE.ModularFramework: LocalDevice

Ω = [0.02-0.01im, -0.01+0.02im]
Δ = [0.1, -0.2]

@testset "LocalDevice" begin
    devicetype = LocalDevice
    @test testinfidelity(devicetype, 2, Ω[1:1], Δ[1:1]) < 1e-5          # SINGLE QUBIT
    @test testinfidelity(devicetype, 3, Ω[1:1], Δ[1:1]) < 1e-5          # SINGLE QUTRIT
    @test testinfidelity(devicetype, 2, Ω[1:2], Δ[1:2]) < 1e-5          # TWO QUBITS
    @test testinfidelity(devicetype, 3, Ω[1:2], Δ[1:2]) < 1e-5          # TWO QUTRITS
end
