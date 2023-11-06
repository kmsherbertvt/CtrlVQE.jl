using Test
import .StandardTests

import CtrlVQE

import CtrlVQE: ConstantSignals
import CtrlVQE: Devices, TransmonDevices

import CtrlVQE.Bases: OCCUPATION, DRESSED

using Random: seed!
using LinearAlgebra: Hermitian

##########################################################################################
# ENERGY FUNCTION SELF-CONSISTENCY CHECKS

# TRANSMON DEVICE PARAMETERS
Δ = 2π * [0.3, -0.1]
pulses = [
    ConstantSignals.ComplexConstant( 2π * 0.02, -2π * 0.02),
    ConstantSignals.ComplexConstant(-2π * 0.02,  2π * 0.02),
]
device = CtrlVQE.Systematic(TransmonDevices.TransmonDevice, 2, pulses; m=3)
TransmonDevices.bindfrequencies(device, [
    Devices.resonancefrequency(device, q) + Δ[q] for q in eachindex(Δ)
])
xi = CtrlVQE.Parameters.values(device)

# OBSERVABLE AND REFERENCE STATE, OCCUPATION BASIS
N = CtrlVQE.nstates(device)

seed!(0)
O0 = Hermitian(rand(ComplexF64, N,N))
ψ0 = zeros(ComplexF64, N); ψ0[1] = 1

# OBSERVABLE AND REFERENCE STATE IN DRESSED BASIS, FOR CONSISTENCY CHECK
U = CtrlVQE.Devices.basisrotation(DRESSED, OCCUPATION, device)
Od = CtrlVQE.LinearAlgebraTools.rotate!(U, convert(Matrix, O0))
ψd = CtrlVQE.LinearAlgebraTools.rotate!(U, ψ0)

# ALGORITHM AND BASIS
T = 5.0
r = 1000
grid = CtrlVQE.TrapezoidalIntegration(0.0, T, r)
evolution = CtrlVQE.TOGGLE

# TEST ENERGY FUNCTIONS!
bases = [OCCUPATION, DRESSED]
import CtrlVQE: BareEnergy, ProjectedEnergy, NormalizedEnergy, Normalization
import CtrlVQE.Operators: IDENTITY, UNCOUPLED, STATIC
frames = [IDENTITY, UNCOUPLED, STATIC]

import CtrlVQE.Bases: Occupation, Dressed           # IMPORT TYPES TO FACILITATE LABELING
import CtrlVQE.Operators: Identity, Uncoupled, Static

# @testset "BareEnergy" begin
#     for frame in frames
#     @testset "$(typeof(frame))" begin
#         values = Vector{Float64}(undef, 2)

#         @testset "Occupation" begin
#             fn = BareEnergy(evolution, device, OCCUPATION, frame, grid, ψ0, O0)
#             StandardTests.validate(fn)
#             values[1] = fn(xi)
#         end

#         @testset "Dressed" begin
#             fn = BareEnergy(evolution, device, DRESSED, frame, grid, ψd, Od)
#             StandardTests.validate(fn)
#             values[2] = fn(xi)
#         end

#         # FOR BARE ENERGY ONLY, CHECK CONSISTENCY BETWEEN DIFFERENT BASES
#         @test values[1] ≈ values[2]
#     end; end
# end

# for fn_type in [ProjectedEnergy, NormalizedEnergy]
# @testset "$fn_type" begin
#     for frame in frames
#     @testset "$(typeof(frame))" begin
#         for basis in bases
#         @testset "$(typeof(basis))" begin
#             fn = fn_type(evolution, device, basis, frame, grid, ψ0, O0)
#             StandardTests.validate(fn)
#         end; end
#     end; end
# end; end

# @testset "Normalization" begin
#     for basis in bases
#     @testset "$(typeof(basis))" begin
#         fn = Normalization(evolution, device, basis, grid, ψ0)
#         StandardTests.validate(fn)
#     end; end
# end

# TEST PENALTY FUNCTIONS

@testset "GlobalAmplitudeBounds" begin
    ΩMAX = 0.5      # MOCK UNITS SO THAT RANDOM PARAMETERS [0,1] ARE REASONABLE
    fn = CtrlVQE.GlobalAmplitudeBound(device, grid, ΩMAX, 1.0, ΩMAX)
    StandardTests.validate(fn)
end

@testset "GlobalFrequencyBounds" begin
    ΔMAX = 0.5      # MOCK UNITS SO THAT RANDOM PARAMETERS [0,1] ARE REASONABLE
    νdevice = deepcopy(device)  # We need to manipulate the resonance frequencies...
    νdevice.ω̄ .= [1.0, 1.1]     #   so drive frequencies in range [0,1] are reasonable.
    fn = CtrlVQE.GlobalFrequencyBound(νdevice, grid, ΔMAX, 1.0, ΔMAX)
    StandardTests.validate(fn)
end

@testset "AmplitudeBounds" begin
    ΩMAX = 0.5      # MOCK UNITS SO THAT RANDOM PARAMETERS [0,1] ARE REASONABLE
    L = 6
    Ω = 1:4

    @testset "Paired" begin
        fn = CtrlVQE.AmplitudeBound(ΩMAX, 1.0, ΩMAX, L, Ω, true)
        StandardTests.validate(fn)
    end

    @testset "Unpaired" begin
        fn = CtrlVQE.AmplitudeBound(ΩMAX, 1.0, ΩMAX, L, Ω, false)
        StandardTests.validate(fn)
    end
end

@testset "SmoothBounds" begin
    ΩMAX = 0.0    # MOCK UNITS SO THAT RANDOM PARAMETERS [0,1] ARE REASONABLE
    ω = 0.5       # MOCK UNITS SO THAT RANDOM PARAMETERS [0,1] ARE REASONABLE

    λ̄ = [1.0, 1.0, 0.0]
    σ̄ = [1.0, 0.1, 0.0]
    μ̄R = [ΩMAX, ω+2π, 0.0]
    μ̄L = [-ΩMAX, ω-2π, 0.0]

    fn = CtrlVQE.SmoothBound(λ̄, μ̄R, μ̄L, σ̄)
    StandardTests.validate(fn)
end

@testset "HardBounds" begin
    ΩMAX = 0.0    # MOCK UNITS SO THAT RANDOM PARAMETERS [0,1] ARE REASONABLE
    ω = 0.5       # MOCK UNITS SO THAT RANDOM PARAMETERS [0,1] ARE REASONABLE

    λ̄ = [1.0, 1.0, 0.0]
    σ̄ = [1.0, 0.1, 0.0]
    μ̄R = [ΩMAX, ω+2π, 0.0]
    μ̄L = [-ΩMAX, ω-2π, 0.0]

    fn = CtrlVQE.HardBound(λ̄, μ̄L, μ̄R, σ̄)
    StandardTests.validate(fn)
end

@testset "SoftBounds" begin
    ΩMAX = 0.5    # MOCK UNITS SO THAT RANDOM PARAMETERS [0,1] ARE REASONABLE
    ω = 0.5       # MOCK UNITS SO THAT RANDOM PARAMETERS [0,1] ARE REASONABLE

    λ̄ = [1.0, 1.0, 0.0]
    μ̄ = [0.0, ω, 0.0]
    σ̄ = [ΩMAX, 2π, 0.0]

    fn = CtrlVQE.SoftBound(λ̄, μ̄, σ̄)
    StandardTests.validate(fn)
end

# TEST COMPOSITE FUNCTIONS

@testset "CompositeCostFunction" begin
    ΩMAX = 0.5      # MOCK UNITS SO THAT RANDOM PARAMETERS [0,1] ARE REASONABLE
    energyfn = BareEnergy(evolution, device, OCCUPATION, STATIC, grid, ψ0, O0)
    penaltyfn = CtrlVQE.GlobalAmplitudeBound(device, grid, ΩMAX, 1.0, ΩMAX)
    fn = CtrlVQE.CompositeCostFunction(energyfn, penaltyfn)
    StandardTests.validate(fn)
end

@testset "ConstrainedEnergyFunction" begin
    ΩMAX = 0.5      # MOCK UNITS SO THAT RANDOM PARAMETERS [0,1] ARE REASONABLE
    energyfn = BareEnergy(evolution, device, OCCUPATION, STATIC, grid, ψ0, O0)
    penaltyfn = CtrlVQE.GlobalAmplitudeBound(device, grid, ΩMAX, 1.0, ΩMAX)
    fn = CtrlVQE.ConstrainedEnergyFunction(energyfn, penaltyfn)
    StandardTests.validate(fn)
end