using Test
import .StandardTests

import CtrlVQE

import CtrlVQE: ConstantSignals
import CtrlVQE: TransmonDevices

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
    TransmonDevices.resonancefrequency(device, q) + Δ[q] for q in eachindex(Δ)
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
evolution = CtrlVQE.Toggle(r)

# TEST ENERGY FUNCTIONS!
bases = [OCCUPATION, DRESSED]
import CtrlVQE: BareEnergy, ProjectedEnergy, NormalizedEnergy, Normalization
import CtrlVQE.Operators: IDENTITY, UNCOUPLED, STATIC
frames = [IDENTITY, UNCOUPLED, STATIC]

import CtrlVQE.Bases: Occupation, Dressed           # IMPORT TYPES TO FACILITATE LABELING
import CtrlVQE.Operators: Identity, Uncoupled, Static

@testset "BareEnergy" begin
    for frame in frames
    @testset "$(typeof(frame))" begin
        values = Vector{Float64}(undef, 2)

        @testset "Occupation" begin
            fn = BareEnergy(evolution, device, OCCUPATION, frame, T, ψ0, O0)
            StandardTests.validate(fn)
            values[1] = fn(xi)
        end

        @testset "Dressed" begin
            fn = BareEnergy(evolution, device, DRESSED, frame, T, ψd, Od)
            StandardTests.validate(fn)
            values[2] = fn(xi)
        end

        # FOR BARE ENERGY ONLY, CHECK CONSISTENCY BETWEEN DIFFERENT BASES
        @test values[1] ≈ values[2]
    end; end
end

for fn_type in [ProjectedEnergy, NormalizedEnergy]
@testset "$fn_type" begin
    for frame in frames
    @testset "$(typeof(frame))" begin
        for basis in bases
        @testset "$(typeof(basis))" begin
            fn = fn_type(evolution, device, basis, frame, T, ψ0, O0)
            StandardTests.validate(fn)
        end; end
    end; end
end; end

@testset "Normalization" begin
    for basis in bases
    @testset "$(typeof(basis))" begin
        fn = Normalization(evolution, device, basis, T, ψ0)
        StandardTests.validate(fn)
    end; end
end
