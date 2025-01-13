using CtrlVQE
using Test

@testset "CtrlVQE.jl" begin
    #= Check that each basic device evolves a small state correctly. =#
    @testset "Evolutions" begin
        include("EvolutionTests.jl")
    end

    #= Check that a basic cost function accurately reproduces chemical energies. =#
    @testset "Energies" begin
        include("EnergyTests.jl")
    end
end
