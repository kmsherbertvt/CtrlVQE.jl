using CtrlVQE
using Test
using Documenter

@testset "CtrlVQE.jl" begin
    #= Run doctests, which validate each basic type implementation. =#
    DocMeta.setdocmeta!(CtrlVQE, :DocTestSetup, :(using CtrlVQE); recursive=true)
    doctest(CtrlVQE)

    #= Check that each basic device evolves a small state correctly. =#
    @testset "Evolutions" begin
        include("EvolutionTests.jl")
    end

    #= Check that a basic cost function accurately reproduces chemical energies. =#
    @testset "Energies" begin
        include("EnergyTests.jl")
    end

    #= Check that tests for modular framework all run. =#
    @testset "ModularFramework" begin
        #= Check that each basic device evolves a small state correctly. =#
        @testset "Evolutions" begin
            include("modulars/EvolutionTests.jl")
        end

        #= Check that a basic cost function accurately reproduces chemical energies. =#
        @testset "Energies" begin
            include("modulars/EnergyTests.jl")
        end
    end
end
