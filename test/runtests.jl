using Test

module StandardTests; include("standards/Tests.jl"); end

@testset "CtrlVQE" begin
    @testset "LinearAlgebraTools" include("LinearAlgebraTools.jl")
    @testset "Signals" include("Signals.jl")
    @testset "Devices" include("Devices.jl")
    @testset "Evolutions" include("Evolutions.jl")
    @testset "CostFunctions" include("CostFunctions.jl")
end