using Test

loadedanalyticpulses = false; evolve_transmon = nothing
# NOTE: Comment the below lines out to skip evolutions accuracy tests.
# TEMP: Julia package testing framework evidently can't handle url-packages in manifest.
import Pkg; Pkg.add(url="https://github.com/kmsherbertvt/AnalyticPulses.jl")
# TODO (lo): When fixed, replace "Pkg" dependency with "AnalyticPulses" itself.
import AnalyticPulses: OneQubitSquarePulses
loadedanalyticpulses = true; evolve_transmon = OneQubitSquarePulses.evolve_transmon
Pkg.rm("AnalyticPulses")    # Prevent the package manifest from updating.

module StandardTests; include("standards/Tests.jl"); end

@testset "CtrlVQE" begin
    @testset "LinearAlgebraTools" include("LinearAlgebraTools.jl")
    @testset "Integrations" include("Integrations.jl")
    @testset "Signals" include("Signals.jl")
    @testset "Devices" include("Devices.jl")
    @testset "Evolutions" include("Evolutions.jl")
    @testset "CostFunctions" include("CostFunctions.jl")
end