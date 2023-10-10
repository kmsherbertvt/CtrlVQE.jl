using Test
import .StandardTests

import CtrlVQE

@testset "TrapezoidalIntegration" begin
    grid = CtrlVQE.TrapezoidalIntegration(0.0, 5.0, 100)
    StandardTests.validate(grid)
end