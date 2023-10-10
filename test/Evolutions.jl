using Test
import .StandardTests

import CtrlVQE

@testset "Toggle" begin
    StandardTests.validate(CtrlVQE.TOGGLE)
end

@testset "Direct" begin
    StandardTests.validate(CtrlVQE.DIRECT)
end
