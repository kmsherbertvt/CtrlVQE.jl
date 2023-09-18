using Test
import .StandardTests

import CtrlVQE

@testset "trapezoidaltimegrid" begin
    T = 5.0
    r = 20

    τ, τ̄, t̄ = CtrlVQE.Evolutions.trapezoidaltimegrid(T,r)
    # TRUE FOR ANY TIME GRID
    @test sum(τ̄) == T
    @test 1+r == length(t̄) == length(τ̄)

    # UNIQUE TO TRAPEZOIDAL RULE
    @test first(τ̄) == last(τ̄) == τ/2
end

@testset "Toggle" begin
    evolution = CtrlVQE.Toggle(10000)
    StandardTests.validate(evolution)
end

@testset "Direct" begin
    evolution = CtrlVQE.Direct(10000)
    StandardTests.validate(evolution)
end
