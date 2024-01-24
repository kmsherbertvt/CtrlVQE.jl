using Test
import .StandardTests

import CtrlVQE

@testset "Constant" begin
    signal = CtrlVQE.Constant(0.5)
    StandardTests.validate(signal)
end

@testset "AnalyticConstant" begin
    signal = CtrlVQE.AnalyticSignal(CtrlVQE.Constant(0.5), im)
    StandardTests.validate(signal)
end

@testset "CompositeSignal" begin
    signal = CtrlVQE.CompositeSignal(
        CtrlVQE.Constant(0.75),
        CtrlVQE.Constant(0.25),
    )
    StandardTests.validate(signal)
end

@testset "ModulatedSignal" begin
    signal = CtrlVQE.ModulatedSignal(
        CtrlVQE.Constant(0.75),
        CtrlVQE.Constant(0.25),
    )
    StandardTests.validate(signal)
end

@testset "WeightedCompositeSignal" begin
    signal = CtrlVQE.WeightedCompositeSignal(
        CtrlVQE.Constant(0.75),
        CtrlVQE.Constant(0.25),
    )
    signal.weights .= [0.3, -0.7]
    StandardTests.validate(signal)
end

@testset "WindowedSignal" begin
    signal = CtrlVQE.WindowedSignal([
        CtrlVQE.Constant(0.75),
        CtrlVQE.Constant(0.25),
    ], [0.0, 0.5])
    StandardTests.validate(signal)
end

# NOTE: This signal has ill-defined gradients, so test it with ConstrainedSignal.
@testset "Interval" begin
    signal = CtrlVQE.Interval(0.5, 0.25, 1.25)
    StandardTests.validate(CtrlVQE.ConstrainedSignal(signal, :s1, :s2))
end

# NOTE: This signal has ill-defined gradients, so test it with ConstrainedSignal.
@testset "ComplexInterval" begin
    signal = CtrlVQE.ComplexInterval(0.75, 0.25, 0.25, 1.25)
    StandardTests.validate(CtrlVQE.ConstrainedSignal(signal, :s1, :s2))
end

# NOTE: This signal has ill-defined gradients, so test it with ConstrainedSignal.
@testset "StepFunction" begin
    signal = CtrlVQE.StepFunction(0.75, 0.75)
    StandardTests.validate(CtrlVQE.ConstrainedSignal(signal, :s))
end

@testset "ComplexConstant" begin
    signal = CtrlVQE.ComplexConstant(0.75, 0.25)
    StandardTests.validate(signal)
end

@testset "PolarComplexConstant" begin
    signal = CtrlVQE.PolarComplexConstant(0.75, 0.25)
    StandardTests.validate(signal)
end

@testset "Gaussian" begin
    signal = CtrlVQE.Gaussian(0.75, 0.25, 0.75)
    StandardTests.validate(signal)
end

@testset "Sine" begin
    signal = CtrlVQE.Sine(0.75, 0.25, 0.75)
    StandardTests.validate(signal)
end

@testset "Sinc" begin
    signal = CtrlVQE.Sinc(0.75, 0.25, 0.75)
    StandardTests.validate(signal)
end

@testset "Tanh" begin
    signal = CtrlVQE.Tanh(0.75, 0.25, 0.75)
    StandardTests.validate(signal)
end