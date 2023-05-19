using Test
import .StandardTests

import CtrlVQE: Signals

@testset "Constant" begin
    signal = Signals.Constant(0.5)
    StandardTests.validate(signal)
end

@testset "CompositeSignal" begin
    signal = Signals.CompositeSignal(
        Signals.Constant(0.75),
        Signals.Constant(0.25),
    )
    StandardTests.validate(signal)
end

@testset "ModulatedSignal" begin
    signal = Signals.ModulatedSignal(
        Signals.Constant(0.75),
        Signals.Constant(0.25),
    )
    StandardTests.validate(signal)
end

@testset "WindowedSignal" begin
    signal = Signals.WindowedSignal([
        Signals.Constant(0.75),
        Signals.Constant(0.25),
    ], [0.0, 0.5])
    StandardTests.validate(signal)
end

# NOTE: This signal has ill-defined gradients, so test it with ConstrainedSignal.
@testset "Interval" begin
    signal = Signals.Interval(0.5, 0.25, 1.25)
    StandardTests.validate(Signals.ConstrainedSignal(signal, :s1, :s2))
end

# NOTE: This signal has ill-defined gradients, so test it with ConstrainedSignal.
@testset "ComplexInterval" begin
    signal = Signals.ComplexInterval(0.75, 0.25, 0.25, 1.25)
    StandardTests.validate(Signals.ConstrainedSignal(signal, :s1, :s2))
end

# NOTE: This signal has ill-defined gradients, so test it with ConstrainedSignal.
@testset "StepFunction" begin
    signal = Signals.StepFunction(0.75, 0.75)
    StandardTests.validate(Signals.ConstrainedSignal(signal, :s))
end

@testset "ComplexConstant" begin
    signal = Signals.ComplexConstant(0.75, 0.25)
    StandardTests.validate(signal)
end

@testset "Gaussian" begin
    signal = Signals.Gaussian(0.75, 0.25, 0.75)
    StandardTests.validate(signal)
end