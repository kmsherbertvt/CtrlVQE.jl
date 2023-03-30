using Test
using CtrlVQE: Parameters, Signals

@testset "Signals" begin
    ∂ = Signals.partial

    # TEST CONSTANT SIGNAL
    c = Signals.Constant(5.0)
    @test c(0.0) == 5.0
    @test c(1) == 5.0

    @test ∂(1,c,0.0) == 1.0
    @test ∂(1,c,1) == 1.0

    @test Parameters.count(c) == 1
    Parameters.bind(c, [2.0])
    @test c.A == 2.0

    # TEST STEP SIGNAL
    Θ = Signals.StepFunction(3.0, 0.5)
    @test Θ(0.0) == 0.0
    @test Θ(1) == 3.0

    @test ∂(1,Θ,0.0) == 0.0
    @test ∂(1,Θ,1) == 1.0

    @test ∂(2,Θ,0.0) == 0.0
    @test ∂(2,Θ,1) == 0.0

    @test Parameters.count(Θ) == 2
    Parameters.bind(Θ, [2.0, 0.7])
    @test Θ.A == 2.0
    @test Θ.s == 0.7

    # TEST CONSTRAINED SIGNAL
    θ = Signals.Constrained(Θ, :s)
    @test θ(0.0) == 0.0
    @test θ(1) == 2.0

    @test ∂(1,θ,0.0) == 0.0
    @test ∂(1,θ,1) == 1.0

    @test Parameters.count(θ) == 1
    Parameters.bind(θ, [3.0])
    @test θ.constrained.A == 3.0
    @test θ.constrained.s == 0.7

    # TEST COMPOSITE SIGNAL
    f = Signals.Composite((c, θ))

    @test f(0.0) == 2.0
    @test f(1) == 5.0

    @test ∂(1,f,0.0) == 1.0
    @test ∂(1,f,1) == 1.0
    @test ∂(2,f,0.0) == 0.0
    @test ∂(2,f,1) == 1.0

    @test Parameters.count(f) == 2
    Parameters.bind(f, [1.0, 6.0])
    @test f.components[1].A == 1.0
    @test f.components[2].constrained.A == 6.0
    @test f.components[2].constrained.s == 0.7

    # TEST MODULATED SIGNAL
    F = Signals.Modulated((c, θ))

    @test F(0.0) == 0.0
    @test F(1) == 6.0

    @test ∂(1,F,0.0) == 0.0
    @test ∂(1,F,1) == 6.0
    @test ∂(2,F,0.0) == 0.0
    @test ∂(2,F,1) == 1.0

    @test Parameters.count(F) == 2
    Parameters.bind(F, [3.0, 2.0])
    @test F.components[1].A == 3.0
    @test F.components[2].constrained.A == 2.0
    @test F.components[2].constrained.s == 0.7



    # TODO: Cosine, Gaussian

end
