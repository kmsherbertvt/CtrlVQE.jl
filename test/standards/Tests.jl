#= Systematic and comprehensive accuracy/consistency unit tests. =#
using Test

#= TODO (lo): When Julia fixes url packages, use standard import,
                and don't make accuracy optional. =#
# import AnalyticPulses.OneQubitSquarePulses: evolve_transmon
import ..loadedanalyticpulses, ..evolve_transmon

import CtrlVQE: Parameters, LinearAlgebraTools
import CtrlVQE: Integrations, Signals, Devices, Evolutions, CostFunctions

import CtrlVQE: TrapezoidalIntegration
import CtrlVQE: ConstantSignals
import CtrlVQE: TransmonDevices

import CtrlVQE.Bases: OCCUPATION, DRESSED
import CtrlVQE.Operators: StaticOperator, IDENTITY, COUPLING, STATIC
import CtrlVQE.Operators: Qubit, Channel, Drive, Hamiltonian, Gradient

import CtrlVQE

using Random: seed!
using LinearAlgebra: I, norm, Hermitian, eigen, Diagonal, diag
using FiniteDifferences: grad, central_fdm

const t = 1.0
const T = 5.0
const r = 10
const τ = T / r
const t̄ = range(0.0, T, r+1)
const ϕ̄ = ones(r+1)

const grid = TrapezoidalIntegration(0.0, T, r)
const Δ = 2π * 0.3 # GHz  # DETUNING FOR TESTING

function validate(device::Devices.DeviceType{F,FΩ}) where {F,FΩ}
    m = Devices.nlevels(device)
    n = Devices.nqubits(device)
    N = Devices.nstates(device)
    nD= Devices.ndrives(device)
    nG= Devices.ngrades(device)

    # PARAMETERS AND GRADIENT

    L = Parameters.count(device)
    x̄ = Parameters.values(device)
    @test length(x̄) == length(Parameters.names(device)) == L

    Parameters.bind(device, 2 .* x̄)
    @test Parameters.values(device) ≈ 2 .* x̄
    Parameters.bind(device, x̄)

    ϕ̄_ = repeat(ϕ̄, 1, nG)
    grad = Devices.gradient(device, grid, ϕ̄_)
    @test length(grad) == L
    grad_ = zero(grad); Devices.gradient(device, grid, ϕ̄_; result=grad_)
    @test grad ≈ grad_

    # NOTE: Accuracy of gradient is deferred to unit-tests on energy functions.

    # HAMILTONIAN OPERATORS

    ā = Devices.algebra(device)
    @test Devices.eltype_algebra(device) == eltype(ā)
    @test size(ā) == (N, N, n)

    a0 = Devices.localloweringoperator(device)
    @test Devices.eltype_localloweringoperator(device) == eltype(a0)
    a0_ = zero(a0); Devices.localloweringoperator(device; result=a0_)
    @test a0 ≈ a0_

    G = Devices.staticcoupling(device, ā)
    @test Devices.eltype_staticcoupling(device) == eltype(G)
    G_ = zero(G); Devices.staticcoupling(device, ā; result=G_)
    @test G ≈ G_
    @test G ≈ G'          # NOTE: Sanity check only. We are not testing for correctness!

    h̄ = [Devices.qubithamiltonian(device, ā, q) for q in 1:n]
    for (q, h) in enumerate(h̄)
        @test Devices.eltype_qubithamiltonian(device) == eltype(h)
        h_ = zero(h); Devices.qubithamiltonian(device, ā, q; result=h_)
        @test h ≈ h_
        @test h ≈ h'      # NOTE: Sanity check only. We are not testing for correctness!
    end

    v̄ = [Devices.driveoperator(device, ā, i, t) for i in 1:nD]
    for (i, v) in enumerate(v̄)
        @test Devices.eltype_driveoperator(device) == eltype(v)
        v_ = zero(v); Devices.driveoperator(device, ā, i, t; result=v_)
        @test v ≈ v_
        @test v ≈ v'      # NOTE: Sanity check only. We are not testing for correctness!
    end

    Ā = [Devices.gradeoperator(device, ā, j, t) for j in 1:nG]
    for (j, A) in enumerate(Ā)
        @test Devices.eltype_gradeoperator(device) == eltype(A)
        A_ = zero(A); Devices.gradeoperator(device, ā, j, t; result=A_)
        @test A ≈ A_
        @test A ≈ A'      # NOTE: Sanity check only. We are not testing for correctness!
    end

    # BASIS CONTROL

    āL = Devices.localalgebra(device)
    @test Devices.eltype_algebra(device) == eltype(ā)
    @test size(āL) == (m, m, n)

    for q in 1:n
        aL = āL[:,:,q]
        @test aL ≈ a0
        aG = Devices.globalize(device, aL, q)
        @test aG ≈ ā[:,:,q]
        aG_ = zero(aG); Devices.globalize(device, aL, q; result=aG_)
        @test aG ≈ aG_
    end

    # OPERATOR METHOD ACCURACIES

    @test Devices.operator(IDENTITY, device) ≈ Matrix(I, N, N)
    @test Devices.operator(COUPLING, device) ≈ G
    H0 = sum(h̄) + G; @test Devices.operator(STATIC, device) ≈ H0
    for (q, h) in enumerate(h̄); @test Devices.operator(Qubit(q), device) ≈ h; end
    for (i, v) in enumerate(v̄); @test Devices.operator(Channel(i,t), device) ≈ v; end
    for (j, A) in enumerate(Ā); @test Devices.operator(Gradient(j,t), device) ≈ A; end
    @test Devices.operator(Drive(t), device) ≈ sum(v̄)
    @test Devices.operator(Hamiltonian(t), device) ≈ sum(h̄) + G + sum(v̄)

    # CONSISTENCY OF DRESSED BASIS
    Λ0, U0 = eigen(H0)
    U = Devices.basisrotation(CtrlVQE.DRESSED, CtrlVQE.OCCUPATION, device)
    Hd = CtrlVQE.LinearAlgebraTools.rotate!(U, H0)
    @test Diagonal(Hd) ≈ Hd                 # STATIC HAMILONIAN DIAGONAL IN DRESSED BASIS
    @test sort(diag(Hd)) ≈ Λ0               # EIGENSPECTRUM IS UNCHANGED IN EITHER BASIS

    # OPERATOR METHOD CONSISTENCIES

    ψ0 = zeros(ComplexF64, N); ψ0[N] = 1
    ops = [
        IDENTITY, COUPLING, STATIC,
        Qubit(n), Channel(nD,t), Gradient(nG,t),
        Drive(t), Hamiltonian(t),
    ]

    for op in ops
        H = Devices.operator(op, device)
        @test eltype(op, device) == eltype(H)
        H_ = zero(H); Devices.operator(op, device; result=H_)
        @test H ≈ H_

        U = Devices.propagator(op, device, τ)
        @test U ≈ exp((-im*τ) .* H)
        U_ = zero(U); Devices.propagator(op, device, τ; result=U_)
        @test U ≈ U_

        ψ = copy(ψ0); Devices.propagate!(op, device, τ, ψ)
        @test ψ ≈ U * ψ0

        if op isa StaticOperator
            UT = Devices.evolver(op, device, τ)
            @test UT ≈ U
            UT_ = zero(UT); Devices.evolver(op, device, τ; result=UT_)
            @test UT ≈ UT_

            ψ_ = copy(ψ0); Devices.evolve!(op, device, τ, ψ_)
            @test ψ ≈ ψ_
        end

        E = Devices.expectation(op, device, ψ)
        @test abs(E - (ψ' * H * ψ)) < 1e-8
        V = Devices.braket(op, device, ψ, ψ0)
        @test abs(V - (ψ' * H * ψ0)) < 1e-8

    end

    # LOCAL METHODS

    h̄L = Devices.localqubitoperators(device)
    @test size(h̄L) == (m,m,n)
    h̄L_ = zero(h̄L); Devices.localqubitoperators(device; result=h̄L_)
    @test h̄L ≈ h̄L_

    ūL = Devices.localqubitpropagators(device, τ)
    @test size(ūL) == (m,m,n)
    ūL_ = zero(ūL); Devices.localqubitpropagators(device, τ; result=ūL_)
    @test ūL ≈ ūL_

    ūtL = Devices.localqubitevolvers(device, τ)
    @test size(ūtL) == (m,m,n)
    ūtL_ = zero(ūtL); Devices.localqubitevolvers(device, τ; result=ūtL_)
    @test ūtL ≈ ūtL_

    for q in 1:n
        hL = h̄L[:,:,q]; uL = ūL[:,:,q]; utL = ūtL[:,:,q]
        @test Devices.globalize(device, hL, q) ≈ h̄[q]
        @test uL ≈ exp((-im*τ) .* hL)
        @test uL ≈ utL
    end

    # FREQUENCIES

    Δ = [
        Devices.detuningfrequency(device, i, q)
            for i in 1:Devices.ndrives(device) for q in 1:Devices.nqubits(device)
    ]

    Δ_ = [
        Devices.drivefrequency(device, i) - Devices.resonancefrequency(device, q)
            for i in 1:Devices.ndrives(device) for q in 1:Devices.nqubits(device)
    ]

    @test Δ ≈ Δ_

    # SIGNALS

    signals = Devices.__get__drivesignals(device)
    @test all(isa.(signals, Signals.SignalType{F,FΩ}))

    # Call the mutating set_drivesignal, but we don't want to actually change anything.
    for (i, signal) in enumerate(signals); Devices.set_drivesignal(device, i, signal); end

    # Check the accessor gives identical objects.
    for (i, signal) in enumerate(signals); @assert Devices.drivesignal(device, i) === signal; end


    return true
end

function validate(device::Devices.LocallyDrivenDevice)
    super = invoke(validate, Tuple{Devices.DeviceType}, device)
    !super && return false

    m = Devices.nlevels(device)
    n = Devices.nqubits(device)
    nD= Devices.ndrives(device)
    nG= Devices.ngrades(device)

    # QUBIT ASSIGNMENTS

    for i in 1:nD; @test 1 ≤ Devices.drivequbit(device,i) ≤ n; end
    for j in 1:nG; @test 1 ≤ Devices.gradequbit(device,j) ≤ n; end

    # LOCAL DRIVE METHODS

    ā = Devices.algebra(device)
    v̄ = [Devices.driveoperator(device, ā, i, t) for i in 1:nD]

    v̄L = Devices.localdriveoperators(device, t)
    @test size(v̄L) == (m,m,n)
    v̄L_ = zero(v̄L); Devices.localdriveoperators(device, t; result=v̄L_)
    @test v̄L ≈ v̄L_

    ūL = Devices.localdrivepropagators(device, τ, t)
    @test size(ūL) == (m,m,n)
    ūL_ = zero(ūL); Devices.localdrivepropagators(device, τ, t; result=ūL_)
    @test ūL ≈ ūL_

    for q in 1:n
        vL = v̄L[:,:,q]; uL = ūL[:,:,q]
        @test Devices.globalize(device, vL, q) ≈ v̄[q]
        @test uL ≈ exp((-im*τ) .* vL)
    end

    return true
end






function validate(signal::Signals.SignalType{P,R}) where {P,R}

    # TEST PARAMETERS

    L = Parameters.count(signal)
    x̄ = Parameters.values(signal)
    @test eltype(x̄) == P
    @test length(x̄) == length(Parameters.names(signal)) == L

    Parameters.bind(signal, 2 .* x̄)
    @test Parameters.values(signal) ≈ 2 .* x̄
    Parameters.bind(signal, x̄)

    # TEST FUNCTION CONSISTENCY

    ft = Signals.valueat(signal, t)
    @test eltype(ft) == R

    @test ft ≈ signal(t)

    ft̄ = Signals.valueat(signal, t̄)
    @test eltype(ft̄) == R
    @test ft̄ ≈ [Signals.valueat(signal, t_) for t_ in t̄]
    ft̄_ = zero(ft̄); Signals.valueat(signal, t̄; result=ft̄_)
    @test ft̄ ≈ ft̄_

    # TEST GRADIENT CONSISTENCY

    for i in 1:L
        gt = Signals.partial(i, signal, t)
        @test eltype(gt) == R

        gt̄ = Signals.partial(i, signal, t̄)
        @test eltype(gt̄) == R
        @test gt̄ ≈ [Signals.partial(i, signal, t_) for t_ in t̄]
        gt̄_ = zero(gt̄); Signals.partial(i, signal, t̄; result=gt̄_)
        @test gt̄ ≈ gt̄_
    end

    # CHECK GRADIENT AGAINST THE FINITE DIFFERENCE

    g0 = [Signals.partial(i, signal, t) for i in 1:L]
    @test eltype(g0) == R

    function fℜ(x)
        Parameters.bind(signal, x)
        fx = real(Signals.valueat(signal, t))
        Parameters.bind(signal, x̄)  # RESTORE ORIGINAL VALUES
        return fx
    end
    gΔℜ = grad(central_fdm(5, 1), fℜ, x̄)[1]

    if R <: Complex
        function fℑ(x)
            Parameters.bind(signal, x)
            fx = imag(Signals.valueat(signal, t))
            Parameters.bind(signal, x̄)  # RESTORE ORIGINAL VALUES
            return fx
        end
        gΔℑ = grad(central_fdm(5, 1), fℑ, x̄)[1]
        gΔ = Complex.(gΔℜ, gΔℑ)
    else
        gΔ = gΔℜ
    end

    εg = g0 .- gΔ
    @test √(sum(abs2.(εg))./length(εg)) < 1e-5

    # CONVENIENCE FUNCTIONS
    @test typeof(string(signal)) == String

    return true
end

function validate(evolution::Evolutions.EvolutionType)
    seed!(0)            # FOR GENERATING INITIAL STATEVECTORS AND OBSERVABLES

    workbasis = Evolutions.workbasis(evolution)
    newbasis = (workbasis == OCCUPATION) ? DRESSED : OCCUPATION

    ######################################################################################
    # CONSISTENCY TESTS: 2-QUBIT 3-LEVEL SYSTEM WITH COMPLEX PULSES

    pulses = [
        ConstantSignals.ComplexConstant( 2π * 0.02, -2π * 0.02),
        ConstantSignals.ComplexConstant(-2π * 0.02,  2π * 0.02),
    ]
    device = CtrlVQE.Systematic(TransmonDevices.TransmonDevice, 2, pulses; m=3)

    TransmonDevices.bindfrequencies(device, [
        Devices.resonancefrequency(device, 1) + Δ,
        Devices.resonancefrequency(device, 2) - Δ,
    ])

    N = Devices.nstates(device)
    U = Devices.basisrotation(newbasis, workbasis, device)
                                            # BASIS ROTATION, FOR VALIDATING CONSISTENCY

    ψ0 = rand(ComplexF64, N)          # ARBITRARY INITIAL STATEVECTOR
    ψ0n = copy(ψ0);                   # SAME VECTOR IN NEW BASIS
        LinearAlgebraTools.rotate!(U, ψ0n)

    Ō = rand(ComplexF64, (N,N,2));    # TWO OBSERVABLES TO TEST `gradientsignals`
        for k in axes(Ō,3); Ō[:,:,k] .= Hermitian(@view(Ō[:,:,k])); end
    Ōn = copy(Ō);                     # SAME OBSERVABLES IN NEW BASIS
        for k in axes(Ō,3); LinearAlgebraTools.rotate!(U, @view(Ōn[:,:,k])); end

    # `evolve` METHODS

    ψ = Evolutions.evolve(evolution, device, grid, ψ0)
    ψ_ = zero(ψ)                        # RESULT VECTOR FOR SUBSEQUENT TESTS

    @test norm(ψ0) ≈ norm(ψ)            # TIME EVOLUTION SHOULD BE UNITARY

    ψ_ .= Evolutions.evolve(evolution, device, workbasis, grid, ψ0)
    @test ψ ≈ ψ_
    ψ_ .= 0; Evolutions.evolve(evolution, device, grid, ψ0; result=ψ_)
    @test ψ ≈ ψ_
    ψ_ .= 0; Evolutions.evolve(evolution, device, workbasis, grid, ψ0; result=ψ_)
    @test ψ ≈ ψ_

    ψ_ .= ψ0; Evolutions.evolve!(evolution, device, grid, ψ_)
    @test ψ ≈ ψ_
    ψ_ .= ψ0; Evolutions.evolve!(evolution, device, workbasis, grid, ψ_)
    @test ψ ≈ ψ_

    ψ_ .= ψ0n;
        Evolutions.evolve!(evolution, device, newbasis, grid, ψ_);
        LinearAlgebraTools.rotate!(U', ψ_)
    @test ψ ≈ ψ_

    # `gradientsignals` METHODS

    ϕ̄ = Evolutions.gradientsignals(evolution, device, grid, ψ0, Ō)
    ϕ̄_ = zero(ϕ̄)                        # RESULT VECTOR FOR SUBSEQUENT TESTS

    ϕ̄_ .= Evolutions.gradientsignals(evolution, device, workbasis, grid, ψ0, Ō)
    @test ϕ̄ ≈ ϕ̄_

    ϕ̄_ .= 0; Evolutions.gradientsignals(evolution, device, grid, ψ0, Ō; result=ϕ̄_)
    @test ϕ̄ ≈ ϕ̄_

    ϕ̄_ .= 0;
        Evolutions.gradientsignals(evolution, device, workbasis, grid, ψ0, Ō; result=ϕ̄_)
    @test ϕ̄ ≈ ϕ̄_

    ϕ̄_ .= 0;
        Evolutions.gradientsignals(evolution, device, newbasis, grid, ψ0n,Ōn; result=ϕ̄_)
    @test ϕ̄ ≈ ϕ̄_

    # `gradientsignals`, SINGLE OBSERVABLE

    O1 = Ō[:,:,1]
    O1n = Ōn[:,:,1]

    ϕ1 = Evolutions.gradientsignals(evolution, device, grid, ψ0, O1)
    ϕ1_ = zero(ϕ1)                      # RESULT VECTOR FOR SUBSEQUENT TESTS

    @test ϕ1 ≈ @view(ϕ̄[:,:,1])          # SINGLE/MULTI OBSERVABLE VERSIONS SHOULD MATCH

    ϕ1_ .= Evolutions.gradientsignals(evolution, device, workbasis, grid, ψ0, O1)
    @test ϕ1 ≈ ϕ1_

    ϕ1_ .= 0; Evolutions.gradientsignals(evolution, device, grid, ψ0, O1; result=ϕ1_)
    @test ϕ1 ≈ ϕ1_

    ϕ1_ .= 0;
        Evolutions.gradientsignals(evolution, device, workbasis, grid, ψ0, O1; result=ϕ1_)
    @test ϕ1 ≈ ϕ1_

    ϕ1_ .= 0;
        Evolutions.gradientsignals(evolution, device, newbasis, grid, ψ0n,O1n; result=ϕ1_)
    @test ϕ1 ≈ ϕ1_

    # CHECK CALLBACKS HAVE THE APPROPRIATE FORM

    callback(i, t, ψ) = (
        @assert isa(i, Integer);
        @assert isa(t, Real);
        @assert isa(ψ, AbstractArray{eltype(ψ0)});
        @assert size(ψ) == size(ψ0);
    )

    Evolutions.evolve(evolution, device, grid, ψ0; callback=callback)
    Evolutions.gradientsignals(evolution, device, grid, ψ0, O1; callback=callback)

    ######################################################################################
    # ACCURACY TESTS: 1-QUBIT 2 and 3-LEVEL SYSTEMS WITH COMPLEX PULSES

    loadedanalyticpulses || return true # SKIP ACCURACY TESTS IF WE DIDN'T LOAD THE MODULE

    finegrid = TrapezoidalIntegration(0.0, T, 10000)

    # pulse = ConstantSignals.ComplexConstant(2π * 0.02, -2π * 0.02)
    #= TODO (mid): I'd rather use a complex amplitude here, but it fails for m=3.
        Most likely this is a bug in AnalyticPulses.
        BUT I thought I'd already made both solutions compatible with complex amplitudes,
            so there is a slim chance it is a real problem with the code...
    =#
    pulse = ConstantSignals.Constant(2π * 0.02)

    infidelity(ψ, ϕ) = 1 - abs2(ψ'*ϕ)

    # ONE QUBIT
    device_2 = CtrlVQE.Systematic(TransmonDevices.TransmonDevice, 1, pulse; m=2)
    ω = Devices.resonancefrequency(device_2, 1)
    δ = TransmonDevices.anharmonicity(device_2, 1)
    Ω = Devices.drivesignal(device_2, 1)(T)
    TransmonDevices.bindfrequencies(device_2, [ω+Δ])

    ψ02 = rand(ComplexF64, Devices.nstates(device_2)); ψ02 ./= norm(ψ02)
    ψ2 = Evolutions.evolve(evolution, device_2, OCCUPATION, finegrid, ψ02)
    ψ2_ = evolve_transmon(ω, δ, Ω, ω+Δ, T, ψ02)
    @test ψ2 ≈ ψ2_

    # ONE QUTRIT
    device_3 = CtrlVQE.Systematic(TransmonDevices.TransmonDevice, 1, pulse; m=3)
    ω = Devices.resonancefrequency(device_3, 1)
    δ = TransmonDevices.anharmonicity(device_3, 1)
    Ω = Devices.drivesignal(device_3, 1)(T)
    TransmonDevices.bindfrequencies(device_3, [ω+Δ])

    ψ03 = rand(ComplexF64, Devices.nstates(device_3)); ψ03 ./= norm(ψ03)
    ψ3 = Evolutions.evolve(evolution, device_3, OCCUPATION, finegrid, ψ03)
    ψ3_ = evolve_transmon(ω, δ, Ω, ω+Δ, T, ψ03)
    @test ψ3 ≈ ψ3_

    # NOTE: Accuracy of gradient is deferred to unit-tests on energy functions.

    return true
end

function validate(costfn::CostFunctions.CostFunctionType{F}) where {F}
    L = length(costfn)
    F_ = eltype(costfn)
    @test F_ == F

    seed!(0)
    x̄ = rand(F, L)
    ∇f̄ = Vector{F}(undef, L)

    f  = CostFunctions.cost_function(costfn)
    g! = CostFunctions.grad_function_inplace(costfn)
    g  = CostFunctions.grad_function(costfn)

    # CONSISTENCY OF THE TWO FUNCTION INTERFACES
    @test f(x̄) == costfn(x̄)

    # CONSISTENCY OF THE TWO GRADIENT METHODS
    g!(∇f̄, x̄)
    @test ∇f̄ == g(x̄)

    # ACCURACY OF THE GRADIENT, COMPARED TO FINITE DIFFERENCE
    gΔ = grad(central_fdm(5, 1), f, x̄)[1]

    εg = ∇f̄ .- gΔ
    rms_error = √(sum(abs2.(εg))./length(εg))
    @test rms_error < 1e-5

    return true
end

# TODO: Validate extended interface for energy functions.

function validate(grid::Integrations.IntegrationType)
    r = Integrations.nsteps(grid)

    # CHECK `lattice` MATCHES `timeat`
    t̄ = Integrations.lattice(grid)
    @test length(t̄) == r+1
    @test all( i -> Integrations.timeat(grid,i) ≈ t̄[i+1], 0:r)

    # CHECK `reversed` REALLY IS `grid` BACKWARDS
    reversed = reverse(grid)
    @test all( i -> Integrations.timeat(reversed,r-i) ≈ t̄[i+1], 0:r)

    # CHECK THAT THE SCALAR METHODS ALL DO WHAT THEY'RE SUPPOSED TO
    t0 = Integrations.starttime(grid)
    @test t0 ≈ Integrations.timeat(grid, 0)

    tf = Integrations.endtime(grid)
    @test tf ≈ Integrations.timeat(grid, r)

    T = Integrations.duration(grid)
    @test T ≈ tf - t0

    τ = Integrations.stepsize(grid)
    @test τ ≈ T / r

    # CHECK THAT THE INTEGRATION STEPS...um, INTEGRATE CORRECTLY
    @test sum(Integrations.stepat(grid,i) for i in 0:r) ≈ T

    # CHECK THAT TYPINGS ARE CONSISTENT
    F = eltype(grid)
    @test eltype(t̄) == F
    @test eltype(Integrations.timeat(grid, 0)) == F
    @test eltype(Integrations.stepat(grid, 0)) == F
    @test eltype(t0) == F
    @test eltype(tf) == F
    @test eltype(T) == F
    @test eltype(τ) == F

    # CHECK INTEGRATION METHODS FOR SOME TOY FUNCTIONS
    ONES = ones(F, r+1)
    @test Integrations.integrate(grid, ONES) ≈ T
    @test Integrations.integrate(grid, t -> 1) ≈ T
    @test Integrations.integrate(grid, (t,k) -> k, ONES) ≈ T

end