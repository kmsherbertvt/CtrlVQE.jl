#= Systematic and comprehensive type stability unit tests. =#

import CtrlVQE
import CtrlVQE: Parameters, LinearAlgebraTools
import CtrlVQE: Integrations, Signals, Devices, Evolutions, CostFunctions

import CtrlVQE: TrapezoidalIntegration
import CtrlVQE: ConstantSignals
import CtrlVQE: TransmonDevices

import CtrlVQE: DRESSED, OCCUPATION
import CtrlVQE: StaticOperator, IDENTITY, COUPLING, STATIC
import CtrlVQE: Qubit, Channel, Drive, Hamiltonian, Gradient

using Random: seed!
using LinearAlgebra: Hermitian

const t = 1.0
const T = 5.0
const r = 10000
const τ = T / r
const t̄ = range(0.0, T, r+1)
const ϕ̄ = ones(r+1)

const grid = TrapezoidalIntegration(0.0, T, r)
const Δ = 2π * 0.3 # GHz  # DETUNING FOR TESTING

function check_types(device::Devices.DeviceType)
    @code_warntype Devices.nlevels(device)
    @code_warntype Devices.nqubits(device); q = Devices.nqubits(device)
    @code_warntype Devices.nstates(device); N = Devices.nstates(device)
    @code_warntype Devices.ndrives(device); i = Devices.ndrives(device)
    @code_warntype Devices.ngrades(device); j = Devices.ngrades(device)

    println("Parameters and Gradient")

    @code_warntype Parameters.count(device)
    @code_warntype Parameters.names(device)
    @code_warntype Parameters.values(device); x̄ = Parameters.values(device)
    @code_warntype Parameters.bind(device, x̄)

    ϕ̄_ = repeat(ϕ̄, 1, j)
    @code_warntype Devices.gradient(device, τ̄, t̄, ϕ̄_)
    grad = Devices.gradient(device, τ̄, t̄, ϕ̄_)
    @code_warntype Devices.gradient(device, τ̄, t̄, ϕ̄_; result=grad)


    println("Hamiltonian Terms")
    @code_warntype Devices.algebra(device); ā = Devices.algebra(device)

    @code_warntype Devices.eltype_localloweringoperator(device)
    @code_warntype Devices.localloweringoperator(device)
    a = Devices.localloweringoperator(device)
    @code_warntype Devices.localloweringoperator(device; result=a)

    @code_warntype Devices.eltype_qubithamiltonian(device)
    @code_warntype Devices.qubithamiltonian(device, ā, q)
    B = Devices.qubithamiltonian(device, ā, q)
    @code_warntype Devices.qubithamiltonian(device, ā, q; result=B)

    @code_warntype Devices.eltype_staticcoupling(device)
    @code_warntype Devices.staticcoupling(device, ā)
    C = Devices.staticcoupling(device, ā)
    @code_warntype Devices.staticcoupling(device, ā; result=C)

    @code_warntype Devices.eltype_driveoperator(device)
    @code_warntype Devices.driveoperator(device, ā, i, t)
    D = Devices.driveoperator(device, ā, i, t)
    @code_warntype Devices.driveoperator(device, ā, i, t; result=D)

    @code_warntype Devices.eltype_gradeoperator(device)
    @code_warntype Devices.gradeoperator(device, ā, j, t)
    E = Devices.gradeoperator(device, ā, j, t)
    @code_warntype Devices.gradeoperator(device, ā, j, t; result=E)


    println("Basis Control")

    @code_warntype Devices.globalize(device, a, q)
    F = Devices.globalize(device, a, q)
    @code_warntype Devices.globalize(device, a, q; result=F)

    @code_warntype Devices.diagonalize(DRESSED, device)
    @code_warntype Devices.diagonalize(OCCUPATION, device)
    @code_warntype Devices.diagonalize(OCCUPATION, device, q)

    @code_warntype Devices.basisrotation(DRESSED, OCCUPATION, device)
    @code_warntype Devices.basisrotation(OCCUPATION, OCCUPATION, device)
    @code_warntype Devices.basisrotation(OCCUPATION, OCCUPATION, device, q)
    @code_warntype Devices.localbasisrotations(OCCUPATION, OCCUPATION, device)

    @code_warntype Devices.eltype_algebra(device)
    @code_warntype Devices.eltype_algebra(device, DRESSED)
    @code_warntype Devices.algebra(device, DRESSED)
    @code_warntype Devices.localalgebra(device)

    ψ = zeros(ComplexF64, N); ψ[N] = 1
    ops = [
        IDENTITY,
        COUPLING,
        STATIC,
        Qubit(q),
        Channel(i,t),
        Drive(t),
        Hamiltonian(t),
        Gradient(j,t),
    ]

    println("Operator Methods")
    for op in ops
        println("> $op")

        @code_warntype Devices.eltype(op, device)
        @code_warntype Devices.eltype(op, device, DRESSED)

        @code_warntype Devices.operator(op, device)
        G = Devices.operator(op, device)
        @code_warntype Devices.operator(op, device; result=G)
        @code_warntype Devices.operator(op, device, DRESSED)
        H = Devices.operator(op, device, DRESSED)
        @code_warntype Devices.operator(op, device, DRESSED; result=H)

        @code_warntype Devices.propagator(op, device, τ)
        I = Devices.propagator(op, device, τ)
        @code_warntype Devices.propagator(op, device, τ; result=I)
        @code_warntype Devices.propagator(op, device, DRESSED, τ)
        J = Devices.propagator(op, device, DRESSED, τ)
        @code_warntype Devices.propagator(op, device, DRESSED, τ; result=J)

        @code_warntype Devices.propagate!(op, device, τ, ψ)
        @code_warntype Devices.propagate!(op, device, DRESSED, τ, ψ)

        if op isa StaticOperator
            @code_warntype Devices.evolver(op, device, τ)
            M = Devices.evolver(op, device, τ)
            @code_warntype Devices.evolver(op, device, τ; result=M)
            @code_warntype Devices.evolver(op, device, DRESSED, τ)
            N = Devices.evolver(op, device, DRESSED, τ)
            @code_warntype Devices.evolver(op, device, DRESSED, τ; result=N)

            @code_warntype Devices.evolve!(op, device, τ, ψ)
            @code_warntype Devices.evolve!(op, device, DRESSED, τ, ψ)
        end

        @code_warntype Devices.expectation(op, device, ψ)
        @code_warntype Devices.expectation(op, device, DRESSED, ψ)

        @code_warntype Devices.braket(op, device, ψ, ψ)
        @code_warntype Devices.braket(op, device, DRESSED, ψ, ψ)
    end


    println("Local Methods")
    @code_warntype Devices.localqubitoperators(device)
    Q = Devices.localqubitoperators(device)
    @code_warntype Devices.localqubitoperators(device; result=Q)
    @code_warntype Devices.localqubitoperators(device, OCCUPATION)
    R = Devices.localqubitoperators(device, OCCUPATION)
    @code_warntype Devices.localqubitoperators(device, OCCUPATION; result=R)

    @code_warntype Devices.localqubitpropagators(device, τ)
    S = Devices.localqubitpropagators(device, τ)
    @code_warntype Devices.localqubitpropagators(device, τ; result=S)
    @code_warntype Devices.localqubitpropagators(device, OCCUPATION, τ)
    T = Devices.localqubitpropagators(device, OCCUPATION, τ)
    @code_warntype Devices.localqubitpropagators(device, OCCUPATION, τ; result=T)

    @code_warntype Devices.localqubitevolvers(device, τ)
    U = Devices.localqubitevolvers(device, τ)
    @code_warntype Devices.localqubitevolvers(device, τ; result=U)
    @code_warntype Devices.localqubitevolvers(device, OCCUPATION, τ)
    V = Devices.localqubitevolvers(device, OCCUPATION, τ)
    @code_warntype Devices.localqubitevolvers(device, OCCUPATION, τ; result=V)

    return nothing
end

function check_types(device::Devices.LocallyDrivenDevice)
    invoke(check_types, Tuple{Devices.DeviceType}, device)
    i = Devices.ndrives(device)
    j = Devices.ngrades(device)

    @code_warntype Devices.drivequbit(device, i)
    @code_warntype Devices.gradequbit(device, j)

    println("Local Drive Methods")
    @code_warntype Devices.localdriveoperators(device, t)
    W = Devices.localdriveoperators(device, t)
    @code_warntype Devices.localdriveoperators(device, t; result=W)
    @code_warntype Devices.localdriveoperators(device, OCCUPATION, t)
    X = Devices.localdriveoperators(device, OCCUPATION, t)
    @code_warntype Devices.localdriveoperators(device, OCCUPATION, t;
            result=X)

    @code_warntype Devices.localdrivepropagators(device, τ, t)
    Y = Devices.localdrivepropagators(device, τ, t)
    @code_warntype Devices.localdrivepropagators(device, τ, t; result=Y)
    @code_warntype Devices.localdrivepropagators(device, OCCUPATION, τ, t)
    Z = Devices.localdrivepropagators(device, OCCUPATION, τ, t)
    @code_warntype Devices.localdrivepropagators(device, OCCUPATION, τ, t;
            result=Z)

    return nothing
end





function check_types(signal::Signals.SignalType{P,R}) where {P,R}

    # TEST PARAMETERS

    @code_warntype Parameters.count(signal); L = Parameters.count(signal)
    @code_warntype Parameters.values(signal); x̄ = Parameters.values(signal)
    @code_warntype Parameters.names(signal); names = Parameters.names(signal)
    @code_warntype Parameters.bind(signal, x̄)

    # TEST FUNCTION CONSISTENCY

    @code_warntype Signals.valueat(signal, t)
    @code_warntype signal(t)
    @code_warntype Signals.valueat(signal, t̄); ft̄ = Signals.valueat(signal, t̄)
    @code_warntype Signals.valueat(signal, t̄; result=ft̄)

    # TEST GRADIENT CONSISTENCY

    @code_warntype Signals.partial(L, signal, t)
    @code_warntype Signals.partial(L, signal, t̄); gt̄ = Signals.partial(L, signal, t̄)
    @code_warntype Signals.partial(L, signal, t̄; result=gt̄)

    # CONVENIENCE FUNCTIONS

    @code_warntype string(signal, names)
    @code_warntype string(signal)

    return nothing
end


######################################################################################
# SETUP AUXILIARY OBJECTS FOR EVOLUTION

const pulses = [
    ConstantSignals.ComplexConstant( 2π * 0.02, -2π * 0.02),
    ConstantSignals.ComplexConstant(-2π * 0.02,  2π * 0.02),
]
const device = CtrlVQE.Systematic(TransmonDevices.TransmonDevice, 2, pulses; m=3)

TransmonDevices.bindfrequencies(device, [
    TransmonDevices.resonancefrequency(device, 1) + Δ,
    TransmonDevices.resonancefrequency(device, 2) - Δ,
])

const N = Devices.nstates(device)

seed!(0)
const ψ0 = rand(ComplexF64, N)          # ARBITRARY INITIAL STATEVECTOR
const Ō = rand(ComplexF64, (N,N,2));    # TWO OBSERVABLES TO TEST `gradientsignals`
    for k in axes(Ō,3); Ō[:,:,k] .= Hermitian(@view(Ō[:,:,k])); end
const O1 = Ō[:,:,1]

######################################################################################


function check_types(evolution::Evolutions.EvolutionType)

    println("Static Methods")
    @code_warntype Evolutions.workbasis(evolution)
    basis = Evolutions.workbasis(evolution)

    println("Evolutions")
    @code_warntype Evolutions.evolve(evolution, device, grid, ψ0)
    @code_warntype Evolutions.evolve(evolution, device, basis, grid, ψ0)
    ψ = Evolutions.evolve(evolution, device, basis, grid, ψ0)
    @code_warntype Evolutions.evolve(evolution, device, grid, ψ0; result=ψ)
    @code_warntype Evolutions.evolve(evolution, device, basis, grid, ψ0; result=ψ)
    @code_warntype Evolutions.evolve!(evolution, device, grid, ψ)
    @code_warntype Evolutions.evolve!(evolution, device, basis, grid, ψ)

    println("Gradient Signals - Multi-operator")
    @code_warntype Evolutions.gradientsignals(evolution, device, grid, ψ0, Ō)
    @code_warntype Evolutions.gradientsignals(evolution, device, basis, grid, ψ0, Ō)
    ϕ̄ = Evolutions.gradientsignals(evolution, device, basis, grid, ψ0, Ō)
    @code_warntype Evolutions.gradientsignals(evolution, device, grid, ψ0, Ō; result=ϕ̄)
    @code_warntype Evolutions.gradientsignals(evolution,device,basis,grid,ψ0,Ō;result=ϕ̄)

    println("Gradient Signals - Single-operator")
    @code_warntype Evolutions.gradientsignals(evolution, device, grid, ψ0, O1)
    @code_warntype Evolutions.gradientsignals(evolution, device, basis, grid, ψ0, O1)
    ϕ̄ = Evolutions.gradientsignals(evolution, device, basis, grid, ψ0, O1)
    @code_warntype Evolutions.gradientsignals(evolution, device, grid, ψ0, O1; result=ϕ̄)
    @code_warntype Evolutions.gradientsignals(evolution,device,basis,grid,ψ0,O1;result=ϕ̄)

    return nothing
end

function check_types(costfn::CostFunctions.CostFunctionType)
    println("Static Methods")
    @code_warntype length(costfn); L = length(costfn)
    @code_warntype eltype(costfn); F = eltype(costfn)

    println("Factory Methods")
    @code_warntype CostFunctions.cost_function(costfn);
    f = CostFunctions.cost_function(costfn)
    @code_warntype CostFunctions.grad_function(costfn);
    g = CostFunctions.grad_function(costfn)
    @code_warntype CostFunctions.grad_function_inplace(costfn);
    g! = CostFunctions.grad_function_inplace(costfn)

    x  = zeros(F, L)

    println("Using Factories")
    @code_warntype f(x)
    @code_warntype costfn(x)
    @code_warntype g(x); ∇f = g(x)
    @code_warntype g!(∇f, x)

    return nothing
end

# TODO: Check types of extended interface for energy functions.

function check_types(grid::Integrations.IntegrationType)
    println("Fundamental operations")
    @code_warntype Integrations.nsteps(grid); r = Integrations.nsteps(grid)
    @code_warntype Integrations.timeat(grid, 0)
    @code_warntype Integrations.timeat(grid, r)
    @code_warntype Integrations.stepat(grid, 0)
    @code_warntype Integrations.stepat(grid, r)

    println("Scalar Operations")
    @code_warntype Integrations.starttime(grid)
    @code_warntype Integrations.endtime(grid)
    @code_warntype Integrations.duration(grid)
    @code_warntype Integrations.stepsize(grid)

    println("Heavy Operations")
    @code_warntype Integrations.lattice(grid)
    @code_warntype Integrations.reverse(grid)

    println("Integrations")
    ONES = ones(eltype(grid), r+1)
    @code_warntype Integrations.integrate(grid, ONES)
    @code_warntype Integrations.integrate(grid, t -> 1)
    @code_warntype Integrations.integrate(grid, (t,k) -> k, ONES)

    println("Vector Interface")
    @code_warntype eltype(grid)

end