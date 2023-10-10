#= Systematic and comprehensive timing/allocation unit tests. =#

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

function check_times(device::Devices.DeviceType)
    @time l = Devices.nlevels(device)
    @time q = Devices.nqubits(device)
    @time N = Devices.nstates(device)
    @time i = Devices.ndrives(device)
    @time j = Devices.ngrades(device)

    println("Parameters and Gradient")

    @time L = Parameters.count(device)
    @time Parameters.names(device)
    @time x̄ = Parameters.values(device)

    @time Parameters.bind(device, 2 .* x̄)
    Parameters.bind(device, x̄)  # Don't mutate the device if you can help it...)

    ϕ̄_ = repeat(ϕ̄, 1, j)
    @time grad = Devices.gradient(device, τ̄, t̄, ϕ̄_)
    @time Devices.gradient(device, τ̄, t̄, ϕ̄_; result=grad)


    println("Hamiltonian Terms")
    @time ā = Devices.algebra(device)

    @time Devices.eltype_localloweringoperator(device)
    @time a = Devices.localloweringoperator(device)
    @time Devices.localloweringoperator(device; result=a)

    @time Devices.eltype_qubithamiltonian(device)
    @time B = Devices.qubithamiltonian(device, ā, q)
    @time Devices.qubithamiltonian(device, ā, q; result=B)

    @time Devices.eltype_staticcoupling(device)
    @time C = Devices.staticcoupling(device, ā)
    @time Devices.staticcoupling(device, ā; result=C)

    @time Devices.eltype_driveoperator(device)
    @time D = Devices.driveoperator(device, ā, i, t)
    @time Devices.driveoperator(device, ā, i, t; result=D)

    @time Devices.eltype_gradeoperator(device)
    @time E = Devices.gradeoperator(device, ā, j, t)
    @time Devices.gradeoperator(device, ā, j, t; result=E)


    println("Basis Control")

    @time F = Devices.globalize(device, a, q)
    @time Devices.globalize(device, a, q; result=F)

    @time Devices.diagonalize(DRESSED, device)
    @time Devices.diagonalize(OCCUPATION, device)
    @time Devices.diagonalize(OCCUPATION, device, q)

    @time Devices.basisrotation(DRESSED, OCCUPATION, device)
    @time Devices.basisrotation(OCCUPATION, OCCUPATION, device)
    @time Devices.basisrotation(OCCUPATION, OCCUPATION, device, q)
    @time Devices.localbasisrotations(OCCUPATION, OCCUPATION, device)

    @time Devices.eltype_algebra(device)
    @time Devices.eltype_algebra(device, DRESSED)
    @time Devices.algebra(device, DRESSED)
    @time Devices.localalgebra(device)

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

        @time Devices.eltype(op, device)
        @time Devices.eltype(op, device, DRESSED)

        @time G = Devices.operator(op, device)
        @time Devices.operator(op, device; result=G)
        @time H = Devices.operator(op, device, DRESSED)
        @time Devices.operator(op, device, DRESSED; result=H)

        @time I = Devices.propagator(op, device, τ)
        @time Devices.propagator(op, device, τ; result=I)
        @time J = Devices.propagator(op, device, DRESSED, τ)
        @time Devices.propagator(op, device, DRESSED, τ; result=J)

        @time Devices.propagate!(op, device, τ, ψ)
        @time Devices.propagate!(op, device, DRESSED, τ, ψ)

        if op isa StaticOperator
            @time M = Devices.evolver(op, device, τ)
            @time Devices.evolver(op, device, τ; result=M)
            @time N = Devices.evolver(op, device, DRESSED, τ)
            @time Devices.evolver(op, device, DRESSED, τ; result=N)

            @time Devices.evolve!(op, device, τ, ψ)
            @time Devices.evolve!(op, device, DRESSED, τ, ψ)
        end

        @time Devices.expectation(op, device, ψ)
        @time Devices.expectation(op, device, DRESSED, ψ)

        @time Devices.braket(op, device, ψ, ψ)
        @time Devices.braket(op, device, DRESSED, ψ, ψ)
    end


    println("Local Methods")
    @time Q = Devices.localqubitoperators(device)
    @time Devices.localqubitoperators(device; result=Q)
    @time R = Devices.localqubitoperators(device, OCCUPATION)
    @time Devices.localqubitoperators(device, OCCUPATION; result=R)

    @time S = Devices.localqubitpropagators(device, τ)
    @time Devices.localqubitpropagators(device, τ; result=S)
    @time T = Devices.localqubitpropagators(device, OCCUPATION, τ)
    @time Devices.localqubitpropagators(device, OCCUPATION, τ; result=T)

    @time U = Devices.localqubitevolvers(device, τ)
    @time Devices.localqubitevolvers(device, τ; result=U)
    @time V = Devices.localqubitevolvers(device, OCCUPATION, τ)
    @time Devices.localqubitevolvers(device, OCCUPATION, τ; result=V)

    return nothing
end

function check_times(device::Devices.LocallyDrivenDevice)
    invoke(check_times, Tuple{Devices.DeviceType}, device)
    i = Devices.ndrives(device)
    j = Devices.ngrades(device)

    @time qi = Devices.drivequbit(device, i)
    @time qj = Devices.gradequbit(device, j)

    println("Local Drive Methods")
    @time W = Devices.localdriveoperators(device, t)
    @time Devices.localdriveoperators(device, t; result=W)
    @time X = Devices.localdriveoperators(device, OCCUPATION, t)
    @time Devices.localdriveoperators(device, OCCUPATION, t; result=X)

    @time Y = Devices.localdrivepropagators(device, τ, t)
    @time Devices.localdrivepropagators(device, τ, t; result=Y)
    @time Z = Devices.localdrivepropagators(device, OCCUPATION, τ, t)
    @time Devices.localdrivepropagators(device, OCCUPATION, τ, t; result=Z)

    return nothing
end





function check_times(signal::Signals.SignalType{P,R}) where {P,R}

    println("Parameters Interface")
    @time L = Parameters.count(signal)
    @time x̄ = Parameters.values(signal)
    @time names = Parameters.names(signal)
    @time Parameters.bind(signal, x̄)

    println("Functions")
    @time Signals.valueat(signal, t)
    @time signal(t)
    @time ft̄ = Signals.valueat(signal, t̄)
    @time Signals.valueat(signal, t̄; result=ft̄)

    println("Partials")
    @time Signals.partial(L, signal, t)
    @time gt̄ = Signals.partial(L, signal, t̄)
    @time Signals.partial(L, signal, t̄; result=gt̄)

    println("Convenience Methods")
    @time string(signal, names)
    @time string(signal)

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

function check_times(evolution::Evolutions.EvolutionType)

    println("Static Methods")
    @time basis = Evolutions.workbasis(evolution)

    println("Evolutions")
    @time Evolutions.evolve(evolution, device, grid, ψ0)
    @time ψ = Evolutions.evolve(evolution, device, basis, grid, ψ0)
    @time Evolutions.evolve(evolution, device, grid, ψ0; result=ψ)
    @time Evolutions.evolve(evolution, device, basis, grid, ψ0; result=ψ)
    @time Evolutions.evolve!(evolution, device, grid, ψ)
    @time Evolutions.evolve!(evolution, device, basis, grid, ψ)

    println("Gradient Signals - Multi-operator")
    @time Evolutions.gradientsignals(evolution, device, grid, ψ0, Ō)
    @time ϕ̄ = Evolutions.gradientsignals(evolution, device, basis, grid, ψ0, Ō)
    @time Evolutions.gradientsignals(evolution, device, grid, ψ0, Ō; result=ϕ̄)
    @time Evolutions.gradientsignals(evolution, device, basis, grid, ψ0, Ō; result=ϕ̄)

    println("Gradient Signals - Single-operator")
    @time Evolutions.gradientsignals(evolution, device, grid, ψ0, O1)
    @time ϕ̄ = Evolutions.gradientsignals(evolution, device, basis, grid, ψ0, O1)
    @time Evolutions.gradientsignals(evolution, device, grid, ψ0, O1; result=ϕ̄)
    @time Evolutions.gradientsignals(evolution, device, basis, grid, ψ0, O1; result=ϕ̄)

    return nothing
end

function check_times(costfn::CostFunctions.CostFunctionType)
    println("Static Methods")
    @time L = length(costfn)
    @time F = eltype(costfn)

    println("Factory Methods")
    @time f = CostFunctions.cost_function(costfn)
    @time g = CostFunctions.grad_function(costfn)
    @time g! = CostFunctions.grad_function_inplace(costfn)

    x  = zeros(F, L)

    println("Using Factories")
    @time f(x)
    @time costfn(x)
    @time ∇f = g(x)
    @time g!(∇f, x)

    return nothing
end

# TODO: Time extended interface for energy functions.


function check_times(grid::Integrations.IntegrationType)
    println("Fundamental operations")
    @time r = Integrations.nsteps(grid)
    @time Integrations.timeat(grid, 0)
    @time Integrations.timeat(grid, r)
    @time Integrations.stepat(grid, 0)
    @time Integrations.stepat(grid, r)

    println("Scalar Operations")
    @time Integrations.starttime(grid)
    @time Integrations.endtime(grid)
    @time Integrations.duration(grid)
    @time Integrations.stepsize(grid)

    println("Heavy Operations")
    @time Integrations.lattice(grid)
    @time Integrations.reverse(grid)

    println("Integrations")
    ONES = ones(eltype(grid), r+1)
    @time Integrations.integrate(grid, ONES)
    @time Integrations.integrate(grid, t -> 1)
    @time Integrations.integrate(grid, (t,k) -> k, ONES)

    println("Vector Interface")
    @time eltype(grid)

end