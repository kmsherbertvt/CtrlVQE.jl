#= Systematic and comprehensive timing/allocation unit tests. =#

import CtrlVQE: Parameters, Signals, Devices
import CtrlVQE: DRESSED, OCCUPATION
import CtrlVQE: StaticOperator, IDENTITY, COUPLING, STATIC
import CtrlVQE: Qubit, Channel, Drive, Hamiltonian, Gradient

const t = 0.0
const r = 10
const τ = t / r

const τ̄ = fill(τ, r+1); τ̄[[1,end]] ./=2
const t̄ = range(0.0, t, r+1)
const ϕ̄ = ones(r+1)

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
    @time signal(t)
    @time ft̄ = signal(t̄)
    @time signal(t̄; result=ft̄)

    println("Partials")
    @time Signals.partial(L, signal, t)
    @time gt̄ = Signals.partial(L, signal, t̄)
    @time Signals.partial(L, signal, t̄; result=gt̄)

    println("Convenience Methods")
    @time string(signal, names)
    @time string(signal)

    @time Ip = Signals.integrate_partials(signal, τ̄, t̄, ϕ̄)
    @time Signals.integrate_partials(signal, τ̄, t̄, ϕ̄; result=Ip)

    @time Signals.integrate_signal(signal, τ̄, t̄, ϕ̄)

    return nothing
end

# TODO (hi): add method for CostFunctions