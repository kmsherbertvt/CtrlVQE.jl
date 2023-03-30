using Memoization: @memoize
#= NOTE:

We are using `Memoization` to reconcile the following two considerations:
1. Simple-as-possible interface.
2. Don't re-calculate things you've already calculated.

The `Memoization` package is a bit beyond our control, generating two more problems:
1. Not every use-case will want every function call cached. Doing so is a waste of space.
2. If the state of any argument in a method changes, its cached value is no longer valid.

After toying with ideas for a number of more customized cache implementations,
    I've decided, to simply *not* memoize any function depending on an absolute time.
    I think that more or less solves the worst parts of problem 1.

    By chance, it seems like relative time only appears without absolute time in a context where the relative time could be interpreted as an absolute time (ie. propagation of a static hamiltonian), which means we actually just don't cache any times at all... Huh.

    Naw. We do very much desire staticpropagator(τ) to cache.
    If we definitely do not want staticpropagator(t) to cache,
        thing to do is to split one off into a new method.

As it happens, it also solves problem 2 in the short term,
    because, at present, static device parameters are considered fixed.
So, the changing state of the device would only actually impact time-dependent methods.

BUT

If ever we implement a device with "tunable couplings",
    such that time-independent parameters of a device are changed on `Parameters.bind(⋅)`,
    the implementation of `Parameters.bind` should CLEAR the cache:

    Memoization.empty_all_caches!()

Alternatively, selectively clear caches for affected functions via:

    Memoization.empty_cache!(fn)

I don't know if it's possible to selectively clear cached values for specific methods.
If it can be done, it would require obtaining the actual `IdDict`
    being used as a cache for a particular function,
    figuring out exactly how that cache is indexed,
    and manually removing elements matching your targeted method signature.

=#

using LinearAlgebra: I, Diagonal, Eigen, eigen
using ReadOnlyArrays: ReadOnlyArray
using ..LinearAlgebraTools: List
import ..Bases, ..Operators, ..LinearAlgebraTools, ..Signals


struct Quple
    q1::Int
    q2::Int
    # INNER CONSTRUCTOR: Constrain order so that `Qu...ple(q1,q2) == Qu...ple(q2,q1)`.
    Quple(q1, q2) = q1 > q2 ? new(q2, q1) : new(q1, q2)
end

# IMPLEMENT ITERATION, FOR CONVENIENT UNPACKING
Base.iterate(quple::Quple) = quple.q1, true
Base.iterate(quple::Quple, state) = state ? (quple.q2, false) : nothing

# TODO: Generalize to n qubits (call sort on input arguments) and hopefully subtype Tuple





"""
NOTE: Implements `Parameters` interface.
"""
abstract type Device end




# METHODS NEEDING TO BE IMPLEMENTED
nqubits(::Device)::Int = error("Not Implemented")
nstates(::Device, q::Int)::Int = error("Not Implemented")
ndrives(::Device)::Int = error("Not Implemented")
ngrades(::Device)::Int = error("Not Implemented")

function localloweringoperator(::Device,
    q::Int,
)::AbstractMatrix
    return error("Not Implemented")
end

function qubithamiltonian(::Device,
    ā::List{<:AbstractMatrix},
    q::Int,
)::AbstractMatrix
    return error("Not Implemented")
end

function staticcoupling(::Device,
    ā::List{<:AbstractMatrix},
)::AbstractMatrix
    return error("Not Implemented")
end

function driveoperator(::Device,
    ā::List{<:AbstractMatrix},
    i::Int,
    t::Real,
)::AbstractMatrix
    return error("Not Implemented")
end

function gradeoperator(::Device,
    ā::List{<:AbstractMatrix},
    j::Int,
    t::Real,
)::AbstractMatrix
    # Returns Hermitian Â such that ϕ = ⟨λ|(𝑖Â)|ψ⟩ + h.t.
    return error("Not Implemented")
end

function gradient(::Device,
    τ̄::AbstractVector,
    t̄::AbstractVector,
    ϕ̄::AbstractMatrix,
)::AbstractVector
    return error("Not Implemented")
end







# UTILITIES

function globalize(device::Device, op::AbstractMatrix, q::Int)
    ops = AbstractMatrix{eltype(op)}[]
    for p in 1:nqubits(device)
        if p == q
            push!(ops, op)
            continue
        end

        m = nstates(device, p)
        push!(ops, Matrix{eltype(op)}(I, m, m))
    end
    return LinearAlgebraTools.kron(ops)
end

function _cd_from_ix(i::Int, m̄::List{<:Integer})
    i = i - 1       # SWITCH TO INDEXING FROM 0
    ī = Vector{Int}(undef, length(m̄))
    for q in eachindex(m̄)
        i, ī[q] = divrem(i, m̄[q])
    end
    return ī
end

function _ix_from_cd(ī::AbstractVector{<:Integer}, m̄::List{<:Integer})
    i = 0
    offset = 1
    for q in eachindex(m̄)
        i += offset * ī[q]
        offset *= m̄[q]
    end
    return i + 1    # SWITCH TO INDEXING FROM 1
end

function project(device::Device, op::AbstractMatrix, m̄1::List{Int})
    N1 = size(op, 1)

    m̄2 = [nstates(device,q) for q in 1:nqubits(device)]
    ix_map = Dict(i1 => _ix_from_cd(_cd_from_ix(i1,m̄1),m̄2) for i1 in 1:N1)

    N2 = nstates(device)
    op2 = zeros(eltype(op), N2, N2)
    for i in 1:N1
        for j in 1:N1
            op2[ix_map[i],ix_map[j]] = op[i,j]
        end
    end
    return op2
end

function project(device::Device, op::AbstractMatrix, m::Int)
    return project(device, op, fill(m, nqubits(device)))
end

function project(device::Device, op::AbstractMatrix)
    # ASSUME `op` HAS UNIFORM NUMBER OF STATES ON EACH QUBIT
    m = round(Int, size(op,1) ^ (1/nqubits(device)))
    return project(device, op, m)
end




@memoize nstates(device::Device) = prod(nstates(device,q) for q in 1:nqubits(device))






# BASIS ROTATIONS

@memoize function diagonalize(::Type{Bases.Dressed}, device::Device)
    H0 = operator(Operators.Static, device)
    return eigen(H0)
    # TODO: Move code for Utils.dressedbasis to here.
end

@memoize function diagonalize(basis::Type{<:Bases.LocalBasis}, device::Device)
    ΛU = [diagonalize(basis, device, q) for q in 1:nqubits(device)]
    Λ = LinearAlgebraTools.kron([ΛU[q].values  for q in 1:nqubits(device)])
    U = LinearAlgebraTools.kron([ΛU[q].vectors for q in 1:nqubits(device)])
    return Eigen(Λ, U)
end

@memoize function diagonalize(::Type{Bases.Occupation}, device::Device, q::Int)
    a = localloweringoperator(device, q)
    I = one(a)
    return eigen(I)
end

@memoize function diagonalize(::Type{Bases.Coordinate}, device::Device, q::Int)
    a = localloweringoperator(device, q)
    Q = (a + a') / eltype(a)(√2)
    return eigen(Q)
end

@memoize function diagonalize(::Type{Bases.Momentum}, device::Device, q::Int)
    a = localloweringoperator(device, q)
    P = (a - a') / eltype(a)(√2)
    return eigen(P)
end



@memoize function basisrotation(
    src::Type{<:Bases.BasisType},
    tgt::Type{<:Bases.BasisType},
    device::Device,
)
    Λ0, U0 = diagonalize(src, device)
    Λ1, U1 = diagonalize(tgt, device)
    # |ψ'⟩ ≡ U0|ψ⟩ rotates |ψ⟩ OUT of `src` Bases.
    # U1'|ψ'⟩ rotates |ψ'⟩ INTO `tgt` Bases.
    return ReadOnlyArray(U1' * U0)
end

@memoize function basisrotation(
    src::Type{<:Bases.LocalBasis},
    tgt::Type{<:Bases.LocalBasis},
    device::Device,
)
    ū = localbasisrotations(src, tgt, device)
    return ReadOnlyArray(LinearAlgebraTools.kron(ū))
end

@memoize function basisrotation(
    src::Type{<:Bases.LocalBasis},
    tgt::Type{<:Bases.LocalBasis},
    device::Device,
    q::Int,
)
    Λ0, U0 = diagonalize(src, device, q)
    Λ1, U1 = diagonalize(tgt, device, q)
    # |ψ'⟩ ≡ U0|ψ⟩ rotates |ψ⟩ OUT of `src` Bases.
    # U1'|ψ'⟩ rotates |ψ'⟩ INTO `tgt` Bases.
    return ReadOnlyArray(U1' * U0)
end

@memoize function localbasisrotations(
    src::Type{<:Bases.LocalBasis},
    tgt::Type{<:Bases.LocalBasis},
    device::Device,
)
    return Tuple(
        ReadOnlyArray(basisrotation(src, tgt, device, q)) for q in 1:nqubits(device)
    )
end





#= ALGEBRAS =#

@memoize function algebra(
    device::Device,
    basis::Type{<:Bases.BasisType}=Bases.Occupation,
)
    U = basisrotation(Bases.Occupation, basis, device)
    ā = []
    for q in 1:nqubits(device)
        a0 = localloweringoperator(device, q)

        # CONVERT TO A NUMBER TYPE COMPATIBLE WITH ROTATION
        F = promote_type(eltype(U), eltype(a0))
        a0 = convert(Matrix{F}, a0)

        a = globalize(device, a0, q)
        a = LinearAlgebraTools.rotate!(U, a)
        push!(ā, ReadOnlyArray(a))
    end
    return Tuple(ā)
end

@memoize function localalgebra(
    device::Device,
    basis::Type{<:Bases.LocalBasis}=Bases.Occupation,
)
    ā = []
    for q in 1:nqubits(device)
        U = basisrotation(Bases.Occupation, basis, device, q)
        a0 = localloweringoperator(device, q)

        # CONVERT TO A NUMBER TYPE COMPATIBLE WITH ROTATION
        F = promote_type(eltype(U), eltype(a0))
        a0 = convert(Matrix{F}, a0)

        a = LinearAlgebraTools.rotate!(U, a0)
        push!(ā, ReadOnlyArray(a))
    end
    return Tuple(ā)
end


#= HERMITIAN OPERATORS =#

function operator(mode::Type{<:Operators.OperatorType}, device::Device, args...)
    return operator(mode, device, Bases.Occupation, args...)
end

@memoize function operator(::Type{Operators.Qubit},
    device::Device,
    basis::Type{<:Bases.BasisType},
    q::Int,
)
    ā = algebra(device, basis)
    return ReadOnlyArray(qubithamiltonian(device, ā, q))
end

@memoize function operator(::Type{Operators.Coupling},
    device::Device,
    basis::Type{<:Bases.BasisType},
)
    ā = algebra(device, basis)
    return ReadOnlyArray(staticcoupling(device, ā))
end

function operator(::Type{Operators.Channel},
    device::Device,
    basis::Type{<:Bases.BasisType},
    i::Int,
    t::Real,
)
    ā = algebra(device, basis)
    return driveoperator(device, ā, i, t)
end

function operator(::Type{Operators.Gradient},
    device::Device,
    basis::Type{<:Bases.BasisType},
    j::Int,
    t::Real,
)
    ā = algebra(device, basis)
    return gradeoperator(device, ā, j, t)
end

@memoize function operator(::Type{Operators.Uncoupled},
    device::Device,
    basis::Type{<:Bases.BasisType},
)
    return ReadOnlyArray(sum((
        operator(Operators.Qubit, device, basis, q)
            for q in 1:nqubits(device)
    )))
end

@memoize function operator(::Type{Operators.Static},
    device::Device,
    basis::Type{<:Bases.BasisType},
)
    return ReadOnlyArray(sum((
        operator(Operators.Uncoupled, device, basis),
        operator(Operators.Coupling,  device, basis),
    )))
end

@memoize function operator(::Type{Operators.Static},
    device::Device,
    ::Type{Bases.Dressed},
)
    Λ, U = diagonalize(Bases.Dressed, device)
    return Diagonal(ReadOnlyArray(Λ))
end

function operator(::Type{Operators.Drive},
    device::Device,
    basis::Type{<:Bases.BasisType},
    t::Real,
)
    return sum((
        operator(Operators.Channel, device, basis, i, t)
            for i in 1:ndrives(device)
    ))
end

function operator(::Type{Operators.Hamiltonian},
    device::Device,
    basis::Type{<:Bases.BasisType},
    t::Real,
)
    return sum((
        operator(Operators.Static, device, basis),
        operator(Operators.Drive,  device, basis, t),
    ))
end


function localqubitoperators(device::Device)
    return localqubitoperators(device, Bases.Occupation)
end

@memoize function localqubitoperators(
    device::Device,
    basis::Type{<:Bases.LocalBasis},
)
    ā = localalgebra(device, basis)
    return Tuple(
        ReadOnlyArray(qubithamiltonian(device, ā, q)) for q in 1:nqubits(device)
    )
end




#= PROPAGATORS =#



function propagator(mode::Type{<:Operators.OperatorType}, device::Device, args...)
    return propagator(mode, device, Bases.Occupation, args...)
end

function propagator(mode::Type{<:Operators.OperatorType},
    device::Device,
    basis::Type{<:Bases.BasisType},
    τ::Real,
    args...,
)
    H = operator(mode, device, basis, args...)
    H = convert(Array{LinearAlgebraTools.cis_type(H)}, H)
    return LinearAlgebraTools.cis!(H, -τ)
end

@memoize function propagator(mode::Type{<:Operators.StaticOperator},
    device::Device,
    basis::Type{<:Bases.BasisType},
    τ::Real,
    args...,
)
    # NOTE: This is a carbon copy of the above method, except with caching.
    H = operator(mode, device, basis, args...)
    H = convert(Array{LinearAlgebraTools.cis_type(H)}, H)
    return ReadOnlyArray(LinearAlgebraTools.cis!(H, -τ))
end

@memoize function propagator(::Type{Operators.Uncoupled},
    device::Device,
    basis::Type{<:Bases.LocalBasis},
    τ::Real,
)
    ū = localqubitpropagators(device, basis, τ)
    return ReadOnlyArray(LinearAlgebraTools.kron(ū))
end

@memoize function propagator(::Type{Operators.Qubit},
    device::Device,
    basis::Type{<:Bases.LocalBasis},
    τ::Real,
    q::Int,
)
    ā = localalgebra(device, basis)

    h = qubithamiltonian(device, ā, q)
    h = convert(Array{LinearAlgebraTools.cis_type(h)}, h)
    u = LinearAlgebraTools.cis!(h, -τ)
    return ReadOnlyArray(globalize(device, u, q))
end




function localqubitpropagators(device::Device, τ::Real)
    return localqubitpropagators(device, Bases.Occupation, τ)
end

@memoize function localqubitpropagators(
    device::Device,
    basis::Type{<:Bases.LocalBasis},
    τ::Real,
)
    h̄ = localqubitoperators(device, basis)
    ū = []
    for h in h̄
        h = convert(Array{LinearAlgebraTools.cis_type(h)}, h)
        u = LinearAlgebraTools.cis!(h, -τ)
        push!(ū, ReadOnlyArray(u))
    end
    return Tuple(ū)
end




#= MUTATING PROPAGATION =#

function propagate!(mode::Type{<:Operators.OperatorType}, device::Device, args...)
    return propagate!(mode, device, Bases.Occupation, args...)
end

function propagate!(mode::Type{<:Operators.OperatorType},
    device::Device,
    basis::Type{<:Bases.BasisType},
    τ::Real,
    ψ::AbstractVecOrMat{<:Complex{<:AbstractFloat}},
    args...,
)
    U = propagator(mode, device, basis, τ, args...)
    return LinearAlgebraTools.rotate!(U, ψ)
end

function propagate!(::Type{Operators.Uncoupled},
    device::Device,
    basis::Type{<:Bases.LocalBasis},
    τ::Real,
    ψ::AbstractVecOrMat{<:Complex{<:AbstractFloat}},
)
    ū = localqubitpropagators(device, basis, τ)
    return LinearAlgebraTools.rotate!(ū, ψ)
end

function propagate!(::Type{Operators.Qubit},
    device::Device,
    basis::Type{<:Bases.LocalBasis},
    τ::Real,
    ψ::AbstractVecOrMat{<:Complex{<:AbstractFloat}},
    q::Int,
)
    ā = localalgebra(device, basis)
    h = qubithamiltonian(device, ā, q)
    h = convert(Array{LinearAlgebraTools.cis_type(h)}, h)
    u = LinearAlgebraTools.cis!(h, -τ)
    ops = [p == q ? u : one(u) for p in 1:nqubits(n)]
    return LinearAlgebraTools.rotate!(ops, ψ)
end



#= PROPAGATORS FOR ARBITRARY TIME (static only) =#


function evolver(mode::Type{<:Operators.OperatorType}, device::Device, args...)
    return evolver(mode, device, Bases.Occupation, args...)
end

function evolver(mode::Type{<:Operators.OperatorType},
    device::Device,
    basis::Type{<:Bases.BasisType},
    t::Real,
    args...,
)
    error("Not implemented for non-static operator.")
end

function evolver(mode::Type{<:Operators.StaticOperator},
    device::Device,
    basis::Type{<:Bases.BasisType},
    t::Real,
    args...,
)
    H = operator(mode, device, basis, args...)
    H = convert(Array{LinearAlgebraTools.cis_type(H)}, H)
    return LinearAlgebraTools.cis!(H, -t)
end

function evolver(::Type{Operators.Uncoupled},
    device::Device,
    basis::Type{<:Bases.LocalBasis},
    t::Real,
)
    ū = localqubitevolvers(device, basis, t)
    return LinearAlgebraTools.kron(ū)
end

function evolver(::Type{Operators.Qubit},
    device::Device,
    basis::Type{<:Bases.LocalBasis},
    t::Real,
    q::Int,
)
    ā = localalgebra(device, basis)
    h = qubithamiltonian(device, ā, q)
    h = convert(Array{LinearAlgebraTools.cis_type(h)}, h)
    u = LinearAlgebraTools.cis!(h, -τ)
    return globalize(device, u, q)
end



function localqubitevolvers(device::Device, t::Real)
    return localqubitevolvers(device, Bases.Occupation, t)
end

function localqubitevolvers(
    device::Device,
    basis::Type{<:Bases.LocalBasis},
    t::Real,
)
    h̄ = localqubitoperators(device, basis)
    ū = []
    for h in h̄
        h = convert(Array{LinearAlgebraTools.cis_type(h)}, h)
        u = LinearAlgebraTools.cis!(h, -t)
        push!(ū, u)
    end
    return Tuple(ū)
end


#= MUTATING EVOLUTION FOR ARBITRARY TIME (static only) =#

function evolve!(mode::Type{<:Operators.OperatorType}, device::Device, args...)
    return evolve!(mode, device, Bases.Occupation, args...)
end

function evolve!(mode::Type{<:Operators.OperatorType},
    device::Device,
    basis::Type{<:Bases.BasisType},
    t::Real,
    ψ::AbstractVecOrMat{<:Complex{<:AbstractFloat}},
    args...,
)
    error("Not implemented for non-static operator.")
end

function evolve!(mode::Type{<:Operators.StaticOperator},
    device::Device,
    basis::Type{<:Bases.BasisType},
    t::Real,
    ψ::AbstractVecOrMat{<:Complex{<:AbstractFloat}},
    args...,
)
    U = evolver(mode, device, basis, t, args...)
    return LinearAlgebraTools.rotate!(U, ψ)
end

function evolve!(::Type{Operators.Uncoupled},
    device::Device,
    basis::Type{<:Bases.LocalBasis},
    t::Real,
    ψ::AbstractVecOrMat{<:Complex{<:AbstractFloat}},
)
    ū = localqubitevolvers(device, basis, t)
    return LinearAlgebraTools.rotate!(ū, ψ)
end

function evolve!(::Type{Operators.Qubit},
    device::Device,
    basis::Type{<:Bases.LocalBasis},
    t::Real,
    ψ::AbstractVecOrMat{<:Complex{<:AbstractFloat}},
    q::Int,
)
    ā = localalgebra(device, basis)
    h = qubithamiltonian(device, ā, q)
    h = convert(Array{LinearAlgebraTools.cis_type(h)}, h)
    u = LinearAlgebraTools.cis!(h, -t)
    ops = [p == q ? u : one(u) for p in 1:nqubits(n)]
    return LinearAlgebraTools.rotate!(ops, ψ)
end





#= SCALAR MATRIX OPERATIONS =#

function expectation(mode::Type{<:Operators.OperatorType}, device::Device, args...)
    return expectation(mode, device, Bases.Occupation, args...)
end

function expectation(mode::Type{<:Operators.OperatorType},
    device::Device,
    basis::Type{<:Bases.BasisType},
    ψ::AbstractVector,
    args...,
)
    H = operator(mode, device, basis, args...)
    return LinearAlgebraTools.expectation(H, ψ)
end

function braket(mode::Type{<:Operators.OperatorType}, device::Device, args...)
    return braket(mode, device, Bases.Occupation, args...)
end

function braket(mode::Type{<:Operators.OperatorType},
    device::Device,
    basis::Type{<:Bases.BasisType},
    ψ1::AbstractVector,
    ψ2::AbstractVector,
    args...,
)
    H = operator(mode, device, basis, args...)
    return LinearAlgebraTools.braket(ψ1, H, ψ2)
end

#= TODO: Localize expectation and braket, I suppose. =#














abstract type LocallyDrivenDevice <: Device end

# METHODS NEEDING TO BE IMPLEMENTED
drivequbit(::LocallyDrivenDevice, i::Int)::Int = error("Not Implemented")
gradequbit(::LocallyDrivenDevice, j::Int)::Int = error("Not Implemented")

# LOCALIZING DRIVE OPERATORS

function localdriveoperators(device::LocallyDrivenDevice, t::Real)
    return localdriveoperators(device, Basis.Occupation, t)
end

function localdriveoperators(
    device::LocallyDrivenDevice,
    basis::Type{<:Bases.LocalBasis},
    t::Real,
)
    ā = Devices.localalgebra(device, basis)

    # SINGLE OPERATOR TO FETCH THE CORRECT TYPING
    F = ndrives(device) > 0 ? eltype(Devices.driveoperator(device, ā, 1, t)) : eltype(ā)

    v̄ = Tuple(zeros(F, size(ā[q])) for q in 1:nqubits(device))
    for i in 1:ndrives(device)
        q = drivequbit(device, i)
        v̄[q] .+= Devices.driveoperator(device, ā, i, t)
    end
    return v̄
end

function localdrivepropagators(device::LocallyDrivenDevice, τ::Real, t::Real)
    return localdrivepropagators(device, Basis.Occupation, τ, t)
end

function localdrivepropagators(
    device::LocallyDrivenDevice,
    basis::Type{<:Bases.LocalBasis},
    τ::Real,
    t::Real,
)
    v̄ = localdriveoperators(device, basis, t)
    ū = []
    for v in v̄
        v = convert(Array{LinearAlgebraTools.cis_type(v)}, v)
        u = LinearAlgebraTools.cis!(v, -τ)
        push!(ū, u)
    end
    return Tuple(ū)
end

function Devices.propagator(::Type{Operators.Drive},
    device::LocallyDrivenDevice,
    basis::Type{<:Bases.LocalBasis},
    τ::Real,
    t::Real,
)
    ū = localdrivepropagators(device, basis, τ, t)
    return LinearAlgebraTools.kron(ū)
end

function Devices.propagator(::Type{Operators.Channel},
    device::LocallyDrivenDevice,
    basis::Type{<:Bases.LocalBasis},
    τ::Real,
    i::Int,
    t::Real,
)
    ā = Devices.localalgebra(device, basis)
    v = Devices.driveoperator(device, ā, i, t)
    v = convert(Array{LinearAlgebraTools.cis_type(v)}, v)
    u = LinearAlgebraTools.cis!(v, -τ)
    q = drivequbit(device, i)
    return globalize(device, u, q)
end

function Devices.propagate!(::Type{Operators.Drive},
    device::LocallyDrivenDevice,
    basis::Type{<:Bases.LocalBasis},
    τ::Real,
    ψ::AbstractVecOrMat{<:Complex{<:AbstractFloat}},
    t::Real,
)
    ū = localdrivepropagators(device, basis, τ, t)
    return LinearAlgebraTools.rotate!(ū, ψ)
end

function Devices.propagate!(::Type{Operators.Channel},
    device::LocallyDrivenDevice,
    basis::Type{<:Bases.LocalBasis},
    τ::Real,
    ψ::AbstractVecOrMat{<:Complex{<:AbstractFloat}},
    i::Int,
    t::Real,
)
    ā = Devices.localalgebra(device, basis)
    v = Devices.driveoperator(device, ā, i, t)
    v = convert(Array{LinearAlgebraTools.cis_type(v)}, v)
    u = LinearAlgebraTools.cis!(v, -τ)
    q = drivequbit(device, i)
    ops = [p == q ? u : one(u) for p in 1:nqubits(device)]
    return LinearAlgebraTools.rotate!(ops, ψ)
end

# LOCALIZING GRADIENT OPERATORS

# # TODO: Uncomment when we have localized versions of braket and expectation.
# function Devices.braket(::Type{Operators.Gradient},
#     device::LocallyDrivenDevice,
#     basis::Type{<:Bases.LocalBasis},
#     ψ1::AbstractVector,
#     ψ2::AbstractVector,
#     j::Int,
#     t::Real,
# )
#     ā = Devices.localalgebra(device, basis)
#     A = Devices.gradeoperator(device, ā, j, t)
#     q = gradequbit(device, j)
#     ops = [p == q ? A : one(A) for p in 1:nqubits(device)]
#     return LinearAlgebraTools.braket(ψ1, ops, ψ2)
# end