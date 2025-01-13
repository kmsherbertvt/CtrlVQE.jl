import CtrlVQE: Devices

"""
    algebratype(object)

Fetch the algebra type backing this object.

"""
function algebratype end

##########################################################################################

"""
    AlgebraType{m,n}

Defines a Hilbert space and an operator basis.

Delegated the following methods:
- `Devices.nlevels`
- `Devices.nqubits`
- `Devices.noperators`
- `Devices.localalgebra`

# Type Parameters
- `m`: the number of levels in each qubit
- `n`: the number of qubits

# Implementation

Subtypes `A` must implement the following methods:
- `Devices.noperators(::Type{A})`:
        how many unique operators define an algebra on one qubit.
- `Devices.localalgebra(::A; result)`: a 4d array ā
        where ā[:,:,σ,q] is the σ'th algebraic operator on qubit q, in the bare basis.
    The result is written to `result`, which is mandatory
        (unlike in the interface for device types).

"""
abstract type AlgebraType{m,n} end
algebratype(::A) where {A<:AlgebraType} = A
algebratype(::Type{A}) where {A<:AlgebraType} = A

Devices.noperators(::A) where {A<:AlgebraType} = Devices.noperators(A)

Devices.nlevels(::AlgebraType{m,n}) where {m,n} = m
Devices.nqubits(::AlgebraType{m,n}) where {m,n} = n
Devices.nlevels(::Type{<:AlgebraType{m,n}}) where {m,n} = m
Devices.nqubits(::Type{<:AlgebraType{m,n}}) where {m,n} = n

##########################################################################################

"""
    DriftType{A}

Defines the static Hamiltonian ``\\hat H_0``.

Delegated the following methods:
- `Devices.qubithamiltonian`
- `Devices.staticcoupling`

# Implementation

Subtypes `H` must implement the following methods:
- `Devices.qubithamiltonian(::H, ā, q::Int; result)`:
        the static components of the device Hamiltonian local to qubit q.
- `Devices.staticcoupling(::H, ā; result)`:
        the static components of the device Hamiltonian nonlocal to any one qubit.

The result is written to `result`, which is mandatory
        (unlike in the interface for device types).

The `ā` arg is a 4d array with all algebra operators,
    like the one returned by `Devices.localalgebra`
    (but not necessarily in a local basis).

"""
abstract type DriftType{A<:AlgebraType} end
algebratype(::DriftType{A}) where {A} = A
algebratype(::Type{H}) where {H<:DriftType} = algebratype(H)

##########################################################################################

"""
    DriveType{A}

Defines a drive term ``\\hat V_i``.

Delegated the following methods:
- `Devices.ngrades`
- `Devices.driveoperator`
- `Devices.gradeoperator`
- `Devices.gradient`

# Implementation

Subtypes `V` must implement the `Parameters` interface, and the following methods:
- `ngrades(::Type{<:V})`: the number of distinct gradient operators for this drive type.
- `driveoperator(::V, ā, t::Real; result)`:
        the distinct drive operator for this channel at time `t`.
- `gradeoperator(::V, ā, j::Int, t::Real; result)`:
        the distinct gradient operator indexed by `j` at time `t`.
- `gradient(::V, grid::Integrations.IntegrationType, ϕ; result)`:
        the gradient vector for each variational parameter in the device.

The result is written to `result`, which is mandatory
        (unlike in the interface for device types).

Note that the drive index `i` is omitted from the interface for `driveoperator`,
    and the grade index `j` is with respect to just this channel.
That means `j` will only ever take values between 1 and `ngrades(V)`.
This is true of both the `j` in the signature of `gradeoperator`
    and the column indices of `ϕ` in the signature of `gradient`.

"""
abstract type DriveType{A<:AlgebraType} end
algebratype(::DriveType{A}) where {A} = A
algebratype(::Type{V}) where {V<:DriveType} = algebratype(V)

Devices.ngrades(drive::V) where {V<:DriveType} = Devices.ngrades(V)

###########################################

"""
    LocalDrive{F,A}

Extension of `DriveType` for ``\\hat V_i`` which act on a single qubit.

Delegated `drivequbit`.

# Implementation

Subtypes `D` must implement the `DriveType` interface, and the following method:
- `drivequbit(::D)`: index of the qubit on which this drive is applied.

Note that the drive index `i` is omitted from the interface for `drivequbit`.

"""
abstract type LocalDrive{A<:AlgebraType} end

##########################################################################################

"""
    ParameterMap

Enumerates the protocol for mapping device parameters to each drive.

Delegated `Parameters.names`.

# Implementation

Subtypes `P` must implement the following methods:
- `Parameters.names(::P, device)`: constructs a list of human-readable names
        for each parameter in the `device`,
        which is a device implementing the `Parameters` interface.
    Certain implementations of `ParameterMap` may have additional requirements.
- `sync!(::P, device)`: mutate the internal parameters of a device
        to match those of its drives.
- `map_values(::P, device, i::Int; result)`:
        computes drive parameters from device parameters.
- `map_gradients(::P, device, i::Int; result)`:
        computes drive gradients with respect to device parameters.

"""
abstract type ParameterMap end

"""
    sync!(device)
    sync!(pmap::ParameterMap, device)

Mutate the internal parameters of a device to match those of its drives.

The one-parameter signature requires `device` to have a property `pmap`,
    which is the parameter map that will be used for dispatch.
Both signatures require `device` to have the property `x`
    which is a vector of all the parameters of `device`.
Certain implementations of `ParameterMap` may have additional requirements.

This function may resize `x`.

"""
function sync! end

sync!(device) = sync!(device.pmap, device)


"""
    map_values(pmap::ParameterMap, device, i::Int; result)

Compute the parameters for a drive term, as a function of all device parameters.


# Parameters
- `pmap`: the `ParameterMap` defining the family of functions to map parameters.
- `device`: the device, giving the device parameters (via `Parameters.values`).
        Certain implementations of `ParameterMap` may have additional requirements.
- `i`: identifies which function in the family (i.e. indexes the drive).

The returned result is also written to the `result` kwarg, which is mandatory.

"""
function map_values end

"""
    map_gradients!(device::ModularDevice, i::Int; result)

Compute gradients for parameters in a drive term, with respect to each device parameter.

# Parameters
- `pmap`: the `ParameterMap` defining the family of functions to map parameters.
- `device`: the device, giving the device parameters (via `Parameters.values`).
        Certain implementations of `ParameterMap` may have additional requirements.
- `i`: identifies which function in the family (i.e. indexes the drive).

# Returns
A matrix `g`, such that `g[k,j]` is ``∂_{x_k} y_j``.

The returned result is also written to the `result` kwarg, which is mandatory.

"""
function map_gradients end

##########################################################################################

"""
    ReferenceType

A protocol to prepare an initial statevector.

# Implementation

Subtypes `R` must implement the following methods:
- `prepare(::R, device, basis; result)`:
    constructs the initial statevector in the given basis.

"""
abstract type ReferenceType end

"""
    prepare(preparer, device, basis; result=nothing)

Prepare the state and represent it as a statevector in the given basis.

# Parameters
- `preparer::ReferenceType`: the state preparation protocol.
- `device::DeviceType`: the device for which the state is being prepared.
- `basis::BasisType`: the basis to represent the reference state in.

The returned result is also written to the `result` kwarg when provided.

"""
function prepare end

##########################################################################################

"""
    MeasurementType

A protocol to measure scalars from a statevector.

# Implementation

Subtypes `M` must implement the following methods:
- `measure(::M, device, basis, ψ, t)`: constructs the initial statevector.
- `nobservables(::Type{M})`: the number of distinct observables involved in a measurement.
- `observables(::M, device, basis, t; result)`:
    constructs the observables in the given basis.
- `Devices.gradient(::M, device, grid, ϕ; result)`:
    calculates the gradient of device parameters, given gradient signals.

In `Devices.gradient`, `ϕ[:,j,k]` contains the jth gradient signal
    ``ϕ_j(t)`` evaluated at each point in `grid` for observable `k`.
For the most part, implementing types will simply delegate to `device`,
    whose gradient method expects the 2d array `ϕ[:,:,k]`.
When `M` consists of more than one observable,
    it will need to decide how to combine the resulting gradients into one.

"""
abstract type MeasurementType end

"""
    measure(measurer, device, basis, ψ, t)

Measure the state ψ, provided in the given basis, at time t.

# Parameters
- `measurer::MeasurementType`: the measurement protocol.
- `device::DeviceType`: the device being measured.
- `basis::BasisType`: the basis that `ψ` is represented in.
- `ψ::AbstractVector`: the state to measure.
- `t::Real`: the time at which `ψ` is being measured.

"""
function measure end

"""
    nobservables(measurer)

Identify the number of Hermitian observables needed for this measurement.

For example, to measure the normalized energy,
    separate observables are needed for both energy and normalization,
    and the results are combined in a non-linear way to produce the final outcome.

"""
function nobservables end
nobservables(::M) where {M<:MeasurementType} = nobservables(M)

"""
    observables(measurer, device, basis, t; result=nothing)

Constructs the Hermitian observables involved in this measurement.

# Parameters
- `measurer::MeasurementType`: the measurement protocol.
- `device::DeviceType`: the device being measured.
- `basis::BasisType`: the basis the observables are represented in.
- `t::Real`: the time at which the measurement takes place.

The returned result is also written to the `result` kwarg when provided.

# Returns
A 3darray indexed such that `result[:,:,k]` is the kth matrix,
    the same format required by the parameter Ō in `Evolutions.gradientsignal`.
The size of the third dimension must be equal to `nobservables(measurer)`.

"""
function observables end