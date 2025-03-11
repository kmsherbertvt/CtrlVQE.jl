import CtrlVQE: Bases, Operators
import CtrlVQE: Devices, CostFunctions

import TemporaryArrays: @temparray

"""
    ReferenceType{B}

A protocol to prepare an initial statevector.

# Type Parameters
- `B<:Bases.BasisType`: the basis that a reference was defined in

# Implementation

Subtypes `R` must implement the following methods:
- `prepare(::R, device; result)`: constructs the initial statevector in the initbasis.

While `initbasis` has a default implementation for singleton basis types,
    this method will need to be overridden
    if your reference is ever needed with more complex types.

"""
abstract type ReferenceType{B<:Bases.BasisType}  end

"""
    initbasis(reference)

Fetch the basis that this reference was defined in.

"""
initbasis(::ReferenceType{B}) where {B} = B()

"""
    prepare(reference, device; result=nothing)
    prepare(reference, device, basis; result=nothing)

Prepare the state and represent it as a statevector in the given basis.

# Parameters
- `reference::ReferenceType`: the state preparation protocol.
- `device::DeviceType`: the device for which the state is being prepared.
- `basis::BasisType`: the basis to represent the reference state in
    (defaults to `initbasis` when omitted).

The returned result is also written to the `result` kwarg when provided.

"""
function prepare end

function prepare(
    reference::ReferenceType,
    device::Devices.DeviceType,
    basis::Bases.BasisType;
    result=nothing,
)
    # REPRESENT ψ IN ITS NATIVE BASIS
    result = prepare(reference, device; result=result)

    # ROTATE INTO THE REQUESTED BASIS
    U = Devices.basisrotation(basis, initbasis(reference), device)
    LAT.rotate!(U, result)

    return result
end

##########################################################################################

"""
    MeasurementType{B,O}

A protocol to measure scalars from a statevector.

# Type Parameters
- `B<:Bases.BasisType`: the basis that a measurement takes place in.
- `O<:Operators.OperatorType`: the frame rotation that a measurement takes place in.
    Specifically, identifies the static operator in the interacture picture,
    so `STATIC` identifies the dressed frame and `IDENTITY` identifies the lab frame.

# Implementation

TODO: Rather different now. Be sure to mention overriding initbasis/initframe if needed.

Subtypes `M` must implement the following methods:
- `measure(::M, device, ψ)`: measures a state given in the initbasis.
- `observables(::M, device; result)`: constructs the observables in the initbasis.
- `CostFunctions.nobservables(::Type{M})`:
    the number of distinct observables involved in a measurement.
- `Devices.gradient(::M, device, grid, ϕ, ψ; result)`:
    calculates the gradient of device parameters, given gradient signals.

In `Devices.gradient`, `ϕ[:,j,k]` contains the jth gradient signal
    ``ϕ_j(t)`` evaluated at each point in `grid` for observable `k`,
    while `ψ` contains the wavefunction itself,
    evolved to the end of the grid
    and rotated into the measurement basis.
For the most part, implementing types will simply delegate to `device`,
    whose gradient method expects the 2d array `ϕ[:,k]`.
When `M` consists of more than one observable,
    it will need to decide how to combine the resulting gradients into one.

While `initbasis` and `initframe` have default implementations
    for singleton basis and operator types,
    these methods will need to be overridden
    if your measurement is ever needed with more complex types.

"""
abstract type MeasurementType{
    B <: Bases.BasisType,
    O <: Operators.StaticOperator,
} end

"""
    initbasis(measurement)

Fetch the basis that this measurement takes place in.

"""
initbasis(::MeasurementType{B,O}) where {B,O} = B()

"""
    initframe(measurement)

Fetch the frame that this measurement takes place in.

Specifically, this function fetches the static operator in the interacture picture,
    so `STATIC` identifies the dressed frame and `IDENTITY` identifies the lab frame.

"""
initframe(::MeasurementType{B,O}) where {B,O} = O()

CostFunctions.nobservables(::M) where {M<:MeasurementType} = CostFunctions.nobservables(M)

"""
    measure(measurement, device, ψ)
    measure(measurement, device, ψ, t)
    measure(measurement, device, basis, ψ)
    measure(measurement, device, basis, ψ, t)

Measure the state ψ, provided in the given basis, at time t.

# Parameters
- `measurement::MeasurementType`: the measurement protocol.
- `device::DeviceType`: the device being measured.
- `basis::BasisType`: the basis that `ψ` is represented in.
    Defaults to `initbasis` when omitted.
- `ψ::AbstractVector`: the state to measure.
- `t::Real`: the time at which `ψ` is being measured,
    i.e. the time over which to evolve the frame operator.
    When `t` is omitted, the frame rotation is skipped.

"""
function measure end

function measure(
    measurement::MeasurementType,
    device::Devices.DeviceType,
    ψ::AbstractVector,
    t::Real,
)
    # COPY THE STATE SO WE CAN ROTATE IT
    ψ_ = @temparray(eltype(ψ), size(ψ), :measure)
    ψ_ .= ψ

    # APPLY THE FRAME ROTATION
    Devices.evolve!(initframe(measurement), device, initbasis(measurement), -t, ψ_)

    return measure(measurement, device, ψ_)
end

function measure(
    measurement::MeasurementType,
    device::Devices.DeviceType,
    basis::Bases.BasisType,
    ψ::AbstractVector,
)
    # COPY THE STATE SO WE CAN ROTATE IT
    ψ_ = @temparray(eltype(ψ), size(ψ), :measure)
    ψ_ .= ψ

    # ROTATE THE STATE INTO THE MEASUREMENT BASIS
    U = Devices.basisrotation(initbasis(measurement), basis, device)
    LAT.rotate!(U, ψ_)

    return measure(measurement, device, ψ_)
end

function measure(
    measurement::MeasurementType,
    device::Devices.DeviceType,
    basis::Bases.BasisType,
    ψ::AbstractVector,
    t::Real,
)
    # COPY THE STATE SO WE CAN ROTATE IT
    ψ_ = @temparray(eltype(ψ), size(ψ), :measure)
    ψ_ .= ψ

    # ROTATE THE STATE INTO THE MEASUREMENT BASIS
    U = Devices.basisrotation(initbasis(measurement), basis, device)
    LAT.rotate!(U, ψ_)

    # APPLY THE FRAME ROTATION
    Devices.evolve!(initframe(measurement), device, initbasis(measurement), -t, ψ_)

    return measure(measurement, device, ψ_)
end

"""
    observables(measurement, device; result=nothing)
    observables(measurement, device, t; result=nothing)
    observables(measurement, device, basis; result=nothing)
    observables(measurement, device, basis, t; result=nothing)

Constructs the Hermitian observables involved in this measurement.

# Parameters
- `measurement::MeasurementType`: the measurement protocol.
- `device::DeviceType`: the device being measured.
- `basis::BasisType`: the basis the observables are represented in.
    Defaults to `initbasis` when omitted.
- `t::Real`: the time at which the measurement takes place,
    i.e. the time over which to evolve the frame operator.
    When `t` is omitted, the frame rotation is skipped.

The returned result is also written to the `result` kwarg when provided.

# Returns
A 3darray indexed such that `result[:,:,k]` is the kth matrix,
    the same format required by the parameter Ō in `Evolutions.gradientsignal`.
The size of the third dimension must be equal to `nobservables(measurement)`.

"""
function observables end

function observables(
    measurement::MeasurementType,
    device::Devices.DeviceType,
    t::Real;
    result=nothing
)
    # REPRESENT O IN THE MEASUREMENT BASIS
    result = observables(measurement, device; result=result)

    frame = initframe(measurement)
    for k in 1:CostFunctions.nobservables(measurement)
        # APPLY THE FRAME ROTATION
        Devices.evolve!(frame, device, initbasis(measurement), t, @view(result[:,:,k]))
    end

    return result
end

function observables(
    measurement::MeasurementType,
    device::Devices.DeviceType,
    basis::Bases.BasisType;
    result=nothing
)
    # REPRESENT O IN THE MEASUREMENT BASIS
    result = observables(measurement, device; result=result)

    U = Devices.basisrotation(basis, initbasis(measurement), device)
    for k in 1:CostFunctions.nobservables(measurement)
        # ROTATE INTO THE REQUESTED BASIS
        LAT.rotate!(U, @view(result[:,:,k]))
    end

    return result
end

function observables(
    measurement::MeasurementType,
    device::Devices.DeviceType,
    basis::Bases.BasisType,
    t::Real;
    result=nothing
)
    # REPRESENT O IN THE MEASUREMENT BASIS
    result = observables(measurement, device; result=result)

    frame = initframe(measurement)
    U = Devices.basisrotation(basis, initbasis(measurement), device)
    for k in 1:CostFunctions.nobservables(measurement)
        # APPLY THE FRAME ROTATION
        Devices.evolve!(frame, device, initbasis(measurement), t, @view(result[:,:,k]))
        # ROTATE INTO THE REQUESTED BASIS
        LAT.rotate!(U, @view(result[:,:,k]))
    end

    return result
end