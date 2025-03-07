import CtrlVQE: CostFunctions

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
- `measure(::M, device, basis, ψ, t)`: measures a state.
- `observables(::M, device, basis, t; result)`:
    constructs the observables in the given basis.
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

CostFunctions.nobservables(::M) where {M<:MeasurementType} = CostFunctions.nobservables(M)