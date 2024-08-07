"""
    LocallyDrivenDevice

Super-type for device objects whose drive channels act locally on individual qubits.

Inherit from this type if your `driveoperator` and `gradeoperator` methods
    depend only on a single qubit, i.e. `ā[:,:,:,q]`.
This enables more efficient propagation methods which exploit a tensor product structure.

# Implementation

Any concrete sub-type `D` must implement
    *everything* required in the `DeviceType` interface,
    so consult the documentation for `DeviceType` carefully.

In addition, the following methods must be implemented:
- `drivequbit(::D, i::Int)`: index of the qubit on which channel `i` is applied.
- `gradequbit(::D, j::Int)`: index of the qubit associated with the jth gradient operator.

It's usually trivial to infer the channel index i associated with each gradient operator,
    in which case `gradequbit(device, j) = drivequbit(device, i)`,
    but this is left as an implementation detail.

"""
abstract type LocallyDrivenDevice{F} <: DeviceType{F} end

"""
    drivequbit(device, i::Int)

Index of the qubit on which channel `i` is applied.

"""
function drivequbit(::LocallyDrivenDevice, i::Int)
    error("Not Implemented")
    return 0
end

"""
    gradequbit(device, j::Int)

Index of the qubit associated with the jth gradient operator.

"""
function gradequbit(::LocallyDrivenDevice, j::Int)
    error("Not Implemented")
    return 0
end