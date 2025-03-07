module LocalDevices
    import ..ModularFramework as Modular
    import ..ModularFramework: AlgebraType, DriftType, LocalDrive, ParameterMap

    import CtrlVQE
    import CtrlVQE: Parameters
    import CtrlVQE: Devices, Integrations

    import TemporaryArrays: @temparray

    import LinearAlgebra: mul!

    """
        LocalDevice(algebra, drift, drives, pmap, x)

    A totally modular device, with some restrictions (see below).

    # Parameters
    - `algebra::AlgebraType`: the algebra, defining the Hilbert space.
    - `drift::DriftType`: the drift term, describing the static Hamiltonian.
    - `drives::Vector{<:DriveType}` a vector of drive terms,
            describing the time-dependent Hamiltonian.
    - `pmap::ParameterMap`: the parameter map,
        defining the relationship between device and drive parameters.
    - `x::Vector{<:AbstractFloat}`: the initial device parameters

        LocalDevice(F, algebra, drift, drives, pmap)

    The typical constructor for a `LocalDevice`.

    # Parameters
    - `F`: the float type for this device (typically `Float64`).
    - `algebra::AlgebraType`: the algebra, defining the Hilbert space.
    - `drift::DriftType`: the drift term, describing the static Hamiltonian.
    - `drives::Vector{<:DriveType}` a vector of drive terms,
            describing the time-dependent Hamiltonian.
    - `pmap::ParameterMap`: the parameter map,
        defining the relationship between device and drive parameters.

    The device parameters are initialized to be consistent with `drives`
        (though the *degree* of consistency is defined by `pmap`).

    # Restrictions

    - Drift can't be parametric
      - I could program around this,
        but it would add an unnecessary amount of code for a feature I may never use.
      - In principle, another type could relax this restriction if:
        - compatible drift types implement the `Parameters` interface.
        - compatible parameter map types implement suitable versions
            of `map_values` and `map_gradients`
        - the `bind!` method clears relevant methods from the `Memoization.jl` cache.
        - all loops in this file with parameter manipulation include a step for the drift.

    - Only works with LocalDrives
      - This restriction is necessary only due to lack of multiple inheritance in Julia.
        In other words, I *have* to subtype `LocallyDrivenDevice`
            in order to get the fast evolution,
            and that means I *have* to restrict the drives to those compatible with it.
        But *none* of the code here depends on it,
            other than the parts implementing the interface for `LocallyDrivenDevice`.
      - In principle, another type could relax this restriction if:
        - you copy and paste the whole file.
        - switch out `LocallDrivenDevice` for `LocalDevice`
        - switch out `LocalDrive` with `DriveType`.
        - delete the parts with `drivequbit` and `gradequbit`.

    """
    struct LocalDevice{
        F,
        A <: AlgebraType,
        H <: DriftType{A},
        V <: LocalDrive{A},
        P <: ParameterMap,
    } <: Devices.LocallyDrivenDevice{F}
        algebra::A
        drift::H
        drives::Vector{V}
        pmap::P
        x::Vector{F}
    end

    function LocalDevice(
        ::Type{F},
        algebra::A,
        drift::DriftType{A},
        drives::Vector{<:LocalDrive{A}},
        pmap::P,
    ) where {F<:AbstractFloat,A<:AlgebraType,P}
        device = LocalDevice(algebra, drift, drives, pmap, zeros(F,0))
        Modular.sync!(pmap, device)
        return device
    end

    Modular.algebratype(::LocalDevice) = Modular.algebratype(device.algebra)

    ######################################################################################
    #= `Parameters` interface =#

    Parameters.count(device::LocalDevice) = length(device.x)
    Parameters.values(device::LocalDevice) = device.x

    function Parameters.names(device::LocalDevice)
        # DELEGATE TO PARAMETER MAP
        return Parameters.names(device.pmap, device)
    end

    function Parameters.bind!(device::LocalDevice, x::AbstractVector)
        device.x .= x

        # SYNC DRIVES
        L = maximum(Parameters.count, device.drives)
        y_ = @temparray(eltype(device), (L,), :bind)
        for (i, drive) in enumerate(device.drives)
            y = @view(y_[1:Parameters.count(drive)])
            Modular.map_values(device.pmap, device, i; result=y)
            Parameters.bind!(drive, y)
        end
        return device
    end

    ######################################################################################
    # Counting methods

    Devices.nqubits(device::LocalDevice) = Devices.nqubits(device.algebra)
    Devices.nlevels(device::LocalDevice) = Devices.nlevels(device.algebra)
    Devices.ndrives(device::LocalDevice) = length(device.drives)

    function Devices.ngrades(device::LocalDevice)
        # DELEGATE TO DRIVE
        return Devices.ndrives(device) * Devices.ngrades(eltype(device.drives))
    end

    Devices.noperators(device::LocalDevice) = Devices.noperators(device.algebra)

    ######################################################################################
    # Algebra methods

    function Devices.localalgebra(device::LocalDevice; result=nothing)
        isnothing(result) && return Devices._localalgebra(device)
        return Devices.localalgebra(device.algebra; result=result)
    end

    ######################################################################################
    # Operator methods

    function Devices.qubithamiltonian(
        device::LocalDevice,
        ā,
        q::Int;
        result=nothing,
    )
        C = Complex{eltype(device)}
        isnothing(result) && (result = Array{C}(undef, size(ā)[1:2]))
        # DELEGATE TO DRIFT
        return Devices.qubithamiltonian(device.drift, ā, q; result=result)
    end

    function Devices.staticcoupling(
        device::LocalDevice,
        ā;
        result=nothing,
    )
        C = Complex{eltype(device)}
        isnothing(result) && (result = Array{C}(undef, size(ā)[1:2]))
        # DELEGATE TO DRIFT
        return Devices.staticcoupling(device.drift, ā; result=result)
    end

    function Devices.driveoperator(
        device::LocalDevice,
        ā,
        i::Int,
        t::Real;
        result=nothing,
    )
        C = Complex{eltype(device)}
        isnothing(result) && (result = Array{C}(undef, size(ā)[1:2]))
        # DELEGATE TO DRIVE
        return Devices.driveoperator(device.drives[i], ā, t; result=result)
    end

    function Devices.gradeoperator(
        device::LocalDevice,
        ā,
        j::Int,
        t::Real;
        result=nothing,
    )
        C = Complex{eltype(device)}
        isnothing(result) && (result = Array{C}(undef, size(ā)[1:2]))
        # DELEGATE TO DRIVE
        ng = Devices.ngrades(eltype(device.drives))
        i, j_ = divrem((j-1), ng) .+ 1
        return Devices.gradeoperator(device.drives[i], ā, j_, t; result=result)
    end

    ######################################################################################
    # Gradient methods

    function Devices.gradient(
        device::LocalDevice,
        grid::Integrations.IntegrationType,
        ϕ::AbstractMatrix;
        result=nothing,
    )
        isnothing(result) && (result = Array{eltype(device)}(undef, length(device.x)))
        ng = Devices.ngrades(eltype(device.drives))

        # TEMP ARRAY TO HOLD GRADIENTS FOR EACH CHANNEL (one at a time)
        L = maximum(Parameters.count, device.drives)
        ∂y_ = @temparray(eltype(device), (L,), :gradient, :partial)
        g_ = @temparray(eltype(device), (length(device.x), L), :gradient)

        result .= 0
        for (i, drive) in enumerate(device.drives)
            j0 = ng*(i-1) + 1       # FIRST GRADE INDEX

            # COMPUTE GRADIENT WITH RESPECT TO DRIVE PARAMETERS
            ∂y = @view(∂y_[1:Parameters.count(drive)])
            # DELEGATE TO DRIVE
            Devices.gradient(drive, grid, @view(ϕ[:,j0:i*ng]); result=∂y)

            # CONSTRUCT GRADIENT OF DRIVE PARAMETERS WITH RESPECT TO DEVICE PARAMETERS
            g = @view(g_[:,1:Parameters.count(drive)])
            Modular.map_gradients(device.pmap, device, i; result=g)

            # APPLY CHAIN RULE
            mul!(result, g, ∂y, 1, 1)   # Computes matrix product 1g*1∂y, adds to result.
        end
        return result
    end

    ######################################################################################
    # `LocallyDrivenDevice` interface

    function Devices.drivequbit(device::LocalDevice, i::Int)
        # DELEGATE TO DRIVE
        return Devices.drivequbit(device.drives[i])
    end

    function Devices.gradequbit(device::LocalDevice, j::Int)
        ng = Devices.ngrades(eltype(device.drives))
        i, j_ = divrem((j-1), ng) .+ 1
        return Devices.drivequbit(device.drives[i])
    end

    #= NOTE: Prototyping relies on concrete types,
        so it is handled in `Prototypes.jl`, after the "basics" are defined. =#
end