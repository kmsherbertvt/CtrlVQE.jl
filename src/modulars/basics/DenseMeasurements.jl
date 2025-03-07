module DenseMeasurements
    import ..ModularFramework as Modular
    import ..ModularFramework: AlgebraType, MeasurementType

    import CtrlVQE.LinearAlgebraTools as LAT
    import CtrlVQE: Bases, Operators
    import CtrlVQE: Integrations, Devices, CostFunctions

    import TemporaryArrays: @temparray

    """
        DenseMeasurement(algebra, observable, basis, frame)

    Reperesents a bare measurement of a matrix observable in a given basis and frame,
        without any logical projection prior to frame rotation.

    # Parameters
    - `observable`: the dense matrix observable.
    - `basis`: the `BasisType` identifying the basis `observable` is written in.
    - `frame`: the `OperatorType` identifying the frame where measurements are conducted.

    Specifically, the operator identified by `frame`
        is the one we rotate by for duration `t`
        to move from the "lab" frame to the interaction picture.
    Use `IDENTITY` for the lab frame itslef, and `STATIC` for the dressed frame.

    ```jldoctests
    julia> using CtrlVQE.ModularFramework;

    julia> observable = LAT.basisvectors(4)
    4×4 Matrix{Bool}:
     1  0  0  0
     0  1  0  0
     0  0  1  0
     0  0  0  1

    julia> measurement = DenseMeasurement(observable, Bases.BARE, Operators.STATIC);

    julia> device = Devices.Prototype(LocalDevice{Float64}, 2);

    julia> validate(measurement; device=device);

    julia> Ō = observables(measurement, device, Bases.BARE, 10.0);

    julia> size(Ō)
    (4, 4, 1)

    julia> observable ≈ reshape(Ō, (4,4))
    true

    ```

    """
    struct DenseMeasurement{
        F,  # MAY BE ANY NUMBER TYPE, REAL OR COMPLEX
        B <: Bases.BasisType,
        O <: Operators.StaticOperator,
    } <: MeasurementType
        observable::Matrix{F}
        basis::B
        frame::O
    end

    function Modular.measure(
        measurement::DenseMeasurement,
        device::Devices.DeviceType,
        basis::Bases.BasisType,
        ψ::AbstractVector,
        t::Real,
    )
        # COPY THE STATE SO WE CAN ROTATE IT
        ψ_ = @temparray(eltype(ψ), size(ψ), :measure)
        ψ_ .= ψ

        # ROTATE THE STATE INTO THE MEASUREMENT BASIS
        U = Devices.basisrotation(measurement.basis, basis, device)
        LAT.rotate!(U, ψ_)

        # APPLY THE FRAME ROTATION
        Devices.evolve!(measurement.frame, device, measurement.basis, -t, ψ_)

        # TAKE THE EXPECTATION VALUE
        return real(LAT.expectation(measurement.observable, ψ_))
    end

    CostFunctions.nobservables(::Type{<:DenseMeasurement}) = 1

    function Modular.observables(
        measurement::DenseMeasurement,
        device::Devices.DeviceType,
        basis::Bases.BasisType,
        t::Real;
        result=nothing
    )
        N = Devices.nstates(device)
        isnothing(result) && (result = Array{Complex{eltype(device)}}(undef, N, N, 1))
        O = @view(result[:,:,1])

        # REPRESENT O IN THE MEASUREMENT BASIS
        O .= measurement.observable

        # ROTATE INTO THE REQUESTED BASIS
        U = Devices.basisrotation(basis, measurement.basis, device)
        LAT.rotate!(U, O)

        # APPLY THE FRAME ROTATION
        Devices.evolve!(measurement.frame, device, basis, t, O)

        return result
    end

    function Devices.gradient(
        measurement::DenseMeasurement,
        device::Devices.DeviceType,
        grid::Integrations.IntegrationType,
        ϕ::AbstractArray,
        ψ::AbstractVector;
        result=nothing
    )
        return Devices.gradient(device, grid, @view(ϕ[:,:,1]); result=result)
    end
end