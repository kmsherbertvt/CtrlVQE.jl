module DenseMeasurements
    import CtrlVQE.ModularFramework as Modular
    import CtrlVQE.ModularFramework: AlgebraType, MeasurementType

    import CtrlVQE.LinearAlgebraTools as LAT
    import CtrlVQE: Bases, Operators
    import CtrlVQE: Integrations, Devices, CostFunctions

    import TemporaryArrays: @temparray

    """
        DenseMeasurement(basis, frame, observable)

    Represents a bare measurement of a matrix observable in a given basis and frame,
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

    julia> I = [1 0; 0 1]; X = [0 1; 1 0]; Z = [1 0; 0 -1];

    julia> observable = (0.5 .* kron(X,Z)) .+ (1.0 .* kron(I, Z))
    4×4 Matrix{Float64}:
     1.0   0.0  0.5   0.0
     0.0  -1.0  0.0  -0.5
     0.5   0.0  1.0   0.0
     0.0  -0.5  0.0  -1.0

    julia> measurement = DenseMeasurement(BARE, STATIC, observable);

    julia> device = Prototype(LocalDevice{Float64}; n=2);

    julia> validate(measurement; device=device);

    julia> Ō = observables(measurement, device);

    julia> size(Ō)
    (4, 4, 1)

    julia> observable ≈ reshape(Ō, (4,4))
    true

    ```

    """
    struct DenseMeasurement{F,B,O} <: MeasurementType{B,O}
        observable::Matrix{F}
    end

    function DenseMeasurement(::B, ::O, observable::AbstractMatrix{F}) where {F,B,O}
        return DenseMeasurement{F,B,O}(collect(observable))
    end

    function Modular.measure(
        measurement::DenseMeasurement,
        device::Devices.DeviceType,
        ψ::AbstractVector
    )
        # TAKE THE EXPECTATION VALUE
        return real(LAT.expectation(measurement.observable, ψ))
    end

    CostFunctions.nobservables(::Type{<:DenseMeasurement}) = 1

    function Modular.observables(
        measurement::DenseMeasurement,
        device::Devices.DeviceType;
        result=nothing
    )
        N = Devices.nstates(device)
        isnothing(result) && (result = Array{Complex{eltype(device)}}(undef, N, N, 1))
        O = @view(result[:,:,1])

        # REPRESENT O IN THE MEASUREMENT BASIS
        O .= measurement.observable

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