```@meta
CurrentModule = CtrlVQE
```

# Basic Implementations

```@contents
Pages = ["basics.md"]
Depth = 2:5
```

## Integrations

```@autodocs
Modules = [
    TrapezoidalIntegrations,
]
```

## Signals

```@autodocs
Modules = [
    ConstantSignals,
    CompositeSignals,
    WindowedSignals
]
```

## Devices

```@autodocs
Modules = [
    RealWindowedResonantTransmonDevices,
    WindowedResonantTransmonDevices,
    TransmonDevices,
]
```

## Evolutions

```@autodocs
Modules = [
    RotatingFrameEvolutions,
    QubitFrameEvolutions,
]
```

## Cost Functions

```@autodocs
Modules = [
    DenseLeakageFunctions,
    DenseObservableFunctions,
    CompositeCostFunctions,
    WindowedResonantPenalties,
    SignalStrengthPenalties,
    AmplitudePenalties,
    DetuningPenalties,
]
```