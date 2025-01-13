device = Modular.LocalDevice(Float, A(), drift, [(
    q = ((i-1) >> 1) + 1;
    template = isodd(i) ? template_α : template_β;
    Modular.DipoleDrive{A}(q, ω[q], Windowed(template, T, W), template_ν)
) for i in 1:2n], Modular.DISJOINT)

energyfn = Modular.Energy(QUBIT_FRAME, device, grid, reference, measurement)

penaltyfn = Modular.DrivePenalty(device, [CtrlVQE.SignalStrengthPenalty(
    grid, drive.Ω;
    A=ΩMAX-σΩ, σ=σΩ, λ=λΩ,
) for drive in device.drives])

costfn = CtrlVQE.CompositeCostFunction(energyfn, penaltyfn)
