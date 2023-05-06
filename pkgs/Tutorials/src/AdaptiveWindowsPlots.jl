import Plots

function create_plot_Et()
    global t, E, FCI, FES, REF

    yMAX = max(maximum(E), REF, FES)
    plot = Plots.plot(;
        xlabel= "Time (ns)",
        ylabel= "Energy (Ha)",
        xlims = axislimits(t[1], t[end]),
        ylims = axislimits(FCI, yMAX),
        legend= :topright,
    )

    # PLOT ENERGY, BOTH PROJECTED AND NORMALIZED
    Plots.plot!(plot, t, E, lw=3, label="Energy")
    Plots.plot!(plot, t, E./F, lw=3, label="Normalized")

    # PLOT LEAKAGE ON A SEPARATE AXIS
    # HACK: Fixes the legend.
    Plots.plot!(plot, [0], [2yMAX], lw=3, color=3, label="Leakage")
    Plots.plot!(Plots.twinx(plot),
        t, 1 .- F, color=3, lw=3,
        legend=false, ylabel="Leakage", ylims=axislimits(0,1),
    )

    # TODO: How do we get the line for E_FCI to appear above the leakage?
    #= TODO: The leakage curve is not extending all the way to t=0...
        HINT: I had to plot the hack ABOVE instead of to the LEFT of the plot,
            and that made MOST of the problem vanish. Not sure what it means.
    =#

    # PLOT DASHED LINES AT SPECIFIC ENERGIES
    Plots.hline!(plot, [REF, FCI, FES], color=:black, ls=:dash, label=false)

    return plot
end


function create_plot_Ωt()
    global t, α, β, ΩMAX

    yMAX = ΩMAX / 2π
    plot = Plots.plot(;
        xlabel= "Time (ns)",
        ylabel= "|Amplitude| (GHz)",
        ylims = axislimits(-yMAX, yMAX),
        legend= :topright,
    )

    # HACK: Fix legend.
    Plots.plot!(plot, [0], [2yMAX], lw=3, ls=:solid, color=:black, label="α")
    Plots.plot!(plot, [0], [2yMAX], lw=3, ls=:dot, color=:black, label="β")

    # PLOT AMPLITUDES
    for i in 1:CtrlVQE.ndrives(device)
        Plots.plot!(plot, t, α[:,i]./2π, lw=3, ls=:solid, color=i, label="Drive $i")
        Plots.plot!(plot, t, β[:,i]./2π, lw=3, ls=:dot, color=i, label=false)
    end

    return plot
end

function create_plot_Ωf()
    global t, α̂, β̂, kMAX, Δ

    yMAX = max(maximum(abs.(α̂[2:end])), maximum(abs.(β̂[2:end])))
    plot = Plots.plot(;
        xlabel= "Frequency (GHz)",
        ylabel= "|Amplitude| (GHz)",
        xlims = axislimits(0, f[kMAX]),
        ylims = axislimits(0, yMAX),
        legend= :topright,
    )

    # HACK: Fix legend.
    Plots.plot!(plot, [0], [2yMAX], lw=3, ls=:solid, color=:black, label="α")
    Plots.plot!(plot, [0], [2yMAX], lw=3, ls=:dot, color=:black, label="β")

    # PLOT AMPLITUDES
    k̄ = 2:kMAX  # OMIT ZERO-FREQUENCY MODE
    for i in 1:CtrlVQE.ndrives(device)
        Plots.plot!(plot, f[k̄], abs.(α̂[k̄,i]), lw=3, ls=:solid, color=i, label="Drive $i")
        Plots.plot!(plot, f[k̄], abs.(β̂[k̄,i]), lw=3, ls=:dot, color=i, label=false)
        Plots.vline!(plot, [abs(Δ[i]/2π)], linestyle=:dash, color=i, label=false)
    end

    return plot
end

function create_plot_ϕt()
    global t, ϕα, ϕβ

    yMAX = max(maximum(abs.(ϕα)), maximum(abs.(ϕβ))) / 2π
    plot = Plots.plot(;
        xlabel= "Time (ns)",
        ylabel= "|Gradient Signal| (Ha/GHz)",
        ylims = axislimits(-yMAX, yMAX),
        legend= :topright,
    )

    # HACK: Fix legend.
    Plots.plot!(plot, [0], [2yMAX], lw=3, ls=:solid, color=:black, label="ϕα")
    Plots.plot!(plot, [0], [2yMAX], lw=3, ls=:dot, color=:black, label="ϕβ")

    # PLOT AMPLITUDES
    for i in 1:CtrlVQE.ndrives(device)
        Plots.plot!(plot, t, ϕα[:,i]./2π, lw=3, ls=:solid, color=i, label="Drive $i")
        Plots.plot!(plot, t, ϕβ[:,i]./2π, lw=3, ls=:dot, color=i, label=false)
    end

    return plot
end

function create_plot_ϕf()
    global t, ϕ̂α, ϕ̂β, kMAX, Δ

    yMAX = max(maximum(abs.(ϕ̂α[2:end])), maximum(abs.(ϕ̂β[2:end])))
    plot = Plots.plot(;
        xlabel= "Frequency (GHz)",
        ylabel= "|Gradient Signal| (Ha/GHz)",
        xlims = axislimits(0, f[kMAX]),
        ylims = axislimits(0, yMAX),
        legend= :topright,
    )

    # HACK: Fix legend.
    Plots.plot!(plot, [0], [2yMAX], lw=3, ls=:solid, color=:black, label="ϕ̂α")
    Plots.plot!(plot, [0], [2yMAX], lw=3, ls=:dot, color=:black, label="ϕ̂β")

    # PLOT AMPLITUDES
    k̄ = 2:kMAX  # OMIT ZERO-FREQUENCY MODE
    for i in 1:CtrlVQE.ndrives(device)
        Plots.plot!(plot, f[k̄], abs.(ϕ̂α[k̄,i]), lw=3,ls=:solid,color=i,label="Drive $i")
        Plots.plot!(plot, f[k̄], abs.(ϕ̂β[k̄,i]), lw=3, ls=:dot, color=i, label=false)
        Plots.vline!(plot, [abs(Δ[i]/2π)], linestyle=:dash, color=i, label=false)
    end

    return plot
end

function create_plot_fg()
    global iterations, adaptations
    global trace_fn_energy, trace_gd, trace_gd_energy, trace_gd_penalty

    plot = Plots.plot(;
        xlabel = "Iterations",
        yscale = :log,
        ylims  = [1e-16, 2e0],
        yticks = 10.0 .^ (-16:2:0),
        legend = :bottomleft,
    )

    Plots.plot!(
        plot, iterations, trace_fn_energy .- FCI;
        lw=3, ls=:solid, label="Energy Error",
    )
    Plots.plot!(
        plot, iterations, trace_gd;
        lw=3, ls=:dash, label="Gradient",
    )
    Plots.plot!(
        plot, iterations, trace_gd_energy;
        lw=3, ls=:dot, label="Energy Gradient",
    )
    Plots.plot!(
        plot, iterations, trace_gd_penalty;
        lw=3, ls=:dashdot, label="Penalty Gradient",
    )

    # PLOT DASHED LINES AT CONVERGENCE CRITERIA
    Plots.hline!(plot, [f_tol, g_tol], color=:black, ls=:dash, label=false)

    # PLOT VERTICAL LINES AT EACH ADAPTATION
    Plots.vline!(plot, adaptations, color=:black, ls=:dot, label=false)

    # TODO: Write in number of parameters for each adaptation?

    return plot
end

function create_plot_ct()
    global iterations, trace_f_calls, trace_g_calls

    plot = Plots.plot(;
        xlabel = "Iterations",
        legend = :topleft,
    )

    Plots.plot!(plot, iterations, trace_f_calls, lw=3, label="Function Calls")
    Plots.plot!(plot, iterations, trace_g_calls, lw=3, label="Gradient Calls")

    return plot
end

function axislimits(MIN, MAX)
    pad = (MAX - MIN) * 0.1
    return [MIN-pad, MAX+pad]
end