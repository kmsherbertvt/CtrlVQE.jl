import CtrlVQE

import Random: seed!
import Serialization: serialize, deserialize
import LinearAlgebra: eigen, Hermitian

import NPZ: npzread
import Optim, LineSearches
import FFTW: rfft, rfftfreq

import Plots

const DEFAULTS = Dict{Symbol,Any}(
    :matrixfile => "./pkgs/Tutorials/matrix/H2_sto-3g_singlet_1.5_P-m.npy",
    :T => 5.0, # ns
    :r => 200,
    :m => 2,

    :seed => 9999,
    :init_Ω => 0.0,
    :init_Δ => 0.0, # GHz

    :ΩMAX => 2π * 0.02, # GHz
    :λΩ => 1.0,
    :σΩ => 2π * 0.01, # GHz

    :ΔMAX => 2π * 1.00, # GHz
    :λΔ => 1.0,
    :σΔ => 2π * 0.01, # GHz

    :f_tol => 0,
    :g_tol => 1e-6,
    :maxiter => 100,           # Maximum number of BFGS steps before adaptation.
    :updateiter => 100,
    :fnRATIO => 3,

    :maxadapt => 10,           # Number of adaptations attempted before giving up.
    :ϕtol => 1e-6,              # Neglect switches oscillating within ϕtol.
    :τtol => 0.5,               # Neglect switches happening within space τtol⋅(T/r).
    :Ωtol => 1e-3,              # Merge switches when they optimize to within Ωtol.

    :fFACTOR => 0.01,
)


function run(; subdir="./pkgs/Tutorials/dat/adaptivewindows", loaddir=false, resume=true)
    loaddir == true && return run(; subdir=subdir, loaddir=subdir, resume=resume)

    isdir(subdir) && !(subdir == loaddir) && error("""
        Directory $subdir already exists. Do one of:
        1. Rename (or delete) the existing directory.
        2. Call `run(; subdir="<path>")` with a different directory.
        3. Call `run(; loaddir=true)` to resume the existing experiment.
    """)
    !(subdir == loaddir) && mkpath(subdir)

    if loaddir isa String
        load(loaddir; resume=resume)
    else
        initialize()
    end

    run_optimization(subdir)
    update(subdir)

    while run_adaptation(subdir)
        run_optimization(subdir)
        update(subdir)
    end
end


function unpack_settings(settings::Dict{Symbol,<:Any})
    for (field, value) in settings
        @eval global $field = $value
    end
end
unpack_settings(DEFAULTS)

modify_settings(; kwargs...) = unpack_settings(Dict(kwargs))

function pack_settings()
    settings = Dict{Symbol,Any}()
    for field in keys(DEFAULTS)
        @eval global $field
        settings[field] = @eval $field
    end
    return settings
end


function load(subdir; resume=false)
    load_settings(subdir)
    initialize()
    load_x(subdir)
    initialize_pulses()     # RE-INITIALIZE PULSES WITH THE NEW STARTTIMES
    update_device()         # UPDATE DEVICE WITH THE NEW PULSES AND PARAMETERS

    resume && load_trace(subdir)
end

load_settings(subdir) = unpack_settings(deserialize("$subdir/settings"))

function load_x(subdir)
    global x = deserialize("$subdir/x")
    global starttimes = deserialize("$subdir/starttimes")
end

function load_trace(subdir)
    # LOAD COUNTER ARRAYS
    global iterations = deserialize("$subdir/iterations")
    global trace_f_calls = deserialize("$subdir/trace_f_calls")
    global trace_g_calls = deserialize("$subdir/trace_g_calls")

    # SET COUNTER OFFSETS
    global offset_iterations = iterations[end]
    global offset_f_calls = trace_f_calls[end]
    global offset_g_calls = trace_g_calls[end]

    # LOAD FUNCTION VALUE AND GRADIENT NORM ARRAYS
    global trace_fn = deserialize("$subdir/trace_fn")
    global trace_fn_energy = deserialize("$subdir/trace_fn_energy")
    global trace_gd = deserialize("$subdir/trace_gd")
    global trace_gd_energy = deserialize("$subdir/trace_gd_energy")
    global trace_gd_penalty = deserialize("$subdir/trace_gd_penalty")

    # LOAD
    global adaptations = deserialize("$subdir/adaptations")
end

function save(subdir)
    if !isdir(subdir); mkpath(subdir); end
    save_settings(subdir)
    save_x(subdir)
    save_trace(subdir)
end

save_settings(subdir) = serialize("$subdir/settings", pack_settings())
save_x(subdir) = (serialize("$subdir/x", x); serialize("$subdir/starttimes", starttimes))

function save_trace(subdir)
    # SAVE COUNTER ARRAYS
    serialize("$subdir/iterations", iterations)
    serialize("$subdir/trace_f_calls", trace_f_calls)
    serialize("$subdir/trace_g_calls", trace_g_calls)

    serialize("$subdir/adaptations", adaptations)

    # SAVE FUNCTION VALUE AND GRADIENT NORM ARRAYS
    serialize("$subdir/trace_fn", trace_fn)
    serialize("$subdir/trace_fn_energy", trace_fn_energy)
    serialize("$subdir/trace_gd", trace_gd)
    serialize("$subdir/trace_gd_energy", trace_gd_energy)
    serialize("$subdir/trace_gd_penalty", trace_gd_penalty)
end



function initialize()
    initialize_matrix()
    initialize_starttimes()
    initialize_pulses()
    initialize_device()
    initialize_algorithm()
    initialize_timegrid()
    initialize_parameters()
    initialize_transmonspace()
    initialize_energyfn()
    initialize_normfn()
    initialize_penaltyfn()
    initialize_lossfn()
    initialize_trace()
    initialize_arrays()
end

function initialize_matrix()
    global H = npzread(matrixfile)
    global n = CtrlVQE.QubitOperators.nqubits(H)
    global ψ_REF = CtrlVQE.QubitOperators.reference(H)
    global REF = real(ψ_REF' * H * ψ_REF)

    global Λ, U = eigen(Hermitian(H))
    global FCI, FES = Λ[1:2]
    global ψ_FCI = U[:,1]
end

function initialize_starttimes()
    global starttimes = [[0.0] for q in 1:n]
end

function initialize_pulses()
    global pulses = [CtrlVQE.Signals.Windowed(
        [CtrlVQE.Signals.ComplexConstant(0.0, 0.0) for t in starttimes[q]],
        starttimes[q],
    ) for q in 1:n]
end

function initialize_device()
    global device = CtrlVQE.SystematicTransmonDevice(m,n,pulses)
end

function initialize_algorithm()
    global algorithm = CtrlVQE.Rotate(r)
end

function initialize_timegrid()
    global τ, dt, t = CtrlVQE.trapezoidaltimegrid(T, r)
    global f = rfftfreq(r, 1/τ)
end

function initialize_parameters()
    seed!(seed)
    global xi = CtrlVQE.Parameters.values(device)

    L = length(xi)
    global Ω = 1:L-n
    global ν = 1+L-n:L

    xi[Ω] .+= (2 .* rand(L-n) .- 1) .* init_Ω
    xi[ν] .+= (2 .* rand(n) .- 1) .* init_Δ

    global x = copy(xi)
end

function initialize_transmonspace()
    global O0 = CtrlVQE.QubitOperators.project(H, device)
    global ψ0 = CtrlVQE.QubitOperators.project(ψ_REF, device)
end

function initialize_energyfn()
    global fn_energy, gd_energy = CtrlVQE.ProjectedEnergy.functions(
        O0, ψ0, T, device, r;
        frame=CtrlVQE.STATIC,
    )
end

function initialize_normfn()
    global fn_norm, gd_norm = CtrlVQE.Normalization.functions(ψ0, T, device, r)
end

function initialize_penaltyfn()
    λ  = zeros(length(x));  λ[Ω] .=    λΩ;  λ[ν] .= λΔ
    xL = zeros(length(x)); xL[Ω] .= -ΩMAX; xL[ν] .= device.ω̄ .- ΔMAX
    xR = zeros(length(x)); xR[Ω] .= +ΩMAX; xR[ν] .= device.ω̄ .+ ΔMAX
    σ  = zeros(length(x));  σ[Ω] .=    σΩ;  σ[ν] .= σΔ
    global fn_penalty, gd_penalty = CtrlVQE.HardBounds.functions(λ, xL, xR, σ)
end

function initialize_lossfn()
    global fn = CtrlVQE.CompositeCostFunction(fn_energy, fn_penalty)
    global gd = CtrlVQE.CompositeGradientFunction(gd_energy, gd_penalty)
end

function initialize_trace()
    global offset_iterations = 0
    global offset_f_calls = 0
    global offset_g_calls = 0

    global iterations = Int[]
    global trace_f_calls = Int[]
    global trace_g_calls = Int[]

    global adaptations = Int[0]

    global trace_fn = Float64[]
    global trace_fn_energy = Float64[]
    global trace_gd = Float64[]
    global trace_gd_energy = Float64[]
    global trace_gd_penalty = Float64[]
end

function initialize_arrays()
    global ϕ = Array{Float64}(undef, r+1, CtrlVQE.ngrades(device))
    global E = Array{Float64}(undef, r+1)
    global F = Array{Float64}(undef, r+1)

    global α  = Array{Float64}(undef, r+1, CtrlVQE.ndrives(device))
    global β  = Array{Float64}(undef, r+1, CtrlVQE.ndrives(device))
    global ϕα = Array{Float64}(undef, r+1, CtrlVQE.ndrives(device))
    global ϕβ = Array{Float64}(undef, r+1, CtrlVQE.ndrives(device))

    global α̂  = Array{ComplexF64}(undef, length(f), CtrlVQE.ndrives(device))
    global β̂  = Array{ComplexF64}(undef, length(f), CtrlVQE.ndrives(device))
    global ϕ̂α = Array{ComplexF64}(undef, length(f), CtrlVQE.ndrives(device))
    global ϕ̂β = Array{ComplexF64}(undef, length(f), CtrlVQE.ndrives(device))
end

function run_optimization(subdir)
    global optimizer = Optim.LBFGS(
        alphaguess=LineSearches.InitialStatic(scaled=true),
        linesearch=LineSearches.MoreThuente(),
    )
    global options = Optim.Options(
        show_trace = true,
        show_every = 1,
        f_tol = f_tol,
        g_tol = g_tol,
        iterations = maxiter,
        callback = callback_optimizer(subdir),
    )
    global optimization = Optim.optimize(
        x -> (print("\tf:"); @time fn(x)),
        (G,x) -> (print("\tg:"); @time gd(G,x)),
        x,
        optimizer,
        options,
    )
end

function run_pulse()
    global Δ = device.ω̄ .- x[ν]

    CtrlVQE.Parameters.bind(device, x)
    CtrlVQE.gradientsignals(
        device, T, ψ0, r, fn_energy.OT;
        result=ϕ, evolution=algorithm, callback=callback_gradient,
    )

    for i in 1:CtrlVQE.ndrives(device)
        Ω = device.Ω̄[i](t)
        α[:,i] .= real.(Ω)
        β[:,i] .= imag.(Ω)

        j = 2(i-1) + 1
        ϕα[:,i] .= ϕ[:,j  ]
        ϕβ[:,i] .= ϕ[:,j+1]
    end
end

function run_fft()
    #= NOTE: I skip the very first time point in the DFT so that units are consistent.

    The issue is that `fft` expects one sample per evenly-spaced time,
        and the sample at t=0 doesn't really fit into this framework.
    This does *cost information*, so I don't particularly like doing it,
        and I'd bet there is a more sophisticated algorithm accounting for it,
        but I don't know what it is and it probably isn't as fast.
    =#
    α̂  .= rfft( α[2:end,:], 1) ./ r;    α̂[2:end] .*= 2
    β̂  .= rfft( β[2:end,:], 1) ./ r;    β̂[2:end] .*= 2
    ϕ̂α .= rfft(ϕα[2:end,:], 1) ./ r;    ϕ̂α[2:end] .*= 2
    ϕ̂β .= rfft(ϕβ[2:end,:], 1) ./ r;    ϕ̂β[2:end] .*= 2
    #= NOTE: Normalization is a bit odd-looking, isn't it?

        Division by r is the standard DFT normalization;
            I'm not honestly sure why it isn't included in the method.

        Multiplication by 2 represents contributions from NEGATIVE frequencies,
            which are trivial for real pulses and thus omitted in the `rfft` method,
            but are nevertheless included in the standard DFT normalization.
        The multiplication by 2 basically merges
            each positive frequency with its negative counterpart.
        Note that the zero-frequency component HAS no negative counterpart,
            so it is NOT doubled.
    =#

    peak = max(maximum(abs.(ϕ̂α)), maximum(abs.(ϕ̂β)))
    default(x,y) = isnothing(x) ? y : x
    global kMAX = 1
    for i in 1:CtrlVQE.ndrives(device)
        kMAX = max(kMAX,
            default(findlast(abs.(ϕ̂α[:,i]) .> (peak * fFACTOR)), 0),
            default(findlast(abs.(ϕ̂α[:,i]) .> (peak * fFACTOR)), 0),
        )
    end
end

function callback_gradient(i, t_, ψ)
    E[i] = CtrlVQE.evaluate(fn_energy, ψ, t_)
    F[i] = CtrlVQE.evaluate(fn_norm, ψ, t_)
end


function callback_optimizer(subdir)
    return state -> (
        update_trace(state);
        (state.iteration % updateiter == 0) && update(subdir);
        return terminate(state)
    )
end

function terminate(state)
    iteration, f_calls = iterations[end], trace_f_calls[end]
    iteration > 10 && f_calls > fnRATIO * iteration && (
        println("Linesearch excessively difficult");
        return true;
    )
    return false
end

function update_trace(state)
    # FETCH COUNTS
    iteration = offset_iterations + state.iteration
    f_calls = offset_f_calls + fn.counter[]
    g_calls = offset_g_calls + gd.counter[]

    # UPDATE COUNT TRACES
    push!(iterations, iteration)
    push!(trace_f_calls, f_calls)
    push!(trace_g_calls, g_calls)

    # UPDATE FUNCTION VALUE AND GRADIENT NORM TRACES
    push!(trace_fn, state.value)
    push!(trace_fn_energy, fn.values[1])
    push!(trace_gd, state.g_norm)
    push!(trace_gd_energy, gd.norms[1])
    push!(trace_gd_penalty, gd.norms[2])
end

function update_device()
    for q in 1:n
        device.Ω̄[q] = pulses[q]
    end
    CtrlVQE.Parameters.bind(device, x)
end

function update(subdir)
    # UPDATE PARAMETER ARRAY
    global x = CtrlVQE.Parameters.values(device)
    save(subdir)                        # SAVES CURRENT settings, x, AND trace

    # PRINT OUT A STATUS REPORT
    generate_statusreport()

    # ARCHIVE THE CURRENT PARAMETER ARRAY WITH A PERSISTENT NAME
    id = lpad(length(iterations) > 0 ? iterations[end] : 0, 6, "0")
    serialize("$subdir/x_$id", x)

    # GENERATE PLOTS
    generate_pulseplot(subdir, id)
    generate_traceplot(subdir)
    global plot_master = Plots.plot(plot_pulse, plot_trace, layout=(1,2))
    Plots.gui(plot_master)
end

function generate_pulseplot(subdir, id=nothing)
    if id === nothing   # USE CURRENT ITERATION AND PARAMETERS
        id = lpad(length(iterations) > 0 ? iterations[end] : 0, 6, "0")
        xplot = x
    else                # LOAD x FROM GIVEN ITERATION ID
        xplot = deserialize("$subdir/x_$id")
    end

    # RUN CALCULATIONS TO GET SIGNALS IN TIME AND FREQUENCY DOMAINS
    CtrlVQE.Parameters.bind(device, xplot)
    run_pulse()
    run_fft()

    # GENERATE AND SAVE TRANSIENT STAND-ALONE VERSIONS OF EACH SUB-PLOT
    global plot_Et = create_plot_Et(); Plots.savefig(plot_Et, "$subdir/plot_Et.pdf")
    global plot_Ωt = create_plot_Ωt(); Plots.savefig(plot_Ωt, "$subdir/plot_Ωt.pdf")
    global plot_Ωf = create_plot_Ωf(); Plots.savefig(plot_Ωf, "$subdir/plot_Ωf.pdf")
    global plot_ϕt = create_plot_ϕt(); Plots.savefig(plot_ϕt, "$subdir/plot_ϕt.pdf")
    global plot_ϕf = create_plot_ϕf(); Plots.savefig(plot_ϕf, "$subdir/plot_ϕf.pdf")

    # REMOVE REDUNDANT LEGENDS AND y-LABELS
    Plots.plot!(plot_Ωt; legend=false)
    Plots.plot!(plot_Ωf; ylabel="")
    Plots.plot!(plot_ϕt; legend=false)
    Plots.plot!(plot_ϕf; legend=false, ylabel="")

    # ASSEMBLE SUB-PLOTS INTO A MASTER PLOT
    global plot_pulse = Plots.plot(
        plot_Et,
        plot_Ωt, plot_Ωf,
        plot_ϕt, plot_ϕf;

        layout = (Plots.@layout [
            E{0.2h}
            Plots.grid(2,2)
        ]),
        size=(1600,800),
    )
    Plots.savefig(plot_pulse, "$subdir/pulses_$id.pdf")
end

function generate_traceplot(subdir)
    # GENERATE AND SAVE TRANSIENT STAND-ALONE VERSIONS OF EACH SUB-PLOT
    global plot_fg = create_plot_fg(); Plots.savefig(plot_fg, "$subdir/plot_fg.pdf")
    global plot_ct = create_plot_ct(); Plots.savefig(plot_ct, "$subdir/plot_ct.pdf")

    # ASSEMBLE SUB-PLOTS INTO A MASTER PLOT
    global plot_trace = Plots.plot(plot_fg, plot_ct, layout = (2, 1), size=(1600,800))
    Plots.savefig(plot_trace, "$subdir/plot_trace.pdf")
end


include("AdaptiveWindowsPlots.jl")         # Provides methods to create individual subplots.


function generate_statusreport()
    # TODO (mid): flesh this out
    println("""
        Adaptation: $(length(adaptations))
        Parameter count: $(length(x))
        Iteration: $(iterations[end])
        Energy Error: $(trace_fn_energy[end] - FCI)
    """)
end



function run_adaptation(subdir)
    adaptation = length(adaptations)

    # ARCHIVE THE CURRENT PARAMETER ARRAY WITH A PERSISTENT NAME
    id = lpad(adaptation, 6, "0")
    serialize("$subdir/x_A$id", x)
    serialize("$subdir/starttimes_A$id", starttimes)

    # CHECK IF WE'VE REACHED THE MAXIMUM NUMBER OF ADAPTATIONS
    if adaptation ≥ maxadapt
        println("Maximum adaptations reached.")
        return false
    end

    # GENERATE THE GRADIENT SIGNALS
    run_pulse()

    # INTEGRATE GRADIENT SIGNALS AND CHECK IF WE ARE UNDER CONVERGENCE THRESHOLD
    ϕMAX = max(maximum(abs.(ϕα)), maximum(abs.(ϕβ)))
    if ϕMAX ≤ ϕtol
        println("Gradient signal is converged within ϕtol.")
        return false
    end

    # ITERATE ADAPTATION
    adaptation += 1
    push!(adaptations, iterations[end])

    # SET COUNTER OFFSETS
    global offset_iterations = iterations[end]
    global offset_f_calls = trace_f_calls[end]
    global offset_g_calls = trace_g_calls[end]

    # COMBINE PRIOR STARTTIMES WITH NODES OF EACH SWITCHING FUNCTION
    for q in 1:n
        _filter_similar_windows!(starttimes[q], device.Ω̄[q])
        _merge_switching_times!(starttimes[q], _identify_switching_times(ϕα[:,q]))
        _merge_switching_times!(starttimes[q], _identify_switching_times(ϕβ[:,q]))
    end
    # NOTE: UPDATES `global starttimes`

    # CONSTRUCT NEW PARAMETER VECTORS FROM PRIOR OPTIMAL VALUES
    global xi = Float64[]
    for q in 1:n
        xα, xβ = reim(device.Ω̄[q](starttimes[q]))   # OPTIMAL AMPLITUDE FROM PRIOR
        append!(xi, [xα xβ]')                       # INTERLEAVE REAL AND COMPLEX PARTS
    end
    append!(xi, device.ν̄)                           # OPTIMAL FREQUENCIES FROM PRIOR

    # ARCHIVE THE INITIAL PARAMETER ARRAY FOR THIS ADAPTATION WITH A PERSISTENT NAME
    id = lpad(adaptation, 6, "0")
    serialize("$subdir/xi_A$id", xi)

    # FORMALLY UPDATE PARAMETER VARIABLES
    L = length(xi)
    global Ω = 1:L-n
    global ν = 1+L-n:L
    global x = xi

    # UPDATE PARAMETER-DEPENDENT OBJECTS
    initialize_pulses()     # RE-INITIALIZE PULSES WITH NEW STARTTIMES
    update_device()         # UPDATE DEVICE WITH NEW PULSES AND PARAMETERS
    initialize_penaltyfn()  # UPDATE PENALTY FUNCTION TO MATCH NEW PARAMETER PROFILE
    initialize_lossfn()     # UPDATE LOSS FUNCTION WITH THE NEW PENALTY FUNCTION

    return true             # INDICATE WE ARE READY TO RE-OPTIMIZE
end

function _identify_switching_times(ϕ)
    # CALCULATE SIGN OF ϕ AT EACH STEP, TREATING |ϕ|<ϕMIN AS 0
    ϕSIGN = [ (x ≤ -ϕtol ? -1 : x ≥ ϕtol ? +1 : 0) for x in ϕ ]

    # IDENTIFY INDICES WHERE SIGN SWITCHES: 0's handled a little ad-hoc
    switching_indices = Int[1]  # ALL INDICES WHERE SIGN IS SAID TO HAVE "SWITCHED"

    init_search = true      # FLAG INDICATING WE ARE SEARCHING FOR A NON-ZERO SIGN
    prior = 0               # TRACKS SIGN PRIOR TO ENCOUNTERING STRING OF ZERO
    cursor = 0              # TRACKS INDEX OF CURRENT STRING
    for i in eachindex(ϕSIGN)
        init_search && ϕSIGN[i] == 0 && continue    # Search until we find a non-zero.
        if init_search                              # Begin the algorithm in earnest.
            cursor = i
            init_search = false
        end

        ϕSIGN[i] == ϕSIGN[cursor] && continue       # Same sign as current string. Skip.

        if ϕSIGN[cursor] == 0    # (`prior` →) 0 → ±

            if ϕSIGN[i] == prior     # ± → 0 → ±    # Treat 0 as opposite sign.
                push!(switching_indices, cursor)
                push!(switching_indices, i)
                cursor = i
                continue
            else                    # ∓ → 0 → ±     # Treat 0 as transition region.
                push!(switching_indices, (cursor+i)÷2)
                cursor = i
                continue
            end
        end

        if ϕSIGN[i] == 0         # ± → 0 (→ ???)    # Must identify ??? before updating.
            prior = ϕSIGN[cursor]
            cursor = i
            continue
        else                    # ± → ∓             # Obviously a switch.
            push!(switching_indices, i)
            cursor = i
            continue
        end
    end

    # EXTRACT TIMES
    return t[switching_indices]
    #= NOTE: We could do this more scientifically with an interpolation package,
        but that may be numerically unstable.
    =#
end

function _filter_similar_windows!(olds, Ω)
    cursor = 2
    while cursor <= length(olds)
        if abs(Ω(olds[cursor]) - Ω(olds[cursor-1])) < Ωtol
            deleteat!(olds, cursor)
        else
            cursor += 1
        end
    end
end

function _merge_switching_times!(olds, news)
    cursor = 1      # TRACKS LOCATION IN `olds`
    temps = eltype(olds)[]
    for i in eachindex(news)
        # ALWAYS INSERT ELEMENTS FROM `olds`
        while cursor ≤ length(olds) && olds[cursor] ≤ news[i]
            push!(temps, olds[cursor])
            cursor += 1
        end

        # INSERT ELEMENT FROM `news` ONLY IF IT ISN'T TOO CLOSE TO ANOTHER VALUE
        length(temps) ≥ 1 && abs(news[i]-temps[end]) ≤ τtol*τ && continue
        cursor ≤ length(olds) && abs(news[i]-olds[cursor]) ≤ τtol*τ && continue
        push!(temps, news[i])
    end

    # FILL IN REMAINING ELEMENTS FROM `olds`
    while cursor ≤ length(olds)
        push!(temps, olds[cursor])
        cursor += 1
    end

    # EMPTY `olds` AND RE-FILL IT WITH `temps` VALUES
    empty!(olds)
    append!(olds, temps)
end
