run = false
plt = true

filename = "resources"

datapath = "pkgs/Benchmarking/dat"
figpath  = "pkgs/Benchmarking/fig"
import Dates; today = Dates.today()
# today = "2023-04-11"
csvpath = "$datapath/$filename.$today.csv"

if run
    import Experiments
    import Benchmarking
    import Benchmarking: ResourceBenchmark as Benchmark
    import CtrlVQE

    if !isfile(csvpath)
        open(csvpath, "a") do io
            Experiments.header(io,
                Benchmark.Control,
                Benchmark.Independent,
                Benchmark.Dependent
            )
            flush(io)
        end
    end

    println("Collecting data...")

    for i in 1:21
        N, m, n = Benchmarking.Nmn[i]

        benchmark = Benchmark.Control(
            Float64,                    # NUMBER TYPE
            CtrlVQE.Evolutions.Rotate,  # ALGORITHM TO BENCHMARK
            m, n,                       # SYSTEM SIZE
            1, 1,               # NUMBERS IN OBSERVABLE ARE IRRELEVANT FOR TIMING
            2π*0.02, 2π*0.75,   # STANDARD PULSE PARAMETERS
            1, 1,               # NOT USING INDEX IN THIS BENCHMARK...
        )

        setup = Experiments.initialize(benchmark)

        r = 2^11                        # "SUFFICIENT" TROTTERIZATION FOR OPTIMAL ACCURACY
        xvars = Benchmark.Independent(
            r, r,               # USE CONSISTENT TROTTER STEPS FOR EVOLUTION AND GRADIENT
            1.0,                # TIME DURATION DOESN'T MATTER FOR TIMING
        )

        result = Experiments.runtrial(benchmark, setup, xvars)
        yvars = Experiments.synthesize(benchmark, setup, xvars, result)

        open(csvpath, "a") do io
            Experiments.record(io, benchmark, xvars, yvars)
            flush(io)
        end
    end

    println("...finished!")
end

if plt
    import DataFrames: DataFrame, groupby, combine, AsTable
    import DelimitedFiles: readdlm
    using Plots

    # READ IN DATA
    data, header = readdlm(csvpath, '\t', header=true)
    df = DataFrame(data, vec(header))

    # ADD AN EXTRA COLUMN FOR TOTAL HILBERT SPACE
    df.N = df.m .^ df.n

    # CONSTRAIN CONTROL PARAMETERS
    df = df[(df.float .== 1) .&& (df.alg_ix .== 1) .&& (df.rE .== df.rG .== 2048), :]
        # NOTE: All the others should be utterly irrelevant to benchmarking data.


    ###########################################################
    # ABSOLUTE TIME PLOT

    fig_time = scatter(
        title  = "Runtime: 2048 Trotter Steps",
        xlabel = "Size of Hilbert Space", ylabel = "Time (s)",
        xscale = :log, yscale = :log,
        xlims = [1, 1e4], ylims=[1e-2, 1e5],
        legend = :topleft,
        dpi = 300,
    )

    scatter!(fig_time, df.N, df.time_E ./ 1.0e9, label="Evolution",
        markershape=:auto, markeralpha=0.7)
    scatter!(fig_time, df.N, df.time_G ./ 1.0e9, label="Gradient Signal",
        markershape=:auto, markeralpha=0.7)

    savefig(fig_time, "$figpath/$filename.time.pdf")

    ###########################################################
    # RELATIVE TIME PLOT

    fig_gradcost = scatter(
        title  = "Gradient Runtime, wrt Evolution Runtime",
        xlabel = "Size of Hilbert Space", ylabel = "Multiple",
        xscale = :log, yscale = :log,
        xlims = [1, 1e4], ylims=[1, 1e3],
        legend = false,
        dpi = 300,
    )

    scatter!(fig_gradcost, df.N, df.time_G ./ df.time_E,
        markershape=:auto, markeralpha=0.7)

    savefig(fig_gradcost, "$figpath/$filename.gradcost.pdf")

    ###########################################################
    # SCALING PLOT

    fig_scaling = scatter(
        title  = "Runtime Scaling",
        xlabel = "Size of Hilbert Space", ylabel = "Power of N",
        xscale = :log, # yscale = :log,
        xlims = [1, 1e4], ylims=[0, 10],
        legend = :topleft,
        dpi = 300,
    )

    gps = groupby(df, :N)
    gpdf = combine(
        gps,
        [:time_E, :time_G]
        => ((tE, tG) -> (time_E=min(tE...), time_G=min(tG...)))
        => AsTable
    )


    σ = sortperm(gpdf.N)
    pN = log10.(gpdf.N[σ])
    pE = log10.(gpdf.time_E[σ])
    pG = log10.(gpdf.time_G[σ])

    cfd(y,x,k,l=k) = [(y[i+k]-y[i-k]) / (x[i+k]-x[i-k]) for i in 1+l:length(y)-l]

    x = gpdf.N[σ][1+2:end-2]
    # yE = (4*cfd(pE,pN,1,2) .- cfd(pE,pN,2)) ./ 3
    # yG = (4*cfd(pG,pN,1,2) .- cfd(pG,pN,2)) ./ 3
    yE = cfd(pE,pN,1)
    yG = cfd(pG,pN,1)

    scatter!(fig_scaling, x, yE, label="Evolution",
        markershape=:auto, markeralpha=0.7)
    scatter!(fig_scaling, x, yG, label="Gradient",
        markershape=:auto, markeralpha=0.7)

    savefig(fig_scaling, "$figpath/$filename.scaling.pdf")
end