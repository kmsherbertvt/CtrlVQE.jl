import ..CtrlVQE: Validation

function Validation.validate(grid::IntegrationType{F}) where {F}
    # ABSTRACT INTERFACE DEFINED
    r = nsteps(grid);           @assert r isa Int
    t0 = timeat(grid,0);        @assert t0 isa F
    τ0 = stepat(grid,0);        @assert τ0 isa F

    # CONSISTENT INTEGRAL
    a = starttime(grid);        @assert a isa F
    b = endtime(grid);          @assert b isa F
    T = duration(grid);         @assert T isa F
    τ = stepsize(grid);         @assert τ isa F
        @assert T == b-a
        @assert T == r*τ

    # CONSISTENT VECTORIZATION
    latis = lattice(grid);    @assert latis isa Vector{F}
    latis_ = similar(latis)
    latis__ = lattice(grid; result=latis_)
    latis___ = [timeat(grid,i) for i in eachindex(grid)]
    latis____ = collect(grid)
        @assert latis_ == latis
        @assert latis_ === latis__
        @assert latis_ == latis___
        @assert latis_ == latis____
        @assert first(latis_) == a
        @assert last(latis_) == b
        @assert length(latis_) == r + 1

    # CONSISTENT INTEGRATION
    I = integrate(grid, latis)
    I_ = integrate(grid, identity)
    I__ = integrate(grid, (t,f)->f, latis)
        @assert I == I_
        @assert I == I__
        @assert abs(I - (b^2 - a^2)/2) < eps(F)
end