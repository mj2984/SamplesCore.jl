module DomainArrays

using Base: @propagate_inbounds, IndexStyle, axes, size, eltype, show
using Printf: @sprintf

############################
# Core domain array types  #
############################

abstract type AbstractDomainArray{T,N} <: AbstractArray{T,N} end

struct DomainArray{T,N,A<:AbstractArray{T,N},R<:NTuple{N,Union{Nothing,Real}}} <: AbstractDomainArray{T,N}
    data::A
    rate::R
end

struct DomainView{T,N,A<:AbstractArray{T,N},
                  R<:NTuple{N,Union{Nothing,Real}},
                  O<:NTuple{N,Float64}} <: AbstractDomainArray{T,N}
    data::A
    rate::R
    offset::O
end

Base.IndexStyle(::Type{<:AbstractDomainArray}) = IndexLinear()
Base.size(D::AbstractDomainArray) = size(D.data)
Base.eltype(::Type{<:AbstractDomainArray{T}}) where T = T

Base.strides(D::AbstractDomainArray) = strides(D.data)
Base.pointer(D::DomainArray) = pointer(D.data)
Base.unsafe_convert(::Type{Ptr{T}}, D::DomainArray{T}) where {T} =
    Base.unsafe_convert(Ptr{T}, D.data)

########################
# Rounding + indexing  #
########################

abstract type TimeRounding end
struct DefaultRounding <: TimeRounding end
struct ExtremeRounding <: TimeRounding end

const _default_rounding = DefaultRounding()

########################
# Explicit sample index#
########################

struct Samples
    value::Int
end

########################
# Domain index + axes  #
########################

struct DomainIdx{T}
    t::T
    i::Int
end

struct DomainAxis{T}
    first_t::T
    step_t::T
    len::Int
end

Base.length(ax::DomainAxis) = ax.len

Base.getindex(ax::DomainAxis{T}, k::Int) where {T} =
    DomainIdx{T}(ax.first_t + (k-1)*ax.step_t, k)

function Base.iterate(ax::DomainAxis{T}, state::Int=1) where {T}
    state > ax.len && return nothing
    return (ax[state], state+1)
end

Base.IteratorSize(::Type{<:DomainAxis}) = Base.HasLength()
Base.eltype(::Type{DomainAxis{T}}) where {T} = DomainIdx{T}

########################
# Sample + domain axes #
########################

sampleaxes(D::AbstractDomainArray, d::Int) = axes(D.data, d)

sampleaxes(D::AbstractDomainArray) =
    ntuple(d -> sampleaxes(D, d), ndims(D))

function domainaxis(D::AbstractDomainArray, d::Int)
    r = D.rate[d]
    idxs = sampleaxes(D, d)
    if r === nothing
        return idxs
    else
        first_i = first(idxs)
        len     = length(idxs)
        first_t = (first_i - 1) / r
        step_t  = 1 / r
        return DomainAxis(first_t, step_t, len)
    end
end

domainaxes(D::AbstractDomainArray) =
    ntuple(d -> domainaxis(D, d), ndims(D))

Base.axes(D::AbstractDomainArray) = domainaxes(D)

function domainsize(D::AbstractDomainArray)
    ntuple(d -> begin
        ax = domainaxis(D, d)
        if ax isa AbstractRange
            return last(ax) - first(ax)
        elseif ax isa DomainAxis
            ax.step_t * (ax.len - 1)
        else
            return length(ax)
        end
    end, ndims(D))
end

function domainrange(D::AbstractDomainArray, d::Int)
    ax = domainaxis(D, d)
    if ax isa AbstractRange
        return (first(ax), last(ax))
    elseif ax isa DomainAxis
        lo = ax.first_t
        hi = ax.first_t + (ax.len - 1) * ax.step_t
        return (lo, hi)
    else
        return (1, length(ax))
    end
end

domainrange(D::AbstractDomainArray) =
    ntuple(d -> domainrange(D, d), ndims(D))

function domainextent(D::AbstractDomainArray)
    ntuple(d -> begin
        lo, hi = domainrange(D, d)
        hi - lo
    end, ndims(D))
end

########################
# Indexing helpers     #
########################

_to_index_and_offset(idx::DomainIdx, r, ::TimeRounding) = (idx.i, 0.0)

_to_index_and_offset(s::Samples, r, ::TimeRounding) = (s.value, 0.0)

function _to_index_and_offset(t::Real, r::Real, ::DefaultRounding)
    raw = floor(Int, t * r)
    i = max(raw, 1)
    t1 = (i - 1) / r
    return (i, t1 - t)
end

function _to_index_and_offset(t::Real, r::Real, ::ExtremeRounding)
    raw = ceil(Int, t * r)
    i = max(raw, 1)
    t1 = (i - 1) / r
    return (i, t1 - t)
end

function _to_index_and_offset(tr::AbstractRange{<:Real}, r::Real, ::DefaultRounding)
    lo = first(tr); hi = last(tr)
    i1 = ceil(Int, lo * r) + 1
    i2 = floor(Int, hi * r)
    i2 = max(i2, i1)
    t1 = (i1 - 1) / r
    return (i1:i2, t1 - lo)
end

function _to_index_and_offset(tr::AbstractRange{<:Real}, r::Real, ::ExtremeRounding)
    lo = first(tr); hi = last(tr)
    i1 = floor(Int, lo * r) + 1
    i2 = ceil(Int, hi * r)
    i2 = max(i2, i1)
    t1 = (i1 - 1) / r
    return (i1:i2, t1 - lo)
end

_to_index_and_offset(i::Integer, ::Nothing, ::TimeRounding) = (i, 0.0)
_to_index_and_offset(::Colon, r, ::TimeRounding) = (Colon(), 0.0)

function _indices_and_offsets(rate::NTuple{N,Union{Nothing,Real}},
                              I::NTuple{N,Any},
                              rounding::TimeRounding) where {N}
    inds = ntuple(i -> nothing, N)
    offs = ntuple(i -> 0.0, N)
    for k in 1:N
        idx, off = _to_index_and_offset(I[k], rate[k], rounding)
        inds = Base.setindex(inds, idx, k)
        offs = Base.setindex(offs, off, k)
    end
    return inds, offs
end

_is_scalar_index(i) = i isa Integer || i isa CartesianIndex || i isa DomainIdx
_is_scalar_index(::Colon) = false
_is_scalar_index(i::AbstractRange) = false
_is_scalar_index(i::AbstractVector) = false

########################
# Indexing & views     #
########################

@propagate_inbounds function Base.getindex(D::AbstractDomainArray{T,N}, I::Vararg{Any,N}) where {T,N}
    inds, _ = _indices_and_offsets(D.rate, I, _default_rounding)
    if all(_is_scalar_index, I)
        return D.data[inds...]
    else
        new_data = D.data[inds...]
        return DomainArray(new_data, D.rate)
    end
end

@propagate_inbounds function Base.setindex!(D::DomainArray{T,N}, v, I::Vararg{Any,N}) where {T,N}
    inds, _ = _indices_and_offsets(D.rate, I, _default_rounding)
    return setindex!(D.data, v, inds...)
end

@propagate_inbounds function Base.setindex!(D::DomainView{T,N}, v, I::Vararg{Any,N}) where {T,N}
    inds, _ = _indices_and_offsets(D.rate, I, _default_rounding)
    return setindex!(D.data, v, inds...)
end

function domainslice(D::DomainArray{T,N}, I::Vararg{Any,N}) where {T,N}
    inds, _ = _indices_and_offsets(D.rate, I, _default_rounding)
    new_data = D.data[inds...]
    return DomainArray(new_data, D.rate)
end

function domainview(D::DomainArray{T,N}, I::Vararg{Any,N}) where {T,N}
    inds, offs = _indices_and_offsets(D.rate, I, _default_rounding)
    v = @view D.data[inds...]
    offset = ntuple(d -> offs[d], N)
    return DomainView{T,N,typeof(v),typeof(D.rate),typeof(offset)}(v, D.rate, offset)
end

function domainview(D::DomainView{T,N}, I::Vararg{Any,N}) where {T,N}
    inds, offs = _indices_and_offsets(D.rate, I, _default_rounding)
    v = @view D.data[inds...]
    offset = ntuple(d -> D.offset[d] + offs[d], N)
    return DomainView{T,N,typeof(v),typeof(D.rate),typeof(offset)}(v, D.rate, offset)
end

function domainview_extreme(D::DomainArray{T,N}, I::Vararg{Any,N}) where {T,N}
    inds, offs = _indices_and_offsets(D.rate, I, ExtremeRounding())
    v = @view D.data[inds...]
    offset = ntuple(d -> offs[d], N)
    return DomainView{T,N,typeof(v),typeof(D.rate),typeof(offset)}(v, D.rate, offset)
end

macro extreme_view(ex)
    @assert ex.head === :call
    fn = ex.args[1]
    if fn === :domainview
        return :(domainview_extreme($(ex.args[2:end]...)))
    else
        error("@extreme_view only wraps `domainview`")
    end
end

macro sampleidx(ex)
    return _rewrite_sampleidx(ex)
end

function _rewrite_sampleidx(ex)
    if ex isa Expr && ex.head === :ref
        arr = ex.args[1]
        idxs = ex.args[2:end]
        return :( $(esc(arr)).data[$(map(esc, idxs)...) ] )
    end
    if ex isa Expr
        newargs = map(_rewrite_sampleidx, ex.args)
        return Expr(ex.head, newargs...)
    end
    return ex
end

@propagate_inbounds function Base.view(D::AbstractDomainArray{T,N}, I::Vararg{Any,N}) where {T,N}
    return domainview(D, I...)
end

########################
# Domain shifting      #
########################

function shiftdomain(D::DomainArray, shifts::Vararg{Real})
    @assert length(shifts) == length(D.rate)
    newoffset = ntuple(i -> D.rate[i] === nothing ? 0.0 : shifts[i], length(D.rate))
    DomainView(D.data, D.rate, newoffset)
end
shiftdomain(D::DomainArray, shift::Real) = shiftdomain(D, (shift,))

function shiftdomain(D::DomainView, shifts::Vararg{Real})
    @assert length(shifts) == length(D.rate)
    newoffset = ntuple(i -> D.offset[i] + (D.rate[i] === nothing ? 0.0 : shifts[i]),
                       length(D.rate))
    DomainView(D.data, D.rate, newoffset)
end
shiftdomain(D::DomainView, shift::Real) = shiftdomain(D, (shift,))

########################
# Rates + arithmetic   #
########################

_rates_match(a::AbstractDomainArray, b::AbstractDomainArray) = a.rate == b.rate

_unwrap(x::AbstractDomainArray) = x.data
_unwrap(x) = x

function _binary_op(op, A::AbstractDomainArray, B::AbstractDomainArray)
    _rates_match(A, B) || error("Domain rates do not match")
    data = op.(_unwrap(A), _unwrap(B))
    return DomainArray(data, A.rate)
end

Base.:+(A::AbstractDomainArray, B::AbstractDomainArray) = _binary_op(+, A, B)
Base.:-(A::AbstractDomainArray, B::AbstractDomainArray) = _binary_op(-, A, B)
Base.:*(A::AbstractDomainArray, B::AbstractDomainArray) = _binary_op(*, A, B)
Base.:/(A::AbstractDomainArray, B::AbstractDomainArray) = _binary_op(/, A, B)

Base.:+(A::AbstractDomainArray, x::Number) = DomainArray(A.data .+ x, A.rate)
Base.:+(x::Number, A::AbstractDomainArray) = A + x

Base.:-(A::AbstractDomainArray, x::Number) = DomainArray(A.data .- x, A.rate)
Base.:-(x::Number, A::AbstractDomainArray) = DomainArray(x .- A.data, A.rate)

Base.:*(A::AbstractDomainArray, x::Number) = DomainArray(A.data .* x, A.rate)
Base.:*(x::Number, A::AbstractDomainArray) = A * x

Base.:/(A::AbstractDomainArray, x::Number) = DomainArray(A.data ./ x, A.rate)

function Base.broadcast(f, A::AbstractDomainArray, Bs...)
    rate = A.rate
    for B in Bs
        if B isa AbstractDomainArray
            _rates_match(A, B) || error("Domain rates do not match")
        end
    end
    data = Base.broadcast(f, _unwrap(A), map(_unwrap, Bs)...)
    return DomainArray(data, rate)
end

function Base.similar(D::AbstractDomainArray, ::Type{T}, dims::Dims) where {T}
    DomainArray(similar(D.data, T, dims), D.rate)
end

Base.similar(D::AbstractDomainArray) = DomainArray(similar(D.data), D.rate)

########################
# Pretty-printing      #
########################

function _span(D)
    dims = size(D)
    rates = D.rate
    spans = Vector{Float64}(undef, length(dims))
    for i in eachindex(dims)
        r = rates[i]
        spans[i] = r === nothing ? dims[i] : dims[i] / r
    end
    return spans
end

function _pretty_rate(rate::NTuple{N,Union{Nothing,Real}}) where {N}
    out = Vector{Any}(undef, N)
    for i in 1:N
        r = rate[i]
        out[i] = r === nothing ? "_" : r
    end
    return tuple(out...)
end

function Base.show(io::IO, ::MIME"text/plain", D::DomainArray)
    spans = _span(D)
    span_str = "(" * join((@sprintf("%.6g", s) for s in spans), ", ") * ")"
    rate_str = "(" * join(_pretty_rate(D.rate), ", ") * ")"
    print(io, span_str, " DomainArray @ ", rate_str, "\n")
    inner = IOContext(io, :compact => true)
    Base.show(inner, MIME"text/plain"(), D.data)
end

function Base.show(io::IO, ::MIME"text/plain", D::DomainView)
    spans = _span(D)
    span_str = "(" * join((@sprintf("%.6g", s) for s in spans), ", ") * ")"
    rate_str = "(" * join(_pretty_rate(D.rate), ", ") * ")"
    print(io, span_str, " DomainView @ ", rate_str,
          " with offset ", D.offset, "\n")
    inner = IOContext(io, :compact => true)
    Base.show(inner, MIME"text/plain"(), D.data)
end

########################
# Dimension conversion #
########################

_to_pair(x::Tuple{<:Real,<:Union{Real,Nothing}}) = x
_to_pair(x::Real) = (x, nothing)

_normalize_dims(dims::Vararg{Any}) = map(_to_pair, dims)
_normalize_dims(dims::Tuple) = map(_to_pair, dims)

_dim_from_pair((len, rate)::Tuple{<:Real,<:Real}) = Int(round(len * rate))
_dim_from_pair((len, rate)::Tuple{<:Integer,Nothing}) = len
_dim_from_pair((len, rate)::Tuple{<:Real,Nothing}) =
    error("Real-valued dimension requires a sampling rate")

_rate_from_pair((_, rate)) = rate

function _make_domainarray(f, T, dims)
    sizes = map(_dim_from_pair, dims)
    rates = map(_rate_from_pair, dims)
    arr   = f(T, sizes...)
    return DomainArray(arr, tuple(rates...))
end

domainzeros(dims...) = _make_domainarray(zeros, Float64, _normalize_dims(dims...))
domainzeros(T::Type, dims...) = _make_domainarray(zeros, T, _normalize_dims(dims...))

domainones(dims...) = _make_domainarray(ones, Float64, _normalize_dims(dims...))
domainones(T::Type, dims...) = _make_domainarray(ones, T, _normalize_dims(dims...))

domainrand(dims...) = _make_domainarray(rand, Float64, _normalize_dims(dims...))
domainrand(T::Type, dims...) = _make_domainarray(rand, T, _normalize_dims(dims...))

domainfill(value, dims...) = _make_domainarray((T,s...)->fill(value, s...), Float64, _normalize_dims(dims...))
domainfill(T::Type, value, dims...) = _make_domainarray((T,s...)->fill(value, s...), T, _normalize_dims(dims...))

domainfull = domainfill

macro domains(exprs...)
    pairs = map(x -> :(($x, nothing)), exprs)
    return Expr(:tuple, pairs...)
end

########################
# Interpolation API    #
########################

abstract type AbstractInterpolator end

"""
    NearestInterp()

Simple nearest-sample interpolator: just uses domain indexing.
You can plug in more sophisticated ones later.
"""
struct NearestInterp <: AbstractInterpolator end

interp(::NearestInterp, A::AbstractDomainArray, t) = A[t]

########################
# Loop macros          #
########################

"""
    @domainloop for t in axes(A, d)
        body
    end

Rewrites to iterate over `DomainIdx`s, binding:

- `t` to the domain coordinate (`idx.t`)
- `idx` (hidden) to the domain index object

and uses `A[idx]` for writes when you write `A[t]`.
"""
macro domainloop(ex)
    @assert ex.head === :for "Usage: @domainloop for t in iter ... end"
    loopvar, iter = ex.args[1], ex.args[2]
    body = ex.args[3]

    idx_sym = gensym(:idx)
    t_sym   = loopvar

    # Rewrite A[t] on LHS of assignments to A[idx_sym]
    function rewrite_lhs(e)
        if e isa Expr && e.head === :(=)
            lhs, rhs = e.args
            if lhs isa Expr && lhs.head === :ref
                arr = lhs.args[1]
                idxs = lhs.args[2:end]
                new_idxs = map(x -> x == t_sym ? idx_sym : x, idxs)
                new_lhs = Expr(:ref, arr, new_idxs...)
                return Expr(:(=), new_lhs, rewrite_lhs(rhs))
            end
        end
        if e isa Expr
            return Expr(e.head, map(rewrite_lhs, e.args)...)
        end
        return e
    end

    new_body = rewrite_lhs(body)

    return quote
        for $idx_sym in $iter
            local $t_sym = $idx_sym.t
            $new_body
        end
    end |> esc
end

"""
    @with_interpolator Method src for t in axes(A, d)
        body
    end

All reads from `src[t]` are rewritten to `interp(Method(), src, t)`,
while writes like `A[t] = ...` are lowered to `A[idx] = ...` using
the precomputed sample index from `DomainIdx`.
"""
macro with_interpolator(method, src, ex)
    @assert ex.head === :for "Usage: @with_interpolator Method src for t in iter ... end"
    loopvar, iter = ex.args[1], ex.args[2]
    body = ex.args[3]

    idx_sym = gensym(:idx)
    t_sym   = loopvar
    m_sym   = gensym(:method)

    # Rewrite src[t] on RHS to interp(method, src, t)
    function rewrite_reads(e)
        if e isa Expr && e.head === :ref
            arr = e.args[1]
            idxs = e.args[2:end]
            if arr == src && any(x -> x == t_sym, idxs)
                return :(interp($m_sym, $arr, $t_sym))
            end
        end
        if e isa Expr
            return Expr(e.head, map(rewrite_reads, e.args)...)
        end
        return e
    end

    # Rewrite A[t] on LHS to A[idx_sym]
    function rewrite_lhs(e)
        if e isa Expr && e.head === :(=)
            lhs, rhs = e.args
            if lhs isa Expr && lhs.head === :ref
                arr = lhs.args[1]
                idxs = lhs.args[2:end]
                new_idxs = map(x -> x == t_sym ? idx_sym : x, idxs)
                new_lhs = Expr(:ref, arr, new_idxs...)
                return Expr(:(=), new_lhs, rewrite_lhs(rhs))
            end
        end
        if e isa Expr
            return Expr(e.head, map(rewrite_lhs, e.args)...)
        end
        return e
    end

    body1 = rewrite_reads(body)
    body2 = rewrite_lhs(body1)

    return quote
        local $m_sym = $(esc(method))()
        for $idx_sym in $iter
            local $t_sym = $idx_sym.t
            $body2
        end
    end |> esc
end

end # module
