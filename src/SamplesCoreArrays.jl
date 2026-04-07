############################
# Core domain array types  #
############################

abstract type AbstractDomainArray{T,N} <: AbstractArray{T,N} end

"""
    DomainArray{T,N,A,R}

Wrapper around an `AbstractArray{T,N}` with per-dimension sampling rates.

- `A` is the underlying array type
- `R` is `NTuple{N,Union{Nothing,Real}}`
  - `Real`    → this dimension is domain-indexed (e.g. time)
  - `Nothing` → this dimension is sample-indexed (plain integer axis)
"""
struct DomainArray{T,N,A<:AbstractArray{T,N},R<:NTuple{N,Union{Nothing,Real}}} <: AbstractDomainArray{T,N}
    data::A
    rate::R
end

"""
    DomainView{T,N,A,R,O}

View of a `DomainArray` that:

- wraps a `SubArray`
- preserves sampling rates
- tracks accumulated domain offsets per dimension
"""
struct DomainView{T,N,A<:AbstractArray{T,N},
                  R<:NTuple{N,Union{Nothing,Real}},
                  O<:NTuple{N,Float64}} <: AbstractDomainArray{T,N}
    data::A
    rate::R
    offset::O
end

########################
# Array interface      #
########################

Base.IndexStyle(::Type{<:AbstractDomainArray}) = IndexLinear()
Base.size(D::AbstractDomainArray) = size(D.data)
Base.eltype(::Type{<:AbstractDomainArray{T}}) where T = T

########################
# Rounding + indexing  #
########################

abstract type TimeRounding end
struct DefaultRounding <: TimeRounding end
struct ExtremeRounding <: TimeRounding end

const _default_rounding = DefaultRounding()

# Sample-indexed dimension: pass through
_to_index_and_offset(idx, ::Nothing, ::TimeRounding) = (idx, 0.0)

# Scalar time → sample index (default rounding)
function _to_index_and_offset(t::Real, r::Real, ::DefaultRounding)
    raw = floor(Int, t * r)
    i = max(raw, 1)
    t1 = (i - 1) / r
    return (i, t1 - t)
end

# Scalar time → sample index (extreme rounding)
function _to_index_and_offset(t::Real, r::Real, ::ExtremeRounding)
    raw = ceil(Int, t * r)
    i = max(raw, 1)
    t1 = (i - 1) / r
    return (i, t1 - t)
end

# Time range → integer range (default rounding)
function _to_index_and_offset(tr::AbstractRange{<:Real}, r::Real, ::DefaultRounding)
    lo = first(tr); hi = last(tr)
    i1 = ceil(Int, lo * r) + 1
    i2 = floor(Int, hi * r)
    i2 = max(i2, i1)
    t1 = (i1 - 1) / r
    return (i1:i2, t1 - lo)
end

# Time range → integer range (extreme rounding)
function _to_index_and_offset(tr::AbstractRange{<:Real}, r::Real, ::ExtremeRounding)
    lo = first(tr); hi = last(tr)
    i1 = floor(Int, lo * r) + 1
    i2 = ceil(Int, hi * r)
    i2 = max(i2, i1)
    t1 = (i1 - 1) / r
    return (i1:i2, t1 - lo)
end

# Integer / Colon indices
_to_index_and_offset(i::Integer, r, ::TimeRounding) = (i, 0.0)
_to_index_and_offset(::Colon, r, ::TimeRounding) = (Colon(), 0.0)

# Disambiguation for integer + real rate + specific rounding
_to_index_and_offset(i::Integer, r::Real, ::DefaultRounding) = (i, 0.0)
_to_index_and_offset(i::Integer, r::Real, ::ExtremeRounding) = (i, 0.0)

# Apply to all dimensions
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

_is_scalar_index(i) = i isa Integer || i isa CartesianIndex
_is_scalar_index(::Colon) = false
_is_scalar_index(i::AbstractRange) = false
_is_scalar_index(i::AbstractVector) = false

########################
# Domain axes          #
########################

"""
    sampleaxes(D, d)

Return the underlying sample-space axis (integer-based) for dimension `d`.
"""
sampleaxes(D::AbstractDomainArray, d::Int) = axes(D.data, d)

sampleaxes(D::AbstractDomainArray) =
    ntuple(d -> sampleaxes(D, d), ndims(D))

"""
    domainaxis(D, d)

Return the domain-space axis for dimension `d`.

- If `rate[d] === nothing`, this is just the sample axis.
- Otherwise, it is a `StepRange` in domain units.
"""
function domainaxis(D::AbstractDomainArray, d::Int)
    r = D.rate[d]
    idxs = sampleaxes(D, d)
    if r === nothing
        return idxs
    else
        return (first(idxs)-1)/r : 1/r : (last(idxs)-1)/r
    end
end

domainaxes(D::AbstractDomainArray) =
    ntuple(d -> domainaxis(D, d), ndims(D))

# Make Base.axes domain-first
Base.axes(D::AbstractDomainArray) = domainaxes(D)

function domainsize(D::AbstractDomainArray)
    ntuple(d -> begin
        ax = domainaxis(D, d)
        if isa(ax, AbstractRange)
            return last(ax) - first(ax)
        else
            return length(ax)
        end
    end, ndims(D))
end

function domainrange(D::AbstractDomainArray, d::Int)
    ax = domainaxis(D, d)
    return (first(ax), last(ax))
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
# Indexing & views     #
########################

@propagate_inbounds function Base.getindex(D::AbstractDomainArray{T,N}, I::Vararg{Any,N}) where {T,N}
    inds, _ = _indices_and_offsets(D.rate, I, _default_rounding)

    if all(_is_scalar_index, I)
        return D.data[inds...]  # element
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

"""
    domainslice(D, inds...)

Slice in domain space, returning a new `DomainArray`.
"""
function domainslice(D::DomainArray{T,N}, I::Vararg{Any,N}) where {T,N}
    inds, _ = _indices_and_offsets(D.rate, I, _default_rounding)
    new_data = D.data[inds...]
    return DomainArray(new_data, D.rate)
end

"""
    domainview(D, inds...)

View in domain space, returning a `DomainView`.
"""
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

"""
    @sampleidx A[...]
Rewrite `A[...]` to `A.data[...]` for explicit sample-domain indexing.
"""
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
# Explicit sample index#
########################

struct Samples
    value::Int
end
_to_index_and_offset(s::Samples, r, rounding) = (s.value, 0.0)

########################
# Domain shifting      #
########################

"""
    shiftdomain(D, shifts...)

Shift the domain origin by the given amounts (in domain units) per dimension.
"""
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

# generic broadcast that preserves DomainArray and checks rates
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

# Delegate all mixed operations to broadcast
Base.:+(A::AbstractDomainArray, B) = broadcast(+, A, B)
Base.:+(A, B::AbstractDomainArray) = broadcast(+, A, B)

Base.:-(A::AbstractDomainArray, B) = broadcast(-, A, B)
Base.:-(A, B::AbstractDomainArray) = broadcast(-, A, B)

Base.:*(A::AbstractDomainArray, B) = broadcast(*, A, B)
Base.:*(A, B::AbstractDomainArray) = broadcast(*, A, B)

Base.:/(A::AbstractDomainArray, B) = broadcast(/, A, B)
Base.:/(A, B::AbstractDomainArray) = broadcast(/, A, B)

# similar: preserve rate, wrap in DomainArray
function Base.similar(D::AbstractDomainArray, ::Type{T}, dims::Dims) where {T}
    DomainArray(similar(D.data, T, dims), D.rate)
end

Base.similar(D::AbstractDomainArray) = DomainArray(similar(D.data), D.rate)

########################
# Pretty-printing      #
########################

# Compute span for each dimension (size/rate)
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

# Format sampling rates: Real → itself, Nothing → "_"
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

    # Header
    print(io, span_str, " DomainArray @ ", rate_str, "\n")

    # Indent underlying array for readability
    inner = IOContext(io, :compact => true)
    Base.show(inner, MIME"text/plain"(), D.data)
end

function Base.show(io::IO, ::MIME"text/plain", D::DomainView)
    spans = _span(D)
    span_str = "(" * join((@sprintf("%.6g", s) for s in spans), ", ") * ")"
    rate_str = "(" * join(_pretty_rate(D.rate), ", ") * ")"

    # Header
    print(io, span_str, " DomainView @ ", rate_str,
          " with offset ", D.offset, "\n")

    # Indent underlying array for readability
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

# Core constructor
function _make_domainarray(f, T, dims)
    sizes = map(_dim_from_pair, dims)
    rates = map(_rate_from_pair, dims)
    arr   = f(T, sizes...)
    return DomainArray(arr, tuple(rates...))
end

# Public constructors
domainzeros(dims...) = _make_domainarray(zeros, Float64, _normalize_dims(dims...))
domainzeros(T::Type, dims...) = _make_domainarray(zeros, T, _normalize_dims(dims...))

domainones(dims...) = _make_domainarray(ones, Float64, _normalize_dims(dims...))
domainones(T::Type, dims...) = _make_domainarray(ones, T, _normalize_dims(dims...))

domainrand(dims...) = _make_domainarray(rand, Float64, _normalize_dims(dims...))
domainrand(T::Type, dims...) = _make_domainarray(rand, T, _normalize_dims(dims...))

domainfill(value, dims...) = _make_domainarray((T,s...)->fill(value, s...), Float64, _normalize_dims(dims...))
domainfill(T::Type, value, dims...) = _make_domainarray((T,s...)->fill(value, s...), T, _normalize_dims(dims...))

domainfull = domainfill

# Macro support for dimension specs
macro domains(exprs...)
    pairs = map(x -> :(($x, nothing)), exprs)
    return Expr(:tuple, pairs...)
end
