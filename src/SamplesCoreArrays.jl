abstract type AbstractSampleArray{T,N} <: AbstractArray{T,N} end

"""
    SampleArray{T,N,A,R}

Wrapper around an `AbstractArray{T,N}` with per-dimension sampling rates.

- `A` is the underlying array type
- `R` is `NTuple{N,Union{Nothing,Real}}`
  - `Real`   → this dimension is time-indexed
  - `Nothing` → this dimension is sample-indexed
"""
struct SampleArray{T,N,A<:AbstractArray{T,N},R<:NTuple{N,Union{Nothing,Real}}} <: AbstractSampleArray{T,N}
    sample::A
    rate::R
end

"""
    SampleView{T,N,A,R,O}

View of a `SampleArray` that:

- wraps a `SubArray`
- preserves sampling rates
- tracks accumulated time offsets per dimension
"""
struct SampleView{T,N,A<:AbstractArray{T,N},
                  R<:NTuple{N,Union{Nothing,Real}},
                  O<:NTuple{N,Float64}} <: AbstractSampleArray{T,N}
    sample::A
    rate::R
    offset::O
end

Base.IndexStyle(::Type{<:AbstractSampleArray}) = IndexStyle(Array)
Base.size(S::AbstractSampleArray) = size(S.sample)
Base.axes(S::AbstractSampleArray) = axes(S.sample)
Base.eltype(::Type{<:AbstractSampleArray{T}}) where T = T

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

@propagate_inbounds function Base.getindex(S::AbstractSampleArray{T,N}, I::Vararg{Any,N}) where {T,N}
    inds, _ = _indices_and_offsets(S.rate, I, _default_rounding)

    if all(_is_scalar_index, I)
        return S.sample[inds...]  # element
    else
        new_sample = S.sample[inds...]
        return SampleArray(new_sample, S.rate)
    end
end

@propagate_inbounds function Base.setindex!(S::SampleArray{T,N}, v, I::Vararg{Any,N}) where {T,N}
    inds, _ = _indices_and_offsets(S.rate, I, _default_rounding)
    return setindex!(S.sample, v, inds...)
end

@propagate_inbounds function Base.setindex!(S::SampleView{T,N}, v, I::Vararg{Any,N}) where {T,N}
    inds, _ = _indices_and_offsets(S.rate, I, _default_rounding)
    return setindex!(S.sample, v, inds...)
end

function timeslice(S::SampleArray{T,N}, I::Vararg{Any,N}) where {T,N}
    inds, _ = _indices_and_offsets(S.rate, I, _default_rounding)
    new_sample = S.sample[inds...]
    return SampleArray(new_sample, S.rate)
end

function timeview(S::SampleArray{T,N}, I::Vararg{Any,N}) where {T,N}
    inds, offs = _indices_and_offsets(S.rate, I, _default_rounding)
    v = @view S.sample[inds...]
    offset = ntuple(d -> offs[d], N)
    return SampleView{T,N,typeof(v),typeof(S.rate),typeof(offset)}(v, S.rate, offset)
end

function timeview(S::SampleView{T,N}, I::Vararg{Any,N}) where {T,N}
    inds, offs = _indices_and_offsets(S.rate, I, _default_rounding)
    v = @view S.sample[inds...]
    offset = ntuple(d -> S.offset[d] + offs[d], N)
    return SampleView{T,N,typeof(v),typeof(S.rate),typeof(offset)}(v, S.rate, offset)
end

function timeview_extreme(S::SampleArray{T,N}, I::Vararg{Any,N}) where {T,N}
    inds, offs = _indices_and_offsets(S.rate, I, ExtremeRounding())
    v = @view S.sample[inds...]
    offset = ntuple(d -> offs[d], N)
    return SampleView{T,N,typeof(v),typeof(S.rate),typeof(offset)}(v, S.rate, offset)
end

macro extreme_view(ex)
    @assert ex.head === :call
    fn = ex.args[1]
    if fn === :timeview
        return :(timeview_extreme($(ex.args[2:end]...)))
    else
        error("@extreme_view only wraps `timeview`")
    end
end

"""
    @sampleidx A[...]
Rewrite `A[...]` to `A.sample[...]` for explicit sample-domain indexing.
"""
macro sampleidx(ex)
    return _rewrite_sampleidx(ex)
end

function _rewrite_sampleidx(ex)
    if ex isa Expr && ex.head === :ref
        arr = ex.args[1]
        idxs = ex.args[2:end]
        return :( $(esc(arr)).sample[$(map(esc, idxs)...) ] )
    end
    if ex isa Expr
        newargs = map(_rewrite_sampleidx, ex.args)
        return Expr(ex.head, newargs...)
    end
    return ex
end

@propagate_inbounds function Base.view(S::AbstractSampleArray{T,N}, I::Vararg{Any,N}) where {T,N}
    return timeview(S, I...)
end

struct Samples
    value::Int
end
_to_index_and_offset(s::Samples, r, rounding) = (s.value, 0.0)

function shiftaxis(S::SampleArray, shifts::Vararg{Real})
    @assert length(shifts) == length(S.rate)
    newoffset = ntuple(i -> S.rate[i] === nothing ? 0.0 : shifts[i], length(S.rate))
    SampleView(S.sample, S.rate, newoffset)
end
shiftaxis(S::SampleArray, shift::Real) = shiftaxis(S, (shift,))

function shiftaxis(S::SampleView, shifts::Vararg{Real})
    @assert length(shifts) == length(S.rate)
    newoffset = ntuple(i -> S.offset[i] + (S.rate[i] === nothing ? 0.0 : shifts[i]),
                       length(S.rate))
    SampleView(S.sample, S.rate, newoffset)
end
shiftaxis(S::SampleView, shift::Real) = shiftaxis(S, (shift,))

# rate compatibility
_rates_match(a::AbstractSampleArray, b::AbstractSampleArray) = a.rate == b.rate

# unwrap underlying storage or pass through
_unwrap(x::AbstractSampleArray) = x.sample
_unwrap(x) = x

# binary op with rate check, returns SampleArray
function _binary_op(op, A::AbstractSampleArray, B::AbstractSampleArray)
    _rates_match(A, B) || error("Sample rates do not match")
    sample = op.(_unwrap(A), _unwrap(B))
    return SampleArray(sample, A.rate)
end

# basic arithmetic between two sample arrays
Base.:+(A::AbstractSampleArray, B::AbstractSampleArray) = _binary_op(+, A, B)
Base.:-(A::AbstractSampleArray, B::AbstractSampleArray) = _binary_op(-, A, B)
Base.:*(A::AbstractSampleArray, B::AbstractSampleArray) = _binary_op(*, A, B)
Base.:/(A::AbstractSampleArray, B::AbstractSampleArray) = _binary_op(/, A, B)

# scalar operations (use broadcast)
Base.:+(A::AbstractSampleArray, x::Number) = SampleArray(A.sample .+ x, A.rate)
Base.:+(x::Number, A::AbstractSampleArray) = A + x

Base.:-(A::AbstractSampleArray, x::Number) = SampleArray(A.sample .- x, A.rate)
Base.:-(x::Number, A::AbstractSampleArray) = SampleArray(x .- A.sample, A.rate)

Base.:*(A::AbstractSampleArray, x::Number) = SampleArray(A.sample .* x, A.rate)
Base.:*(x::Number, A::AbstractSampleArray) = A * x

Base.:/(A::AbstractSampleArray, x::Number) = SampleArray(A.sample ./ x, A.rate)

# generic broadcast that preserves SampleArray and checks rates
function Base.broadcast(f, A::AbstractSampleArray, Bs...)
    rate = A.rate
    # check other sample arrays for rate compatibility
    for B in Bs
        if B isa AbstractSampleArray
            _rates_match(A, B) || error("Sample rates do not match")
        end
    end
    sample = Base.broadcast(f, _unwrap(A), map(_unwrap, Bs)...)
    return SampleArray(sample, rate)
end

# similar: preserve rate, wrap in SampleArray
function Base.similar(S::AbstractSampleArray, ::Type{T}, dims::Dims) where {T}
    SampleArray(similar(S.sample, T, dims), S.rate)
end

Base.similar(S::AbstractSampleArray) = SampleArray(similar(S.sample), S.rate)

# promotion rule between two sample arrays (for element type)
Base.promote_rule(::Type{SA}, ::Type{SB}) where {T,S,N,
                                                SA<:AbstractSampleArray{T,N},
                                                SB<:AbstractSampleArray{S,N}} =
    AbstractSampleArray{promote_type(T,S),N}

# Format sampling rates: Real → itself, Nothing → "_"
function _pretty_rate(rate::NTuple{N,Union{Nothing,Real}}) where {N}
    out = Vector{Any}(undef, N)
    for i in 1:N
        r = rate[i]
        out[i] = r === nothing ? "_" : r
    end
    return tuple(out...)
end

# Compute span for each dimension (size/rate)
function _span(S)
    dims = size(S)
    rates = S.rate
    spans = Vector{Float64}(undef, length(dims))
    for i in eachindex(dims)
        r = rates[i]
        spans[i] = r === nothing ? dims[i] : dims[i] / r
    end
    return spans
end

function Base.show(io::IO, ::MIME"text/plain", S::SampleArray)
    spans = _span(S)
    span_str = "(" * join((@sprintf("%.6g", s) for s in spans), ", ") * ")"
    rate_str = "(" * join(_pretty_rate(S.rate), ", ") * ")"

    # Header
    print(io, span_str, " SampleArray @ ", rate_str, "\n")

    # Indent underlying array for readability
    inner = IOContext(io, :compact => true)
    Base.show(inner, MIME"text/plain"(), S.sample)
end

function Base.show(io::IO, ::MIME"text/plain", S::SampleView)
    spans = _span(S)
    span_str = "(" * join((@sprintf("%.6g", s) for s in spans), ", ") * ")"
    rate_str = "(" * join(_pretty_rate(S.rate), ", ") * ")"

    # Header
    print(io, span_str, " SampleView @ ", rate_str,
          " with offset ", S.offset, "\n")

    # Indent underlying array for readability
    inner = IOContext(io, :compact => true)
    Base.show(inner, MIME"text/plain"(), S.sample)
end

# Dimension conversion
_to_pair(x::Tuple{<:Real,<:Union{Real,Nothing}}) = x
_to_pair(x::Real) = (x, nothing)

_normalize_dims(dims::Vararg{Any}) = map(_to_pair, dims)
_normalize_dims(dims::Tuple) = map(_to_pair, dims)

_dim_from_pair((len, rate)::Tuple{<:Real,<:Real}) = Int(round(len * rate))
_dim_from_pair((len, rate)::Tuple{<:Integer,Nothing}) = len
_dim_from_pair((len, rate)::Tuple{<:Real,Nothing}) = error("Real-valued dimension requires a sampling rate")
_dim_from_pair((len, rate)::Tuple{<:Integer,<:Real}) = error("Integer dimension with a real sampling rate is ambiguous")

_rate_from_pair((_, rate)) = rate

# Core constructor
function _make_samplearray(f, T, dims)
    sizes = map(_dim_from_pair, dims)
    rates = map(_rate_from_pair, dims)
    arr   = f(T, sizes...)
    return SampleArray(arr, tuple(rates...))
end

# Public constructors
samplezeros(dims...) = _make_samplearray(zeros, Float64, _normalize_dims(dims...))
samplezeros(T::Type, dims...) = _make_samplearray(zeros, T, _normalize_dims(dims...))

sampleones(dims...) = _make_samplearray(ones, Float64, _normalize_dims(dims...))
sampleones(T::Type, dims...) = _make_samplearray(ones, T, _normalize_dims(dims...))

samplerand(dims...) = _make_samplearray(rand, Float64, _normalize_dims(dims...))
samplerand(T::Type, dims...) = _make_samplearray(rand, T, _normalize_dims(dims...))

samplefill(value, dims...) = _make_samplearray((T,s...)->fill(value, s...), Float64, _normalize_dims(dims...))
samplefill(T::Type, value, dims...) = _make_samplearray((T,s...)->fill(value, s...), T, _normalize_dims(dims...))

samplefull = samplefill

# Macro support
macro samples(exprs...)
    pairs = map(x -> :(($x, nothing)), exprs)
    return Expr(:tuple, pairs...)
end

# Domain-space utilities
function domainaxis(S::AbstractSampleArray, d::Int)
    r = S.rate[d]
    idxs = axes(S, d)
    if r === nothing
        return idxs
    else
        return (first(idxs)-1)/r : 1/r : (last(idxs)-1)/r
    end
end

domainaxes(S::AbstractSampleArray) =
    ntuple(d -> domainaxis(S, d), ndims(S))

function domainsize(S::AbstractSampleArray)
    ntuple(d -> begin
        ax = domainaxis(S, d)
        if isa(ax, AbstractRange)
            return last(ax) - first(ax)
        else
            return length(ax)
        end
    end, ndims(S))
end

function domainrange(S::AbstractSampleArray, d::Int)
    ax = domainaxis(S, d)
    return (first(ax), last(ax))
end

domainrange(S::AbstractSampleArray) =
    ntuple(d -> domainrange(S, d), ndims(S))

function domainextent(S::AbstractSampleArray)
    ntuple(d -> begin
        lo, hi = domainrange(S, d)
        hi - lo
    end, ndims(S))
end
