"""
    SampleArray{T,N,A,R}

Wrapper around an `AbstractArray{T,N}` with per-dimension sampling rates.

- `A` is the underlying array type
- `R` is `NTuple{N,Union{Nothing,Real}}`
  - `Real`   → this dimension is time-indexed
  - `Nothing` → this dimension is sample-indexed
"""
struct SampleArray{T,N,A<:AbstractArray{T,N},R<:NTuple{N,Union{Nothing,Real}}} <: AbstractArray{T,N}
    sample::A
    rate::R
end

Base.IndexStyle(::Type{SampleArray{T,N,A,R}}) where {T,N,A,R} = IndexStyle(A)
Base.size(S::SampleArray) = size(S.sample)
Base.axes(S::SampleArray) = axes(S.sample)
Base.eltype(::Type{SampleArray{T}}) where {T} = T

"""
    SampleView{T,N,A,R,O}

View of a `SampleArray` that:

- wraps a `SubArray`
- preserves sampling rates
- tracks accumulated time offsets per dimension
"""
struct SampleView{T,N,A<:AbstractArray{T,N},
                  R<:NTuple{N,Union{Nothing,Real}},
                  O<:NTuple{N,Float64}} <: AbstractArray{T,N}
    sample::A
    rate::R
    offset::O
end

Base.IndexStyle(::Type{SampleView{T,N,A,R,O}}) where {T,N,A,R,O} = IndexStyle(A)
Base.size(S::SampleView) = size(S.sample)
Base.axes(S::SampleView) = axes(S.sample)
Base.eltype(::Type{SampleView{T}}) where {T} = T

# ROUNDING MODES
abstract type TimeRounding end
struct DefaultRounding <: TimeRounding end
struct ExtremeRounding <: TimeRounding end

const _default_rounding = DefaultRounding()

# INDEX CONVERSION HELPERS

# Sample-indexed dimension: pass through
_to_index_and_offset(idx, ::Nothing, ::TimeRounding) = (idx, 0.0)

# Scalar time → sample index (default rounding)
function _to_index_and_offset(t::Real, r::Real, ::DefaultRounding)
    raw = floor(Int, t * r)
    i = max(raw, 1)                 # clamp to 1 for t ≤ 0
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
    i1 = ceil(Int, lo * r) + 1      # 0.0 → 1
    i2 = floor(Int, hi * r)         # inclusive upper
    i2 = max(i2, i1)                # avoid empty / inverted ranges
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

# TIME-DOMAIN INDEXING

@propagate_inbounds function Base.getindex(S::SampleArray{T,N}, I::Vararg{Any,N}) where {T,N}
    inds, _ = _indices_and_offsets(S.rate, I, _default_rounding)
    return getindex(S.sample, inds...)
end

@propagate_inbounds function Base.getindex(S::SampleView{T,N}, I::Vararg{Any,N}) where {T,N}
    inds, _ = _indices_and_offsets(S.rate, I, _default_rounding)
    return getindex(S.sample, inds...)
end

# SETINDEX! SUPPORT

@propagate_inbounds function Base.setindex!(S::SampleArray{T,N}, v, I::Vararg{Any,N}) where {T,N}
    inds, _ = _indices_and_offsets(S.rate, I, _default_rounding)
    return setindex!(S.sample, v, inds...)
end

@propagate_inbounds function Base.setindex!(S::SampleView{T,N}, v, I::Vararg{Any,N}) where {T,N}
    inds, _ = _indices_and_offsets(S.rate, I, _default_rounding)
    return setindex!(S.sample, v, inds...)
end

# SLICE (COPY) AND VIEW (OFFSET-TRACKING)

"""
    timeslice(S, I...)

Copy-based slicing in time domain. Returns a new `SampleArray`.
"""
function timeslice(S::SampleArray{T,N}, I::Vararg{Any,N}) where {T,N}
    inds, _ = _indices_and_offsets(S.rate, I, _default_rounding)
    new_sample = S.sample[inds...]
    return SampleArray(new_sample, S.rate)
end

"""
    timeview(S, I...)

View-based slicing with offset accumulation. Returns a `SampleView`.
"""
function timeview(S::SampleArray{T,N}, I::Vararg{Any,N}) where {T,N}
    inds, offs = _indices_and_offsets(S.rate, I, _default_rounding)
    v = @view S.sample[inds...]
    offset = ntuple(d -> offs[d], N)
    return SampleView{T,N,typeof(v),typeof(S.rate),typeof(offset)}(v, S.rate, offset)
end

# Cascaded views: accumulate offsets
function timeview(S::SampleView{T,N}, I::Vararg{Any,N}) where {T,N}
    inds, offs = _indices_and_offsets(S.rate, I, _default_rounding)
    v = @view S.sample[inds...]
    offset = ntuple(d -> S.offset[d] + offs[d], N)
    return SampleView{T,N,typeof(v),typeof(S.rate),typeof(offset)}(v, S.rate, offset)
end

# EXTREME VIEW (floor lower, ceil upper)
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

# SAMPLE-DOMAIN INDEXING MACRO
"""
    @sampleidx A[...]
Rewrite `A[...]` to `A.sample[...]` for explicit sample-domain indexing.
"""
macro sampleidx(ex)
    return _rewrite_sampleidx(ex)
end

function _rewrite_sampleidx(ex)
    # Case 1: A[...] → :(A.sample[...])
    if ex isa Expr && ex.head === :ref
        arr = ex.args[1]
        idxs = ex.args[2:end]
        return :( $(esc(arr)).sample[$(map(esc, idxs)...) ] )
    end

    # Case 2: recursively rewrite inside expressions
    if ex isa Expr
        newargs = map(_rewrite_sampleidx, ex.args)
        return Expr(ex.head, newargs...)
    end

    # Case 3: literals, symbols, etc.
    return ex
end

@propagate_inbounds function Base.view(S::SampleArray{T,N}, I::Vararg{Any,N}) where {T,N}
    inds, _ = _indices_and_offsets(S.rate, I, _default_rounding)
    return @view S.sample[inds...]
end

@propagate_inbounds function Base.view(S::SampleView{T,N}, I::Vararg{Any,N}) where {T,N}
    inds, _ = _indices_and_offsets(S.rate, I, _default_rounding)
    return @view S.sample[inds...]
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
        if r === nothing
            spans[i] = dims[i]          # raw sample count
        else
            spans[i] = dims[i] / r      # scaled span
        end
    end
    return spans
end

function Base.show(io::IO, ::MIME"text/plain", S::SampleArray)
    spans = _span(S)
    rates = _pretty_rate(S.rate)

    # Format spans as "(0.5, 10)" etc.
    span_str = "(" * join((@sprintf("%.6g", s) for s in spans), ", ") * ")"

    print(io,
        span_str, " at rates ", rates,
        " (SampleArray of ", typeof(S.sample), ")\n"
    )

    Base.show(io, MIME"text/plain"(), S.sample)
end

function Base.show(io::IO, ::MIME"text/plain", S::SampleView)
    spans = _span(S)
    rates = _pretty_rate(S.rate)

    span_str = "(" * join((@sprintf("%.6g", s) for s in spans), ", ") * ")"

    print(io,
        span_str, " at rates ", rates,
        " (SampleView of ", typeof(S.sample), ")",
        " with offset ", S.offset, "\n"
    )

    Base.show(io, MIME"text/plain"(), S.sample)
end
