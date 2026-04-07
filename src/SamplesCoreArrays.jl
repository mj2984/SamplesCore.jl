export SampleArray, SampleView, timeslice, timeview, timeview_extreme, @extreme_view, @sampleidx
using Base: size, axes, IndexStyle, getindex, setindex!, @propagate_inbounds, show

# ────────────────────────────────────────────────────────────────────────────────
# TYPE DEFINITIONS
# ────────────────────────────────────────────────────────────────────────────────

"""
    SampleArray{T,N,A,R}

A wrapper around an `AbstractArray{T,N}` with per‑dimension sampling rates.

- `A` is the underlying array type
- `R` is `NTuple{N,Union{Nothing,Real}}`
  - `Real` → this dimension is time‑indexed
  - `Nothing` → this dimension is sample‑indexed
"""
struct SampleArray{T,N,A<:AbstractArray{T,N},R<:NTuple{N,Union{Nothing,Real}}} <: AbstractArray{T,N}
    sample::A
    rate::R
end

Base.IndexStyle(::Type{<:SampleArray}) = IndexStyle(A) where {T,N,A,R}
Base.size(S::SampleArray) = size(S.sample)
Base.axes(S::SampleArray) = axes(S.sample)
Base.eltype(::Type{SampleArray{T}}) where {T} = T


"""
    SampleView{T,N,A,R,O}

A view of a `SampleArray` that:

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

Base.IndexStyle(::Type{<:SampleView}) = IndexStyle(A) where {T,N,A,R,O}
Base.size(S::SampleView) = size(S.sample)
Base.axes(S::SampleView) = axes(S.sample)
Base.eltype(::Type{SampleView{T}}) where {T} = T


# ────────────────────────────────────────────────────────────────────────────────
# PRETTY PRINTING
# ────────────────────────────────────────────────────────────────────────────────

function Base.show(io::IO, S::SampleArray)
    print(io, "SampleArray(rate=", S.rate, ", size=", size(S), ")")
end

function Base.show(io::IO, S::SampleView)
    print(io, "SampleView(rate=", S.rate,
          ", offset=", S.offset,
          ", size=", size(S), ")")
end


# ────────────────────────────────────────────────────────────────────────────────
# ROUNDING MODES
# ────────────────────────────────────────────────────────────────────────────────

abstract type TimeRounding end
struct DefaultRounding <: TimeRounding end
struct ExtremeRounding <: TimeRounding end

const _default_rounding = DefaultRounding()


# ────────────────────────────────────────────────────────────────────────────────
# INDEX CONVERSION HELPERS
# ────────────────────────────────────────────────────────────────────────────────

_to_index_and_offset(idx, ::Nothing, ::TimeRounding) = (idx, 0.0)

function _to_index_and_offset(t::Real, r::Real, ::DefaultRounding)
    i = round(Int, t * r)
    t1 = (i - 1) / r
    return (i, t1 - t)
end

function _to_index_and_offset(t::Real, r::Real, ::ExtremeRounding)
    i = round(Int, t * r)
    t1 = (i - 1) / r
    return (i, t1 - t)
end

function _to_index_and_offset(tr::AbstractRange{<:Real}, r::Real, ::DefaultRounding)
    lo = first(tr); hi = last(tr)
    i1 = ceil(Int, lo * r)
    i2 = floor(Int, hi * r)
    t1 = (i1 - 1) / r
    return (i1:i2, t1 - lo)
end

function _to_index_and_offset(tr::AbstractRange{<:Real}, r::Real, ::ExtremeRounding)
    lo = first(tr); hi = last(tr)
    i1 = floor(Int, lo * r)
    i2 = ceil(Int, hi * r)
    t1 = (i1 - 1) / r
    return (i1:i2, t1 - lo)
end

_to_index_and_offset(i::Integer, r, ::TimeRounding) = (i, 0.0)
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


# ────────────────────────────────────────────────────────────────────────────────
# TIME-DOMAIN INDEXING
# ────────────────────────────────────────────────────────────────────────────────

@propagate_inbounds function Base.getindex(S::SampleArray{T,N}, I::Vararg{Any,N}) where {T,N}
    inds, _ = _indices_and_offsets(S.rate, I, _default_rounding)
    return getindex(S.sample, inds...)
end

@propagate_inbounds function Base.getindex(S::SampleView{T,N}, I::Vararg{Any,N}) where {T,N}
    inds, _ = _indices_and_offsets(S.rate, I, _default_rounding)
    return getindex(S.sample, inds...)
end


# ────────────────────────────────────────────────────────────────────────────────
# SETINDEX! SUPPORT
# ────────────────────────────────────────────────────────────────────────────────

@propagate_inbounds function Base.setindex!(S::SampleArray{T,N}, v, I::Vararg{Any,N}) where {T,N}
    inds, _ = _indices_and_offsets(S.rate, I, _default_rounding)
    return setindex!(S.sample, v, inds...)
end

@propagate_inbounds function Base.setindex!(S::SampleView{T,N}, v, I::Vararg{Any,N}) where {T,N}
    inds, _ = _indices_and_offsets(S.rate, I, _default_rounding)
    return setindex!(S.sample, v, inds...)
end


# ────────────────────────────────────────────────────────────────────────────────
# SLICE (COPY) AND VIEW (OFFSET-TRACKING)
# ────────────────────────────────────────────────────────────────────────────────

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


# ────────────────────────────────────────────────────────────────────────────────
# EXTREME VIEW (floor lower, ceil upper)
# ────────────────────────────────────────────────────────────────────────────────

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


# ────────────────────────────────────────────────────────────────────────────────
# SAMPLE-DOMAIN INDEXING MACRO
# ────────────────────────────────────────────────────────────────────────────────

macro sampleidx(ex)
    if ex isa Expr && ex.head === :ref
        arr = ex.args[1]
        idxs = ex.args[2:end]
        return :( $(arr).sample[$(idxs...)] )
    else
        error("@sampleidx expects A[...]")
    end
end
