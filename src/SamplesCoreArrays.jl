using Base: @propagate_inbounds, axes, size, eltype, show
using Printf: @sprintf

########################
# Core types           #
########################

abstract type AbstractDomainArray{T,N} <: AbstractArray{T,N} end

struct DomainArray{T,N,A<:AbstractArray{T,N}, R<:NTuple{N,Union{Nothing,Real}}, O<:NTuple{N,Float64}} <: AbstractDomainArray{T,N}
    data::A
    rate::R
    offset::O
end

struct DomainView{T,N,A<:AbstractArray{T,N}, R<:NTuple{N,Union{Nothing,Real}}, O<:NTuple{N,Float64}} <: AbstractDomainArray{T,N}
    data::A
    rate::R
    offset::O
end

@inline DomainArray(data::AbstractArray{T,N}, rate::NTuple{N,Union{Nothing,Real}}) where {T,N} =
    DomainArray(data, rate, ntuple(_ -> 0.0, N))

@inline DomainArray(data::AbstractArray{T,N}) where {T,N} =
    DomainArray(data, ntuple(_ -> nothing, N))

@inline DomainView(data::AbstractArray{T,N}, rate::NTuple{N,Union{Nothing,Real}}) where {T,N} =
    DomainView(data, rate, ntuple(_ -> 0.0, N))

@inline DomainView(data::AbstractArray{T,N}) where {T,N} =
    DomainView(data, ntuple(_ -> nothing, N))

Base.IndexStyle(::Type{<:AbstractDomainArray}) = IndexLinear()
Base.size(D::AbstractDomainArray) = size(D.data)
Base.eltype(::Type{<:AbstractDomainArray{T}}) where T = T

########################
# Domain index & axis  #
########################

abstract type AbstractDomainIndex end

struct DomainIndex{R,CheckRate} <: AbstractDomainIndex
    i::Int
end

struct DomainAxis{R,CheckRate} <: AbstractUnitRange{DomainIndex{R,CheckRate}}
    len::Int
    offset::Float64
end

Base.length(ax::DomainAxis) = ax.len
Base.first(ax::DomainAxis{R,Check}) where {R,Check} = DomainIndex{R,Check}(1)
Base.last(ax::DomainAxis{R,Check})  where {R,Check} = DomainIndex{R,Check}(ax.len)

function Base.iterate(ax::DomainAxis{R,Check}) where {R,Check}
    ax.len == 0 && return nothing
    return (DomainIndex{R,Check}(1), 2)
end

function Base.iterate(ax::DomainAxis{R,Check}, state::Int) where {R,Check}
    state > ax.len && return nothing
    return (DomainIndex{R,Check}(state), state + 1)
end

uncheckrate_axes(ax::DomainAxis{R,true}) where {R} = DomainAxis{R,false}(ax.len, ax.offset)
uncheckrate_axes(ax) = ax
uncheckrate_axes(axs::NTuple{N,Any}) where {N} = ntuple(i -> uncheckrate_axes(axs[i]), N)

macro uncheckrate(ex)
    return :(uncheckrate_axes($(esc(ex))))
end

@inline function _sample_index(idx::DomainIndex{R,Check}, rate::R2) where {R,R2,Check}
    if Check
        rate == R || error("DomainIndex rate mismatch")
    end
    return idx.i
end

########################
# Index classification #
########################

# Sample dims
_to_index_and_offset(i::Integer, ::Nothing) = (i, 0.0)
_to_index_and_offset(::Colon, ::Nothing) = (Colon(), 0.0)
_to_index_and_offset(r::AbstractRange{<:Integer}, ::Nothing) = (r, 0.0)
_to_index_and_offset(v::AbstractVector{<:Integer}, ::Nothing) = (v, 0.0)

# Domain dims
_to_index_and_offset(idx::AbstractDomainIndex, r::Real) = (_sample_index(idx, r), 0.0)
_to_index_and_offset(rng::AbstractRange{<:AbstractDomainIndex}, rate::Real) =
    ([ _sample_index(i, rate) for i in rng ], 0.0)
_to_index_and_offset(v::AbstractVector{<:AbstractDomainIndex}, rate::Real) =
    ([ _sample_index(i, rate) for i in v ], 0.0)
_to_index_and_offset(::Colon, r::Real) = (Colon(), 0.0)
_to_index_and_offset(x::Real, r::Real) =
    error("Real indexing is forbidden on domain dimensions. Use @domainround or DomainIndex.")
_to_index_and_offset(rng::AbstractRange{<:Real}, r::Real) =
    error("Real slicing is forbidden on domain dimensions. Use @domainround or DomainIndex.")

# Treat integer indices as sample indices even on domain dims
_to_index_and_offset(i::Integer, ::Real) = (i, 0.0)
_to_index_and_offset(r::AbstractRange{<:Integer}, ::Real) = (r, 0.0)
_to_index_and_offset(v::AbstractVector{<:Integer}, ::Real) = (v, 0.0)

function _indices_and_offsets(rate, I)
    N = length(I)
    inds = ntuple(i -> nothing, N)
    offs = ntuple(i -> 0.0, N)
    for k in 1:N
        inds_k, offs_k = _to_index_and_offset(I[k], rate[k])
        inds = Base.setindex(inds, inds_k, k)
        offs = Base.setindex(offs, offs_k, k)
    end
    return inds, offs
end

_is_scalar_index(i) = i isa Integer || i isa CartesianIndex || i isa AbstractDomainIndex
_is_scalar_index(::Colon) = false
_is_scalar_index(i::AbstractRange) = false
_is_scalar_index(i::AbstractVector) = false

########################
# Axes: sample vs domain
########################

sampleaxes(D::AbstractDomainArray, d::Int) = axes(D.data, d)
sampleaxes(D::AbstractDomainArray) = ntuple(d -> sampleaxes(D, d), ndims(D))

function domainaxis(D::AbstractDomainArray, d::Int)
    r    = D.rate[d]
    idxs = sampleaxes(D, d)
    off  = D.offset[d]
    if r === nothing
        return idxs
    else
        first_dom = off + (first(idxs) - 1) / r
        return DomainAxis{r,true}(length(idxs), first_dom)
    end
end

domainaxes(D::AbstractDomainArray) = ntuple(d -> domainaxis(D, d), ndims(D))

# Core axes = sample space only
Base.axes(D::AbstractDomainArray, d::Int) = sampleaxes(D, d)
Base.axes(D::AbstractDomainArray) = ntuple(d -> axes(D, d), ndims(D))

########################
# DomainIndex constructors
########################

DomainIndex(t::Real, rate::Real, offset::Real=0) =
    DomainIndex{rate,true}(round(Int, (t - offset) * rate) + 1)

DomainIndex(r::AbstractRange{<:Real}, rate::Real, offset::Real=0) =
    DomainIndex(first(r), rate, offset) : DomainIndex(last(r), rate, offset)

DomainIndex(v::AbstractVector{<:Real}, rate::Real, offset::Real=0) =
    DomainIndex.(v, rate, offset)

DomainIndex(t::Tuple, rate::Real, offset::Real=0) =
    map(x -> DomainIndex(x, rate, offset), t)

DomainIndex(::Colon, rate::Real, offset::Real=0) = Colon()

########################
# DomainIndex arithmetic
########################

Base.isless(a::DomainIndex{R,Check}, b::DomainIndex{R,Check}) where {R,Check} = a.i < b.i
Base.isless(a::DomainIndex{R_a,false}, b::DomainIndex{R_b,false}) where {R_a<:Integer, R_b<:Integer} =
    (a.i * R_b) < (b.i * R_a)

Base.:(==)(a::DomainIndex{R,Check}, b::DomainIndex{R,Check}) where {R,Check} = a.i == b.i
Base.:(==)(a::DomainIndex{R_a,false}, b::DomainIndex{R_b,false}) where {R_a<:Integer, R_b<:Integer} =
    (a.i * R_b) == (b.i * R_a)

Base.:-(a::DomainIndex{R,Check}, b::DomainIndex{R,Check}) where {R,Check} = a.i - b.i
Base.:-(a::DomainIndex{R_a,false}, b::DomainIndex{R_b,false}) where {R_a<:Integer, R_b<:Integer} =
    a.i - ((b.i * R_a) ÷ R_b)

Base.:+(a::DomainIndex{R,Check}, b::DomainIndex{R,Check}) where {R,Check} = a.i + b.i
Base.:+(a::DomainIndex{R_a,false}, b::DomainIndex{R_b,false}) where {R_a<:Integer, R_b<:Integer} =
    a.i + ((b.i * R_a) ÷ R_b)

Base.:+(a::DomainIndex{R,Check}, n::Integer) where {R,Check} =
    DomainIndex{R,Check}(a.i + n)
Base.:+(n::Integer, a::DomainIndex{R,Check}) where {R,Check} =
    DomainIndex{R,Check}(a.i + n)

Base.:-(a::DomainIndex{R,Check}, n::Integer) where {R,Check} =
    DomainIndex{R,Check}(a.i - n)

Base.:-(n::Integer, a::DomainIndex{R,Check}) where {R,Check} =
    DomainIndex{R,Check}(n - a.i)

Base.oneunit(::Type{DomainIndex{R,Check}}) where {R,Check} = DomainIndex{R,Check}(1)
Base.one(::Type{DomainIndex{R,Check}}) where {R,Check} = DomainIndex{R,Check}(1)
Base.zero(::Type{DomainIndex{R,Check}}) where {R,Check} = DomainIndex{R,Check}(0)

########################
# DomainArray indexing  #
########################

@propagate_inbounds function Base.getindex(D::AbstractDomainArray{T,N},
                                           I::Vararg{Union{Int,
                                                           AbstractDomainIndex,
                                                           Colon,
                                                           AbstractRange,
                                                           AbstractVector},N}) where {T,N}
    inds, _ = _indices_and_offsets(D.rate, I)

    if all(_is_scalar_index, I)
        return D.data[inds...]
    else
        return domainslice(D, I...)
    end
end

_is_kept_index(idx) = idx isa AbstractRange || idx isa Colon

function _slice_metadata(rate, offset, offs, inds)
    new_rate   = ()
    new_offset = ()
    for (d, idx) in enumerate(inds)
        if _is_kept_index(idx)
            new_rate   = (new_rate...,   rate[d])
            new_offset = (new_offset..., offset[d] + offs[d])
        end
    end
    return new_rate, new_offset
end

function domainslice(D::DomainArray{T,N}, I::Vararg{Any,N}) where {T,N}
    inds, offs = _indices_and_offsets(D.rate, I)
    new_data   = D.data[inds...]
    new_rate, new_offset = _slice_metadata(D.rate, D.offset, offs, I)
    return DomainArray(new_data, new_rate, new_offset)
end

function domainview(D::DomainArray{T,N}, I::Vararg{Any,N}) where {T,N}
    inds, offs = _indices_and_offsets(D.rate, I)
    v = @view D.data[inds...]
    offset = ntuple(d -> D.offset[d] + offs[d], N)
    return DomainView{T,N,typeof(v),typeof(D.rate),typeof(offset)}(v, D.rate, offset)
end

function domainview(D::DomainView{T,N}, I::Vararg{Any,N}) where {T,N}
    inds, offs = _indices_and_offsets(D.rate, I)
    v = @view D.data[inds...]
    offset = ntuple(d -> D.offset[d] + offs[d], N)
    return DomainView{T,N,typeof(v),typeof(D.rate),typeof(offset)}(v, D.rate, offset)
end

Base.view(D::AbstractDomainArray{T,N}, I::Vararg{Any,N}) where {T,N} = domainview(D, I...)

########################
# Macros: sample/domain #
########################

macro sampleindex(ex)
    @assert ex.head === :ref "@sampleindex must wrap A[...]"
    arr = ex.args[1]
    idxs = ex.args[2:end]
    newidxs = Any[]
    for (k, idx) in enumerate(idxs)
        push!(newidxs, :( _sampleindex($(esc(arr)), $k, $(esc(idx))) ))
    end
    return :( $(esc(arr))[$(newidxs...)] )
end

function _sampleindex(A::AbstractDomainArray, dim::Int, idx)
    r = A.rate[dim]
    if r === nothing
        return idx
    end
    if idx isa Integer
        return DomainIndex{typeof(r),false}(idx)
    elseif idx isa AbstractRange{<:Integer}
        return [DomainIndex{typeof(r),false}(i) for i in idx]
    elseif idx isa Colon
        return Colon()
    else
        error("@sampleindex only supports integer indices on domain dimensions")
    end
end

macro domainround(ex)
    @assert ex.head === :ref "@domainround must wrap A[...]"
    arr = ex.args[1]
    idxs = ex.args[2:end]
    newidxs = Any[]
    for (k, idx) in enumerate(idxs)
        push!(newidxs, :( _domainround_index($(esc(arr)), $k, $(esc(idx))) ))
    end
    return :( $(esc(arr))[$(newidxs...)] )
end

function _domainround_index(A::AbstractDomainArray, dim::Int, idx)
    r   = A.rate[dim]
    off = A.offset[dim]
    if r === nothing
        return idx
    end
    if idx isa Real
        return DomainIndex(idx, r, off)
    elseif idx isa AbstractRange{<:Real}
        return DomainIndex(idx, r, off)
    elseif idx isa AbstractVector{<:Real}
        return DomainIndex(idx, r, off)
    elseif idx isa Colon
        return Colon()
    elseif idx isa AbstractDomainIndex
        return idx
    elseif idx isa AbstractRange{<:AbstractDomainIndex} || idx isa AbstractVector{<:AbstractDomainIndex}
        return idx
    else
        error("@domainround: unsupported index type for domain dimension")
    end
end

########################
# Domain shifting       #
########################

function shiftdomain(D::DomainArray, shifts::Vararg{Real})
    @assert length(shifts) == length(D.rate)
    newoffset = ntuple(i -> D.rate[i] === nothing ? D.offset[i] : D.offset[i] + shifts[i],
                       length(D.rate))
    DomainArray(D.data, D.rate, newoffset)
end

shiftdomain(D::DomainArray, shift::Real) = shiftdomain(D, (shift,))

function shiftdomain(D::DomainView, shifts::Vararg{Real})
    @assert length(shifts) == length(D.rate)
    newoffset = ntuple(i -> D.rate[i] === nothing ? D.offset[i] : D.offset[i] + shifts[i],
                       length(D.rate))
    DomainView(D.data, D.rate, newoffset)
end

shiftdomain(D::DomainView, shift::Real) = shiftdomain(D, (shift,))

########################
# Binary ops & broadcast
########################

_rates_match(a::AbstractDomainArray, b::AbstractDomainArray) = a.rate == b.rate
_offsets_match(a::AbstractDomainArray, b::AbstractDomainArray) = a.offset == b.offset

_unwrap(x::AbstractDomainArray) = x.data
_unwrap(x) = x

function _binary_op(op, A::AbstractDomainArray, B::AbstractDomainArray)
    _rates_match(A, B)   || error("Domain rates do not match")
    _offsets_match(A, B) || error("Domain offsets do not match")
    data = op.(_unwrap(A), _unwrap(B))
    return DomainArray(data, A.rate, A.offset)
end

Base.:+(A::AbstractDomainArray, B::AbstractDomainArray) = _binary_op(+, A, B)
Base.:-(A::AbstractDomainArray, B::AbstractDomainArray) = _binary_op(-, A, B)
Base.:*(A::AbstractDomainArray, B::AbstractDomainArray) = _binary_op(*, A, B)
Base.:/(A::AbstractDomainArray, B::AbstractDomainArray) = _binary_op(/, A, B)

Base.:+(A::AbstractDomainArray, x::Number) = DomainArray(A.data .+ x, A.rate, A.offset)
Base.:+(x::Number, A::AbstractDomainArray) = A + x

Base.:-(A::AbstractDomainArray, x::Number) = DomainArray(A.data .- x, A.rate, A.offset)
Base.:-(x::Number, A::AbstractDomainArray) = DomainArray(x .- A.data, A.rate, A.offset)

Base.:*(A::AbstractDomainArray, x::Number) = DomainArray(A.data .* x, A.rate, A.offset)
Base.:*(x::Number, A::AbstractDomainArray) = A * x

Base.:/(A::AbstractDomainArray, x::Number) = DomainArray(A.data ./ x, A.rate, A.offset)

function Base.broadcast(f, A::AbstractDomainArray, Bs...)
    rate = A.rate
    offset = A.offset
    for B in Bs
        if B isa AbstractDomainArray
            _rates_match(A, B)   || error("Domain rates do not match")
            _offsets_match(A, B) || error("Domain offsets do not match")
        end
    end
    data = Base.broadcast(f, _unwrap(A), map(_unwrap, Bs)...)
    return DomainArray(data, rate, offset)
end

########################
# similar / constructors
########################

function Base.similar(D::AbstractDomainArray, ::Type{T}, dims::Dims) where {T}
    DomainArray(similar(D.data, T, dims), D.rate, D.offset)
end

Base.similar(D::AbstractDomainArray) = DomainArray(similar(D.data), D.rate, D.offset)

function _span(D::AbstractDomainArray)
    dims  = size(D)
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
    spans    = _span(D)
    span_str = "(" * join((@sprintf("%.6g", s) for s in spans), ", ") * ")"
    rate_str = "(" * join(_pretty_rate(D.rate), ", ") * ")"

    print(io, span_str, " DomainArray @ ", rate_str,
          " with offset ", D.offset, "\n")

    inner = IOContext(io, :compact => true)
    Base.show(inner, MIME"text/plain"(), D.data)
end

function Base.show(io::IO, ::MIME"text/plain", D::DomainView)
    spans    = _span(D)
    span_str = "(" * join((@sprintf("%.6g", s) for s in spans), ", ") * ")"
    rate_str = "(" * join(_pretty_rate(D.rate), ", ") * ")"

    print(io, span_str, " DomainView @ ", rate_str,
          " with offset ", D.offset, "\n")

    inner = IOContext(io, :compact => true)
    Base.show(inner, MIME"text/plain"(), D.data)
end

########################
# Dimension conversion  #
########################

_to_triple(t::Tuple{<:Real,<:Real,<:Real}) = (t[1], t[2], t[3])
_to_triple(t::Tuple{<:Real,<:Real})        = (t[1], t[2], 0.0)
_to_triple(len::Integer)                   = (len, nothing, 0.0)
_to_triple(t) = error("Invalid dimension specification: $t")

_dim_from_triple((len, rate, _)) = begin
    if rate === nothing
        len isa Integer || error("Integer length required when rate is nothing")
        return len
    else
        return Int(round(len * rate))
    end
end

_rate_from_triple((_, rate, _))  = rate
_off_from_triple((_, _, off))    = off

function _make_domainarray(f, T, dims::Tuple)
    triples = map(_to_triple, dims)
    sizes   = map(_dim_from_triple, triples)
    rates   = map(_rate_from_triple, triples)
    offs    = map(_off_from_triple, triples)
    arr     = f(T, sizes...)
    return DomainArray(arr, tuple(rates...), tuple(offs...))
end

########################
# Public constructors  #
########################

domainzeros(dims...) = _make_domainarray(zeros, Float64, dims)
domainzeros(T::Type, dims...) = _make_domainarray(zeros, T, dims)

domainones(dims...) = _make_domainarray(ones, Float64, dims)
domainones(T::Type, dims...) = _make_domainarray(ones, T, dims)

domainrand(dims...) = _make_domainarray(rand, Float64, dims)
domainrand(T::Type, dims...) = _make_domainarray(rand, T, dims)

domainfill(value, dims...) = _make_domainarray((T,s...)->fill(value, s...), Float64, dims)
domainfill(T::Type, value, dims...) = _make_domainarray((T,s...)->fill(value, s...), T, dims)

domainfull = domainfill

########################
# DomainAxis show      #
########################

function Base.show(io::IO, ax::DomainAxis{R,Check}) where {R,Check}
    first_t = ax.offset
    last_t  = ax.offset + (ax.len - 1) / R
    step_t  = 1 / R
    print(io, "$(first_t):$(step_t):$(last_t) @$(R)")
end

########################
# DomainIndex fallback #
########################

# On normal arrays, DomainIndex behaves like its integer
Base.getindex(A::AbstractArray, i::DomainIndex) = A[i.i]

Base.getindex(A::AbstractArray, r::AbstractRange{<:DomainIndex}) =
    A[[idx.i for idx in r]]

Base.getindex(A::AbstractArray, v::AbstractVector{<:DomainIndex}) =
    A[[idx.i for idx in v]]
