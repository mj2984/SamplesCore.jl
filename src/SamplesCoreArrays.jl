#module DomainArrays

export RelativeOrigin, relativeorigin, TypedOrigin, DomainOffsetTypes,
       SampleSpace, samplespace, TypedDomainSpace, GenericDomainRateTypes,
       DomainOnly, domainonly, ToSampleSpace, ToDomainSpace, to_sample_space, to_domain_space,
       AbstractDomainArray, DomainArray, DomainIndex, DomainAxis,
       domainaxes, domainmap, domainzeros, relativeorigin, rate, interpret_rate, interpret_index, interpret_origin

struct RelativeOrigin end
struct TypedOrigin{O} end
struct SampleSpace end
struct TypedDomainSpace{D} end
struct DomainOnly end
struct ToSampleSpace end
struct ToDomainSpace end
abstract type AbstractDomainArray{T,N} <: AbstractArray{T,N} end

const DomainOffsetTypes = Union{Number,RelativeOrigin,TypedOrigin}
const GenericDomainRateTypes = Union{Number,SampleSpace,TypedDomainSpace}
const IntegerIndexTypes = Union{Integer,AbstractRange,CartesianIndex,AbstractArray{<:Integer}}

const relativeorigin = RelativeOrigin()
const samplespace = SampleSpace()
const domainonly = DomainOnly()
const to_sample_space = ToSampleSpace()
const to_domain_space = ToDomainSpace()

interpret_origin(::RelativeOrigin) = 0
interpret_origin(::TypedOrigin{O}) where {O} = O
interpret_origin(o::Number) = o
interpret_rate(rate::R) where {R} = rate # Allow arbitrary user-defined rate types (Unitful, Measurements, etc.)
interpret_rate(rate::SampleSpace) = 1
interpret_rate(::TypedDomainSpace{D}) where {D} = D
interpret_origin(::ToSampleSpace,::RelativeOrigin,rate) = relativeorigin
interpret_origin(::ToSampleSpace,::TypedOrigin{O},rate) where {O} = TypedOrigin{O*interpret_rate(rate)}()
interpret_origin(::ToSampleSpace,origin::T,rate) where {T<:Number} = T(origin*interpret_rate(rate))
interpret_origin(::ToDomainSpace,origin::DomainOffsetTypes,rate) = interpret_origin(origin)/interpret_rate(rate)
interpret_index(::ToSampleSpace,index::Number,rate) = index * interpret_rate(rate)
interpret_index(::ToDomainSpace,index::Number,rate) = index / interpret_rate(rate)
function interpret_coordinate(::ToSampleSpace,index::T, origin::TIO, rate) where {T,TIO<:DomainOffsetTypes}
    sample_index = interpret_index(to_sample_space,index,rate)
    sample_origin = interpret_origin(to_sample_space,origin,rate)
    if isinteger(sample_index)
        return Int(sample_index),sample_origin
    else
        throw(ArgumentError("Domain-space index $index does not convert to an integer sample index"))
    end
end

# The fields are in sample space. (index, offset, even rate is in sample space for Domains to interpret!)
struct DomainIndex{TIO<:DomainOffsetTypes,TIR,DomainOnlyI}
    index::Int
    origin::TIO
    rate::TIR
    DomainIndex(index::Integer, origin::TIO=relativeorigin, rate::TIR=samplespace) where {TIO<:DomainOffsetTypes,TIR} = new{TIO,TIR,false}(index, origin, rate)
    DomainIndex(::DomainOnly, index::Integer, origin::TIO=relativeorigin, rate::TIR=samplespace) where {TIO<:DomainOffsetTypes,TIR} = new{TIO,TIR,true}(index, origin, rate)
end
DomainIndex(::ToSampleSpace, index::T, origin::TIO=relativeorigin, rate=samplespace) where {T,TIO<:DomainOffsetTypes} = DomainIndex(interpret_coordinate(to_sample_space,index,origin,rate)...,rate)
DomainIndex(::ToSampleSpace, ::DomainOnly, index::T, origin::TIO=relativeorigin, rate=samplespace) where {T,TIO<:DomainOffsetTypes} = DomainIndex(domainonly,interpret_coordinate(to_sample_space,index,origin,rate)...,rate)

struct DomainAxis{TAO<:DomainOffsetTypes,TAR,DomainOnlyA}
    origin::TAO
    rate::TAR
    DomainAxis(origin::TAO=relativeorigin,rate::TAR=samplespace) where {TAO<:DomainOffsetTypes,TAR} = new{TAO,TAR,false}(origin,rate)
    DomainAxis(::DomainOnly,origin::TAO=relativeorigin,rate::TAR=samplespace) where {TAO<:DomainOffsetTypes,TAR} = new{TAO,TAR,true}(origin,rate)
end
DomainAxis(::ToSampleSpace,origin::TAO=relativeorigin,rate=samplespace) where {TAO<:DomainOffsetTypes} = DomainAxis(interpret_origin(to_sample_space,origin,rate),rate)
DomainAxis(::ToSampleSpace,::DomainOnly,origin::TAO=relativeorigin,rate=samplespace) where {TAO<:DomainOffsetTypes} =  DomainAxis(domainonly,interpret_origin(to_sample_space,origin,rate),rate)

struct DomainArray{T,N,A<:AbstractArray{T,N},Dims<:NTuple{N,DomainAxis}} <: AbstractDomainArray{T,N}
    data::A
    dims::Dims
end
DomainArray(data::AbstractArray{T,N}) where {T,N} = DomainArray(data, ntuple(_ -> DomainAxis(), N))
Base.size(A::DomainArray) = size(A.data)
Base.axes(A::DomainArray) = axes(A.data)
rate(A::DomainArray) = map(ax -> ax.rate, A.dims)
rate(A::DomainArray, dim::Integer) = A.dims[dim].rate
origin(A::DomainArray) = map(ax -> ax.origin, A.dims)
origin(A::DomainArray, dim::Integer) = A.dims[dim].origin
domainsize(A::DomainArray) = map((s, ax) -> s / interpret_rate(ax.rate), size(A), A.dims)
domainsize(A::DomainArray, dim::Integer) = size(A, dim) / interpret_rate(A.dims[dim].rate)
interpreted_rate(A::DomainArray) = map(ax -> interpret_rate(ax.rate), A.dims)
interpreted_rate(A::DomainArray, dim::Integer) = interpret_rate(A.dims[dim].rate)
interpreted_origin(A::DomainArray) = map(ax -> interpret_origin(ax.origin), A.dims)
interpreted_origin(A::DomainArray, dim::Integer) = interpret_origin(A.dims[dim].origin)
interpreted_origin(::ToDomainSpace,A::DomainArray) = map(ax -> interpret_origin(to_domain_space,ax.origin,ax.rate), A.dims)
interpreted_origin(::ToDomainSpace,A::DomainArray, dim::Integer) = interpret_origin(to_domain_space,A.dims[dim].origin,A.dims[dim].rate)
@inline function check_index_allowed(axis::DomainAxis{TAO,TAR,DomainOnlyA},idx::DomainIndex{TIO,TIR,DomainOnlyI}) where {TAO,TAR,TIO,TIR,DomainOnlyA,DomainOnlyI}
    if axis.rate isa SampleSpace && DomainOnlyI
        error("Cannot use a domain-only index on a SampleSpace axis")
    end
    if idx.rate isa SampleSpace && DomainOnlyA
        error("Cannot use a SampleSpace index on a domain-only axis")
    end
    return true
end
@inline check_index_allowed(axis::DomainAxis{TAO,TAR,false}, idx::T) where {TAO,TAR,T<:IntegerIndexTypes} = true
@inline check_index_allowed(axis::DomainAxis{TAO,TAR,true}, idx::T) where {TAO,TAR,T<:IntegerIndexTypes} = error("Index type $T forbidden on this axis (DomainOnly=true). Use DomainIndex instead.")
@inline function convert_index(axis::DomainAxis, idx::DomainIndex)
    check_index_allowed(axis, idx)
    return convert_domain_index(axis, idx)
end
@inline convert_index(axis::DomainAxis, idx::Colon) = Colon()
@inline function convert_index(axis::DomainAxis, idx::T) where {T<:IntegerIndexTypes}
    check_index_allowed(axis,idx)
    return idx
end
@inline convert_index(all_axis::NTuple{N,DomainAxis}, I::NTuple{N,Any}) where {N} = map(convert_index,all_axis,I)
function compute_view_axes(all_axis::NTuple{N,DomainAxis}, I::NTuple{N,Any}) where {N}
    preserved_axes = DomainAxis[]
    for (axis_index, index) in enumerate(I)
        if index isa Integer
            continue # dimension removed
        else
            push!(preserved_axes, all_axis[axis_index])
        end
    end
    return Tuple(preserved_axes)
end
@inline function _require_matching_ndims(A, I)
    nd = length(A.dims)
    ni = length(I)
    nd == ni || throw(DimensionMismatch("Expected $nd indices, got $ni"))
end
function Base.view(D::DomainArray, I...)
    _require_matching_ndims(D, I)
    DomainArray(view(D.data, convert_index(D.dims, I)...), compute_view_axes(D.dims, I))
end
function Base.getindex(D::DomainArray, I...)
    _require_matching_ndims(D, I)
    DomainArray(getindex(D.data, convert_index(D.dims, I)...), compute_view_axes(D.dims, I))
end

function convert_domain_index(axis::DomainAxis{TAO,TAR,DomainOnlyA},i::DomainIndex{TIO,TIR,DomainOnlyI}) where {TAO<:DomainOffsetTypes,TAR<:GenericDomainRateTypes,TIO<:DomainOffsetTypes,TIR<:GenericDomainRateTypes,DomainOnlyA,DomainOnlyI}
    axis_rate, index_rate = interpret_rate(axis.rate), interpret_rate(i.rate)
    origin_axis, origin_index = interpret_origin(axis.origin), interpret_origin(i.origin)
    index = i.index
    if index_rate == axis_rate
        delta = origin_index - origin_axis
        if isinteger(delta)
            return index + Int(delta)
        else
            error("DomainIndex offset does not align with axis offset")
        end
    else
        axis_index = ((index + origin_index - 1) * axis_rate / index_rate) - origin_axis + 1
        if isinteger(axis_index)
            return Int(axis_index)
        else
            error("DomainIndex does not map exactly to a sample index")
        end
    end
end

canonicalize_domain_dimension_size(::ToSampleSpace,rounding_method::F,domain_length::Number,axis::DomainAxis) where {F<:Function} = rounding_method(domain_length*interpret_rate(axis.rate))
canonicalize_domain_dimension_size(::ToSampleSpace,rounding_method::F,domain_lengths::NTuple{N,Number},all_axis::NTuple{N,DomainAxis}) where {N,F<:Function} = map((domain_length,axis)->canonicalize_domain_dimension_size(to_sample_space,rounding_method,domain_length,axis),domain_lengths,all_axis)
canonicalize_domain_dim(n::Integer) = (n, DomainAxis())
canonicalize_domain_dim(t::Tuple{<:Integer,DomainAxis}) = (t[1], t[2])
canonicalize_domain_dim(t::Tuple{<:Integer,Vararg}) = (t[1], DomainAxis(t[2:end]...))
canonicalize_domain_dim(::ToSampleSpace,rounding_method::F,t::Tuple{T,DomainAxis}) where {T, F<:Function} = (canonicalize_domain_dimension_size(to_sample_space,rounding_method,t[1],t[2]),t[2])
canonicalize_domain_dim(::ToSampleSpace,rounding_method::F,n::T) where {T,F<:Function} = canonicalize_domain_dim(to_sample_space,rounding_method,(n,DomainAxis()))
canonicalize_domain_dim(::ToSampleSpace,rounding_method::F,t::Tuple{T,Vararg}) where {T,F<:Function} = canonicalize_domain_dim(to_sample_space,rounding_method,(t[1],DomainAxis(to_sample_space,t[2:end]...)))

function domainzeros end
function domainones end

for (fname, basefill) in ((:domainzeros, :zeros),(:domainones, :ones))
    @eval begin
        $fname(::Type{T}, dims...) where {T} = $fname(T, dims)
        $fname(::ToSampleSpace, rounding_method::F, ::Type{T}, dims...) where {T,F<:Function} = $fname(to_sample_space, rounding_method, T, dims)
        $fname(dims...) = $fname(Float64, dims)
        $fname(::ToSampleSpace, rounding_method::F, dims...) where {F<:Function} = $fname(to_sample_space, rounding_method, Float64, dims)
        $fname(::ToSampleSpace, args...) = $fname(to_sample_space, x->ceil(Int,x), args...)
        function $fname(::Type{T}, dims::Tuple) where {T}
            canonical_dims = map(canonicalize_domain_dim, dims)
            return DomainArray($basefill(T, map(first, canonical_dims)),map(last, canonical_dims))
        end
        function $fname(::ToSampleSpace, rounding_method::F, ::Type{T}, dims::Tuple) where {T,F<:Function}
            canonical_dims = map(d -> canonicalize_domain_dim(to_sample_space, rounding_method, d), dims)
            return DomainArray($basefill(T, map(first, canonical_dims)),map(last, canonical_dims))
        end
    end
end

Base.similar(A::DomainArray) = DomainArray(similar(A.data), A.dims)
Base.similar(A::DomainArray, ::Type{T}) where {T} = DomainArray(similar(A.data, T), A.dims)
Base.similar(A::DomainArray{T,N,Data,Dims}, dims::NTuple{N,Integer}) where {T,N,Data,Dims} = DomainArray(similar(A.data, dims), A.dims)
Base.similar(A::DomainArray{T,N,Data,Dims},::ToSampleSpace,rounding_method::F,dims::NTuple{N,Integer}) where {T,N,Data,Dims,F<:Function} = DomainArray(similar(A.data, canonicalize_domain_dimension_size(to_sample_space,rounding_method,dims,A.dims)), A.dims)
Base.similar(A::DomainArray{T,N,Data,Dims},::Type{T1},::ToSampleSpace,rounding_method::F,dims::NTuple{N,Integer}) where {T,T1,N,Data,Dims,F<:Function} = DomainArray(similar(A.data, T1, canonicalize_domain_dimension_size(to_sample_space,rounding_method,dims,A.dims)), A.dims)
Base.similar(A::DomainArray{T,N,Data,Dims}, ::ToSampleSpace, dims::NTuple{N,Integer}) where {T,N,Data,Dims} = similar(A,to_sample_space,x->ceil(Int,x),dims)
Base.similar(A::DomainArray{T,N,Data,Dims}, output_type::Type{T1}, ::ToSampleSpace, dims::NTuple{N,Integer}) where {T,T1,N,Data,Dims} = similar(A,output_type,to_sample_space,x->ceil(Int,x),dims)

struct DomainSpan{O,L}
    origin::O
    length::L
end
#=
to_sample_range(span::DomainSpan, axis::DomainAxis) = begin
    rate = interpret_rate(axis.rate)
    start = span.origin
    stop  = span.origin + span.length
    i1 = ceil(Int, start * rate)
    i2 = floor(Int, stop  * rate)
    i1:i2
end
=#


pretty_origin(::RelativeOrigin, rate) = "_"
pretty_origin(::TypedOrigin{O}, rate) where {O} = string(interpret_origin(to_domain_space, O, rate), "*")
pretty_origin(o::Number, rate) = string(interpret_origin(to_domain_space, o, rate))

pretty_rate(::SampleSpace) = "_"
pretty_rate(::TypedDomainSpace{D}) where {D} = string(D, "*")
pretty_rate(r::Number) = string(r)

Base.show(io::IO, ax::DomainAxis) = print(io, pretty_origin(ax.origin, ax.rate), ":", pretty_rate(ax.rate))
function Base.show(io::IO, ::MIME"text/plain", A::DomainArray)
    # Domain metadata
    ds = domainsize(A)
    rs = map(ax -> pretty_rate(ax.rate), A.dims)
    os = map(ax -> pretty_origin(ax.origin, ax.rate), A.dims)

    # Header: size, origin, then array type and rate
    print(io, "(")
    print(io, join(ds, ", "))
    print(io, ") DomainArray with origin (")
    print(io, join(os, ", "))
    print(io, ") @ (")
    print(io, join(rs, ", "))
    println(io, ")")

    # Underlying array
    inner = IOContext(io, :compact => true)
    Base.show(inner, MIME"text/plain"(), A.data)
end

############################
# domainaxes
############################
#=
function domainaxes(D::DomainArray{T,N,A,Dims}, dim::Int) where {T,N,A,Dims,}
    len = size(D.data, dim)
    diminfo = D.dims[dim]
    rate   = diminfo.rate
    offset = diminfo.offset
    return (DomainIndex{rate, typeof(offset)}(i, offset) for i in 1:len)
end

function _broadcast_safe(D::DomainArray)
    for ax in D.dims
        if ax.DomainOnlyA
            return false
        end
    end
    return true
end

import Base.Broadcast: broadcasted, materialize

function broadcasted(f, D::DomainArray, A::AbstractArray)
    if !_broadcast_safe(D)
        error("Cannot broadcast DomainArray with DomainOnly axes against a normal array.")
    end
    return DomainArray(broadcast(f, D.data, A), D.dims)
end
=#
#end # module
