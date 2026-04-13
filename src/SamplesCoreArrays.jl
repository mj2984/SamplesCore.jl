module DomainArrays

export RelativeOrigin, TypedOrigin, DomainOffsetTypes,
       SampleSpace, TypedDomainSpace,
       DomainDimension, DomainIndex, DomainArray,
       domainaxes, domainmap

struct RelativeOrigin end
struct TypedOrigin{O} end
struct SampleSpace end
struct TypedDomainSpace{D} end
abstract type AbstractDomainArray{T,N} <: AbstractArray{T,N} end

const DomainOffsetTypes = Union{Number,RelativeOrigin,TypedOrigin}
const GenericDomainRateTypes = Union{Number,SampleSpace,TypedDomainSpace}
const IntegerIndexTypes = Union{Integer,AbstractRange,CartesianIndex,AbstractArray{<:Integer}}
# The fields are in sample space. (index, offset, even rate is in sample space for Domains to interpret!)

extract_origin(::RelativeOrigin) = 0
extract_origin(::TypedOrigin{O}) where {O} = O
extract_origin(o::Number) = o
rate_converter(rate::R) where {R} = rate # Allow arbitrary user-defined rate types (Unitful, Measurements, etc.)
rate_converter(rate::SampleSpace) = 1
rate_converter(::TypedDomainSpace{D}) where {D} = D
struct DomainIndex{TIO<:DomainOffsetTypes,TIR,DomainOnlyI}
    index::Int
    origin_shift::TIO
    rate::TIR
end
DomainIndex(index::Integer, origin_shift::TIO, rate::TIR) where {TIO<:DomainOffsetTypes,TIR} = DomainIndex{TIO,TIR,false}(index, origin_shift, rate)
DomainIndex(index::Integer, origin_shift::TIO) where {TIO<:DomainOffsetTypes} = DomainIndex(index, origin_shift, SampleSpace())
DomainIndex(index::Integer) = DomainIndex(index, RelativeOrigin())
DomainOnlyIndex(index::Integer, origin_shift::TIO, rate::TIR) where {TIO<:DomainOffsetTypes,TIR} = DomainIndex{TIO,TIR,true}(index, origin_shift, rate)
DomainOnlyIndex(index::Integer, origin_shift::TIO) where {TIO<:DomainOffsetTypes} = DomainOnlyIndex(index, origin_shift, SampleSpace())
DomainOnlyIndex(index::Integer) = DomainOnlyIndex(index, RelativeOrigin())

struct DomainAxis{TAO<:DomainOffsetTypes,TAR,DomainOnlyA}
    origin_shift::TAO
    rate::TAR
end
DomainAxis(origin_shift::TAO,rate::TAR) where {TAO<:DomainOffsetTypes,TAR} = DomainAxis{TAO,TAR,false}(origin_shift,rate)
DomainAxis(origin_shift::TAO) where {TAO<:DomainOffsetTypes} = DomainAxis(origin_shift,SampleSpace())
DomainAxis() = DomainAxis(RelativeOrigin())
DomainOnlyAxis(origin_shift::TAO,rate::TAR) where {TAO<:DomainOffsetTypes,TAR} = DomainAxis{TAO,TAR,true}(origin_shift,rate)
DomainOnlyAxis(origin_shift::TAO) where {TAO<:DomainOffsetTypes} = DomainOnlyAxis(origin_shift,SampleSpace())
DomainOnlyAxis() = DomainOnlyAxis(RelativeOrigin())
DomainAxisFromDomain(::RelativeOrigin,rate::TAR) where {TAR} = DomainAxis(RelativeOrigin(),rate)
DomainAxisFromDomain(::TypedOrigin{O},rate::TAR) where {O,TAR} = DomainAxis(TypedOrigin{O*rate_converter(rate)}(),rate)
DomainAxisFromDomain(origin_shift::T,rate::TAR) where {T<:Number} = DomainAxis(T(origin_shift*rate_converter(rate)),rate)
DomainAxisFromDomain(origin_shift::TAO) where {TAO<:DomainOffsetTypes} = DomainAxisFromDomain(origin_shift,SampleSpace())
DomainAxisFromDomain() = DomainAxis(RelativeOrigin())
DomainOnlyAxisFromDomain(::RelativeOrigin,rate::TAR) where {TAR} = DomainOnlyAxis(RelativeOrigin(),rate)
DomainOnlyAxisFromDomain(::TypedOrigin{O},rate::TAR) where {O,TAR} = DomainOnlyAxis(TypedOrigin{O*rate_converter(rate)}(),rate)
DomainOnlyAxisFromDomain(origin_shift::T,rate::TAR) where {T<:Number} = DomainOnlyAxis(T(origin_shift*rate_converter(rate)),rate)
DomainOnlyAxisFromDomain(origin_shift::TAO) where {TAO<:DomainOffsetTypes} = DomainOnlyAxisFromDomain(origin_shift,SampleSpace())
DomainOnlyAxisFromDomain() = DomainOnlyAxis(RelativeOrigin())

struct DomainArray{T,N,A<:AbstractArray{T,N},Dims<:NTuple{N,DomainAxis}} <: AbstractDomainArray{T,N}
    data::A
    dims::Dims
end
DomainArray(data::AbstractArray{T,N}) where {T,N} = DomainArray(data, ntuple(_ -> DomainAxis(), N))
Base.size(D::DomainArray) = size(D.data)
Base.axes(D::DomainArray) = axes(D.data)
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
function Base.getindex(D::DomainArray, I...)
    length(I) == length(D.dims) ||
        throw(DimensionMismatch("Expected $(length(D.dims)) indices, got $(length(I))"))

    converted = ntuple(k -> convert_index(D.dims[k], I[k]), length(I))
    return getindex(D.data, converted...)
end

function convert_domain_index(axis::DomainAxis{TAO,TAR,DomainOnlyA},i::DomainIndex{TIO,TIR,DomainOnlyI}) where {TAO<:DomainOffsetTypes,TAR<:GenericDomainRateTypes,TIO<:DomainOffsetTypes,TIR<:GenericDomainRateTypes,DomainOnlyA,DomainOnlyI}
    axis_rate, index_rate = rate_converter(axis.rate), rate_converter(i.rate)
    origin_axis, origin_index = extract_origin(axis.origin_shift), extract_origin(i.origin_shift)
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

normalize_sample_dim(n::Integer) = (n, DomainAxis())
normalize_sample_dim(t::Tuple{<:Integer,DomainAxis}) = (t[1], t[2])
normalize_sample_dim(t::Tuple{<:Integer,Vararg}) = (t[1], DomainAxis(t[2:end]...))
normalize_sample_dim_from_domain(n::T) where {T} = (ceil(Int,n), DomainAxisFromDomain())
normalize_sample_dim_from_domain(t::Tuple{T,DomainAxis}) where {T} = (ceil(Int,t[1]*rate_converter(t[2].rate)),t[2])
normalize_sample_dim_from_domain(t::Tuple{T,Vararg}) where {T} = normalize_sample_dim_from_domain(t[1],DomainAxisFromDomain(t[2:end]...))
normalize_sample_dims(dims...) = map(normalize_sample_dim, dims)
normalize_sample_dims_from_domain(dims...) = map(normalize_sample_dim_from_domain, dims)

function DomainZeros(::Type{T}, dims_argument) where {T}
    dims = normalize_sample_dims(dims_argument...)
    sizes = map(first, dims)
    axes  = map(last, dims)
    data  = zeros(T, Tuple(sizes))
    return DomainArray(data, Tuple(axes))
end

function DomainZerosFromDomain(::Type{T}, dims_argument) where {T}
    dims = normalize_sample_dims_from_domain(dims_argument...)
    sizes = map(first, dims)
    axes  = map(last, dims)
    data  = zeros(T, Tuple(sizes))
    return DomainArray(data, Tuple(axes))
end


############################
# domainaxes
############################

function domainaxes(D::DomainArray{T,N,A,Dims,DomainOnly}, dim::Int) where {T,N,A,Dims,DomainOnly}
    len = size(D.data, dim)
    diminfo = D.dims[dim]
    rate   = diminfo.rate
    offset = diminfo.offset
    return (DomainIndex{rate, typeof(offset), DomainOnly}(i, offset) for i in 1:len)
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

end # module
