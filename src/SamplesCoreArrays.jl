module DomainArrays

export RelativeOrigin, TypedOrigin, DomainOffsetTypes,
       SampleSpace, TypedDomainSpace,
       DomainDimension, DomainIndex, DomainArray,
       domainaxes, domainmap

struct RelativeOrigin end
struct TypedOrigin{O} end
const DomainOffsetTypes = Union{Number,RelativeOrigin,TypedOrigin}
struct SampleSpace end
struct TypedDomainSpace{D} end
const GenericDomainRateTypes = Union{Number,SampleSpace,TypedDomainSpace}
const IntegerIndexTypes = Union{Integer,AbstractRange,CartesianIndex,AbstractArray{<:Integer}}
# The fields are in sample space. (index, offset, even rate is in sample space for Domains to interpret!)
struct DomainIndex{TIO<:DomainOffsetTypes,TIR,DomainOnlyI}
    origin_shift::TIO
    index::Int
    rate::TIR
    DomainIndex(origin_shift::TIO,index::Integer,rate::TIR) where {TIO<:DomainOffsetTypes,TIR} = new{TIO,TIR,false}(origin_shift,index,rate)
    DomainIndex(origin_shift::TIO,index::Integer,rate::TIR;DomainOnly::Bool) where {TIO<:DomainOffsetTypes,TIR} = new{TIO,TIR,DomainOnly}(origin_shift,index,rate)
end
struct DomainAxis{TAO<:DomainOffsetTypes,TAR,DomainOnlyA}
    origin_shift::TAO
    rate::TAR
    DomainAxis(origin_shift::TAO,rate::TAR) where {TAO<:DomainOffsetTypes,TAR} = new{TAO,TAR,false}(origin_shift,rate)
    DomainAxis(origin_shift::TAO,rate::TAR;DomainOnly::Bool) where {TAO<:DomainOffsetTypes,TAR} = new{TAO,TAR,DomainOnly}(origin_shift,rate)
end
abstract type AbstractDomainArray{T,N} <: AbstractArray{T,N} end
struct DomainArray{T,N,A<:AbstractArray{T,N},Dims<:NTuple{N,DomainAxis}} <: AbstractDomainArray{T,N}
    data::A
    dims::Dims
end
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
@inline check_index_allowed(axis::DomainAxis{TAO,TAR,true}, idx::T) where {TAO,TAR,T<:IntegerIndexTypes} =error("Index type $T forbidden on this axis (DomainOnly=true). Use DomainIndex instead.")
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

extract_origin(::RelativeOrigin) = 0
extract_origin(o::TypedOrigin{O}) where {O} = O
extract_origin(o::Number) = o
rate_converter(rate::R) = rate # Allow arbitrary user-defined rate types (Unitful, Measurements, etc.)
rate_converter(rate::SampleSpace) = 1
rate_converter(rate::TypedDomainSpace{D}) = D
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

############################
# Strict domain mapping
############################

offset_to_number(o::RelativeOrigin) = 0
offset_to_number(o::TypedOrigin{O}) where {O} = O
offset_to_number(o) = o  # numeric

rate_to_number(r::SampleSpace) = 1
rate_to_number(r::TypedDomainSpace{D}) where {D} = D
rate_to_number(r) = r  # numeric / unitful

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

"""
    domainmap(D, dim, idx)

Strictly map a DomainIndex `idx` on dimension `dim` to a sample index in `D`.
"""
function domainmap(D::DomainArray, dim::Int,
                   idx::DomainIndex{Ri,TIO,DomainOnly}) where {Ri,TIO,DomainOnly}

    diminfo = D.dims[dim]

    # Source (index)
    Ri_num = rate_to_number(Ri)
    io     = offset_to_number(idx.offset)

    # Destination (array)
    Ra_num = rate_to_number(diminfo.rate)
    ao     = offset_to_number(diminfo.offset)

    i = idx.index

    ###############################################
    # Case 1: Same rate
    ###############################################
    if Ri_num == Ra_num
        delta = io - ao
        if isinteger(delta)
            return i + Int(delta)
        else
            error("DomainIndex does not align with destination sampling grid")
        end
    end

    ###############################################
    # Case 2: Integer/integer rational alignment
    ###############################################
    if Ri_num isa Integer && Ra_num isa Integer
        num = i * Ra_num + (io - ao)
        if num % Ri_num == 0
            return div(num, Ri_num)
        else
            error("DomainIndex does not align with destination sampling grid")
        end
    end

    ###############################################
    # Case 3: General real-valued rates
    ###############################################
    t = (i - 1 + io) / Ri_num
    j_real = t * Ra_num - ao + 1

    if isinteger(j_real)
        return Int(j_real)
    else
        error("DomainIndex does not map exactly to a sample index")
    end
end

end # module
