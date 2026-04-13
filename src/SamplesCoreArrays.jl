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
# The fields are in sample space. (index, offset, even rate is in sample space for Domains to interpret!)
struct DomainIndex{TIO<:DomainOffsetTypes,TIR,DomainOnlyI}
    offset::TIO
    index::Int
    rate::TIR
    DomainIndex(offset::TIO,index::Integer,rate::TIR) where {TIO<:DomainOffsetTypes,TIR} = new{TIO,TIR,false}(offset,index,rate)
    DomainIndex(offset::TIO,index::Integer,rate::TIR;DomainOnly::Bool) where {TIO<:DomainOffsetTypes,TIR} = new{TIO,TIR,DomainOnly}(offset,index,rate)
end
struct DomainAxis{TAO<:DomainOffsetTypes,TAR,DomainOnlyA}
    offset::TAO
    rate::TAR
    DomainAxis(offset::TAO,rate::TAR) where {TAO<:DomainOffsetTypes,TAR} = new{TAO,TAR,false}(offset,rate)
    DomainAxis(offset::TAO,rate::TAR;DomainOnly::Bool) where {TAO<:DomainOffsetTypes,TAR} = new{TAO,TAR,DomainOnly}(offset,rate)
end
struct DomainArray{T,N,A<:AbstractArray{T,N},Dims<:NTuple{N,DomainAxis}} <: AbstractArray{T,N}
    data::A
    dims::Dims
end
Base.size(D::DomainArray) = size(D.data)
Base.axes(D::DomainArray) = axes(D.data)
# Normal indexing with integers: allowed only on axes where DomainOnlyA == false
function Base.getindex(D::DomainArray, I...)
    # Check dimensionality
    length(I) == length(D.dims) ||
        throw(DimensionMismatch("Expected $(length(D.dims)) indices, got $(length(I))"))

    OutputIndices = ()

    for (k, idx) in enumerate(I)
        axis = D.dims[k]

        if idx isa Integer
            # Integer indexing allowed only if DomainOnlyA == false
            if axis.DomainOnlyA
                error("Normal indexing forbidden on axis $k (DomainOnly=true). Use DomainIndex instead.")
            end
            OutputIndices = (OutputIndices..., idx)

        elseif idx isa DomainIndex
            # Convert DomainIndex → sample index
            OutputIndices = (OutputIndices..., NoInterpolationDomainIndexer(axis, idx))

        elseif idx === Colon()
            # Colon indexing always allowed
            OutputIndices = (OutputIndices..., Colon())

        elseif idx isa AbstractRange
            # Range indexing allowed only if axis is not DomainOnly
            if axis.DomainOnlyA
                error("Range indexing forbidden on axis $k (DomainOnly=true). Use DomainIndex instead.")
            end
            OutputIndices = (OutputIndices..., idx)

        else
            error("Unsupported index type $(typeof(idx)) for axis $k")
        end
    end

    return getindex(D.data, OutputIndices...)
end

# DomainIndex indexing
function Base.getindex(D::DomainArray{T,N,A,Dims}, i::DomainIndex{TIO,TIR,DomainOnlyI}, I...) where {T,N,A,Dims,Rate,TIO,DomainOnly}
    iD = NoInterpolationDomainIndexer(D, 1, i)   # strict mapping along dimension 1
    return getindex(D.data, iD, I...)
end

extract_offset(offset::TypedOrigin{O}) = O
extract_offset(offset::Number) = offset
extract_offset(offset::RelativeOrigin) = 0
extract_rate(rate::TypedDomainSpace{D}) = D
extract_rate(rate::Number) = rate
#extract_rate(rate::SampleSpace) = 1
function isRateMatched(indexRate::TIR,axisRate::TAR,DomainOnlyI::Bool,DomainOnlyR::Bool)
function NoInterpolationDomainIndexer(i::DomainIndex{TIO,TIR,DomainOnlyI},a::DomainDimension{TAO,TAR,DomainOnlyR})

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
