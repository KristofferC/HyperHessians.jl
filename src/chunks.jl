struct Chunk{N} end

function Chunk(input_length::Integer, ::Type{T} = Float64) where {T}
    return Chunk{pickchunksize(input_length, T)}()
end

Chunk(x::AbstractArray) = Chunk(length(x), eltype(x))

# All 8 for now...
maxchunksize(::Type{Float64}) = 8
maxchunksize(::Type{Float32}) = 8
maxchunksize(::Type{T}) where {T} = 8

function pickchunksize(input_length, ::Type{T}) where {T}
    max_size = maxchunksize(T)
    input_length <= max_size && return input_length
    return max_size
end

chunksize(::Chunk{N}) where {N} = N::Int
