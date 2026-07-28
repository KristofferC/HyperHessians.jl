struct Chunk{N} end

function Chunk(input_length::Integer, ::Type{T} = Float64) where {T}
    return Chunk{pickchunksize(input_length, T)}()
end

Chunk(x::AbstractArray) = Chunk(length(x), eltype(x))

# Function-agnostic default measured over a benchmark grid (6 function
# families x 20 input sizes x chunks, on AVX2, AVX-512 and NEON; see
# ChunkPicker.jl's benchmark/RESULTS.md): full-vector single evaluation
# while small, then a small chunk — large chunks only pay off for expensive
# functions, which a default cannot know (benchmark with ChunkPicker.jl to
# specialize). Float32 duals are half as wide, moving the thresholds up.
function pickchunksize(input_length, ::Type{T}) where {T}
    input_length <= 10 && return Int(input_length)
    return input_length <= 32 ? 4 : 6
end
function pickchunksize(input_length, ::Type{Float32})
    input_length <= 12 && return Int(input_length)
    return 6
end

chunksize(::Chunk{N}) where {N} = N::Int
