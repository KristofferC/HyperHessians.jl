module ChunkTests

using Test
using HyperHessians: Chunk, chunksize, pickchunksize

@testset "pickchunksize" begin
    for T in (Float64, Float32, BigFloat), n in 0:10
        @test pickchunksize(n, T) == n
    end
    @test pickchunksize(11, Float64) == 4
    @test pickchunksize(32, Float64) == 4
    @test pickchunksize(33, Float64) == 6
    @test pickchunksize(100, Float64) == 6
    @test pickchunksize(64, BigFloat) == 6
    # Float32 duals are half as wide; thresholds move up
    @test pickchunksize(11, Float32) == 11
    @test pickchunksize(12, Float32) == 12
    @test pickchunksize(13, Float32) == 6
    @test pickchunksize(100, Float32) == 6

    @test chunksize(Chunk(rand(6))) == 6
    @test chunksize(Chunk(rand(20))) == 4
    @test chunksize(Chunk(rand(Float32, 20))) == 6
    @test chunksize(Chunk(rand(64))) == 6
end

end # module
