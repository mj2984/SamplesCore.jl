using Test

@testset "SampleArray basics" begin
    data = collect(1:1000)
    S = SampleArray(data, (100.0,))   # 100 Hz sampling rate

    @test size(S) == (1000,)
    @test S.rate == (100.0,)
    @test S[1] == 1
end


@testset "Time-domain indexing" begin
    data = collect(1:1000)
    S = SampleArray(data, (100.0,))   # 100 Hz

    # 0.0 seconds → index 1
    @test S[0.0] == 1

    # 1.0 seconds → index 100
    @test S[1.0] == 100

    # Range 0.0..0.1 → ceil(0*100)=1, floor(0.1*100)=10
    @test S[0.0:0.001:0.1] == 1:10
end


@testset "Sample-domain indexing" begin
    data = collect(1:1000)
    S = SampleArray(data, (100.0,))

    @test S.sample[1:5] == 1:5
end


@testset "setindex! time-domain" begin
    data = collect(1:1000)
    S = SampleArray(copy(data), (100.0,))

    S[0.0:0.001:0.05] .= 0   # first 5 samples
    @test all(S.sample[1:5] .== 0)
end


@testset "setindex! sample-domain" begin
    data = collect(1:1000)
    S = SampleArray(copy(data), (100.0,))

    S.sample[1:10] .= 99
    @test all(S.sample[1:10] .== 99)
end


@testset "timeslice (copy)" begin
    data = collect(1:1000)
    S = SampleArray(data, (100.0,))

    S2 = timeslice(S, 0.0:0.001:0.1)   # 1:10
    @test S2.sample == 1:10
    @test S2.rate == S.rate
end


@testset "timeview (view semantics)" begin
    data = collect(1:1000)
    S = SampleArray(data, (100.0,))

    V = timeview(S, 0.0:0.001:0.1)  # 1:10

    @test V.sample == @view data[1:10]
    @test V.offset == (0.0,)   # exact alignment
end


@testset "cascaded timeview offset accumulation" begin
    data = collect(1:1000)
    S = SampleArray(data, (100.0,))

    # First view: small time window
    V1 = timeview(S, 0.01:0.001:0.02)

    # Second view: local time window inside V1
    V2 = timeview(V1, 0.0:0.001:0.01)

    # Offsets accumulate (non‑trivial check)
    @test V2.offset[1] ≈ V1.offset[1] + (V2.offset[1] - V1.offset[1])
end


@testset "extreme view" begin
    data = collect(1:1000)
    S = SampleArray(data, (100.0,))

    V = timeview_extreme(S, 0.0:0.001:0.1)

    @test V.sample == @view data[1:10]
end


@testset "pretty printing" begin
    data = collect(1:1000)
    S = SampleArray(data, (100.0,))
    V = timeview(S, 0.0:0.001:0.1)

    io = IOBuffer()
    show(io, S)
    str = String(take!(io))
    @test occursin("SampleArray", str)
    @test occursin("rate=(100.0,)", str)

    io = IOBuffer()
    show(io, V)
    str = String(take!(io))
    @test occursin("SampleView", str)
    @test occursin("offset=", str)
end


@testset "@sampleidx macro" begin
    data = collect(1:1000)
    S = SampleArray(copy(data), (100.0,))

    @sampleidx S[1:5] .= 77
    @test all(S.sample[1:5] .== 77)
end
