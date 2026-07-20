using HyperHessians
using ParallelTestRunner

testsuite = find_tests(@__DIR__)
# helpers.jl provides shared utilities but is not a test file itself
haskey(testsuite, "helpers") && delete!(testsuite, "helpers")

push!(ARGS, "--jobs=$(Sys.CPU_THREADS)")
# --threads overrides the JULIA_NUM_THREADS=1 that ParallelTestRunner sets for
# its workers; threaded_tests.jl needs real parallelism.
runtests(HyperHessians, ARGS; testsuite, exeflags = ["--threads=4"])
