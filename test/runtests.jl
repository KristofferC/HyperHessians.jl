using HyperHessians
using ParallelTestRunner

testsuite = find_tests(@__DIR__)
# helpers.jl provides shared utilities but is not a test file itself
haskey(testsuite, "helpers") && delete!(testsuite, "helpers")

push!(ARGS, "--jobs=$(Sys.CPU_THREADS)")
runtests(HyperHessians, ARGS; testsuite)
