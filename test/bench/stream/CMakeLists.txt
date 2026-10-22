add_executable(bench_iterator
    bench_iterator.cpp
)
add_executable(bench_memory_operand
    bench_memory_operand.cpp
)

foreach(tgt bench_iterator bench_memory_operand)
    target_include_directories(${tgt} PRIVATE
        ${CMAKE_SOURCE_DIR}/compat
        ${CMAKE_SOURCE_DIR}/test
        ${CMAKE_CURRENT_SOURCE_DIR}
    )
    target_link_libraries(${tgt} PRIVATE rvjit::rvjit)
endforeach()

# Correctness probes, CTest-registered like bench_matmul_ci -- see
# test/bench/matmul/CMakeLists.txt for why this is gated on RVJIT_BUILD_TESTS
# rather than RVJIT_BUILD_BENCH, and why there's no --mode=P/perf
# equivalent registered here (none exists for these two: instruction-count
# reporting is inherent to every run, not a separate timed mode).
if(RVJIT_BUILD_TESTS)
    add_test(NAME bench_iterator_ci
        COMMAND bench_iterator --shape=8x4 --strategy=pivot)
    add_test(NAME bench_iterator_direct_ci
        COMMAND bench_iterator --shape=8x4 --strategy=direct)
    add_test(NAME bench_memory_operand_ci
        COMMAND bench_memory_operand --shape=8x4 --strategy=pivot)
    add_test(NAME bench_memory_operand_direct_ci
        COMMAND bench_memory_operand --shape=8x4 --strategy=direct)
    # Register strides and inner end axes under outer axes that move
    foreach(tgt bench_iterator bench_memory_operand)
        add_test(NAME ${tgt}_strips_ci
            COMMAND ${tgt} --shape=8x4 --stride=1024,reg\(4\))
        add_test(NAME ${tgt}_base_only_ci
            COMMAND ${tgt} --shape=16x6 --addressing=base_only
                --stride=2048,4)
        add_test(NAME ${tgt}_inner_end_ci
            COMMAND ${tgt} --shape=8x4 --stride=reg\(3000\),4 --end=advance:1)
    endforeach()
endif()
