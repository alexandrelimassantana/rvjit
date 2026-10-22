add_executable(bench_matmul
    bench_matmul.cpp
)

target_include_directories(bench_matmul PRIVATE
    ${CMAKE_SOURCE_DIR}/compat
    ${CMAKE_SOURCE_DIR}/test
    ${CMAKE_CURRENT_SOURCE_DIR}
)

target_link_libraries(bench_matmul PRIVATE rvjit::rvjit)

# Correctness probing, CTest-registered like test_rvjit -- CMAKE_CROSSCOMPILING_
# EMULATOR (set for the whole build, see the root CMakeLists.txt/CI workflow)
# is automatically prepended by CTest, so this runs under QEMU in CI with no
# extra wrapping. Guarded on RVJIT_BUILD_TESTS (not RVJIT_BUILD_BENCH, which
# only controls whether the binary itself gets built): "is this bench probe
# part of the test suite" is a separate question from "is bench built at
# all" -- a RVJIT_BUILD_BENCH=ON, RVJIT_BUILD_TESTS=OFF configuration builds
# bench_matmul for manual/local use without registering it as a test.
#
# --mode=P (performance) or the broader test_matmul_all sweep are
# deliberately never add_test()-registered: QEMU timings aren't
# representative of real hardware, and the full sweep is slower than what
# should gate a PR -- both are for test/bench/scripts/run_matmul*.sh instead.
if(RVJIT_BUILD_TESTS)
    set(_matmul_ci ${CMAKE_CURRENT_SOURCE_DIR}/inputs/test_matmul_ci)

    # One entry per VLEN QEMU supports (128 to 1024 bits), so the backend's
    # LMUL/unroll/tile decisions are exercised at every vector length. The
    # emulator is spelled out in COMMAND (a non-target first word is never
    # prefixed with CMAKE_CROSSCOMPILING_EMULATOR again); a native RV64 build
    # cannot pick its VLEN and registers the single default entry instead
    if(CMAKE_CROSSCOMPILING AND CMAKE_CROSSCOMPILING_EMULATOR)
        foreach(vlen 128 256 512 1024)
            add_test(NAME bench_matmul_ci_vlen${vlen}
                COMMAND ${CMAKE_CROSSCOMPILING_EMULATOR}
                        -cpu max,vlen=${vlen},vext_spec=v1.0
                        $<TARGET_FILE:bench_matmul> --mode=C
                        --batch=${_matmul_ci})
        endforeach()
    else()
        add_test(NAME bench_matmul_ci
            COMMAND bench_matmul --mode=C --batch=${_matmul_ci})
    endif()
endif()
