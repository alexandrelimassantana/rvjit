include(FetchContent)
FetchContent_Declare(
    googletest
    GIT_REPOSITORY https://github.com/google/googletest.git
    GIT_TAG release-1.12.1 # the last release supporting C++11
)
# Match gtest's CRT to this project's rather than forcing its own default.
set(gtest_force_shared_crt ON CACHE BOOL "" FORCE)
FetchContent_MakeAvailable(googletest)

file(GLOB_RECURSE RVJIT_TEST_SOURCES ${CMAKE_CURRENT_SOURCE_DIR}/gtest/test_*.cpp)

add_executable(test_rvjit ${RVJIT_TEST_SOURCES})

target_include_directories(test_rvjit PRIVATE
    ${CMAKE_CURRENT_SOURCE_DIR}/gtest
    ${CMAKE_CURRENT_SOURCE_DIR}
    ${CMAKE_SOURCE_DIR}/compat
)

target_link_libraries(test_rvjit PRIVATE
    rvjit::rvjit
    GTest::gtest_main
)

# Not gtest_discover_tests(): it runs the freshly-built binary at configure
# time to enumerate individual tests, which breaks cross-compiling to RV64
# from a non-RV64 host. Register the whole binary as one CTest entry instead
# -- run it under QEMU or on real RV64 hardware, e.g. `qemu-riscv64
# build/test/test_rvjit`.
add_test(NAME rvjit_tests COMMAND test_rvjit)
