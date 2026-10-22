# test/bench/common is header-only: no add_subdirectory of its own, each driver
# below reaches it via its own target_include_directories(... ${CMAKE_
# SOURCE_DIR}/test) and #include "bench/common/<header>.hpp".
add_subdirectory(matmul)
add_subdirectory(stream)
