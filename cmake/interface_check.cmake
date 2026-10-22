# Compiles every interface and utils header on its own, without the
# xbyak_riscv include path, so an ISA-specific name or include fails the build
file(GLOB_RECURSE _rvjit_isa_free_headers
    RELATIVE ${PROJECT_SOURCE_DIR}
    ${PROJECT_SOURCE_DIR}/rvjit/interface/*.hpp
    ${PROJECT_SOURCE_DIR}/rvjit/utils/*.hpp)

set(_rvjit_interface_check_sources)
foreach(_header ${_rvjit_isa_free_headers})
    string(MAKE_C_IDENTIFIER ${_header} _stem)
    set(_source ${CMAKE_CURRENT_BINARY_DIR}/interface_check/${_stem}.cpp)
    file(GENERATE OUTPUT ${_source} CONTENT "#include \"${_header}\"\n")
    list(APPEND _rvjit_interface_check_sources ${_source})
endforeach()

add_library(rvjit_interface_check OBJECT ${_rvjit_interface_check_sources})
target_include_directories(rvjit_interface_check PRIVATE ${PROJECT_SOURCE_DIR})
if(RVJIT_DEBUG)
    target_compile_definitions(rvjit_interface_check PRIVATE RVJIT_DEBUG)
endif()
set_target_properties(rvjit_interface_check PROPERTIES
    CXX_STANDARD 11 CXX_STANDARD_REQUIRED ON CXX_EXTENSIONS OFF)
