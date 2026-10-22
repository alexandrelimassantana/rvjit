cmake_minimum_required(VERSION 3.14)

project(rvjit LANGUAGES CXX VERSION 0.1.0)

option(RVJIT_BUILD_TESTS "Build the rvjit gtest suite" ON)
option(RVJIT_BUILD_BENCH "Build the rvjit component benchmarks/examples" ON)
option(RVJIT_DEBUG "Enable rvjit DEBUG developer traces on failure" ON)
option(RVJIT_BUILD_DOCS "Build the rvjit Doxygen API documentation" OFF)

set(CMAKE_CXX_STANDARD 11)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)

set(XBYAK_RISCV_V ON CACHE BOOL "" FORCE)
# CMake 4 rejects xbyak_riscv's cmake_minimum_required(2.6...3.0.2).
if(NOT DEFINED CMAKE_POLICY_VERSION_MINIMUM)
    set(CMAKE_POLICY_VERSION_MINIMUM 3.5)
    add_subdirectory(third_party/xbyak_riscv)
    unset(CMAKE_POLICY_VERSION_MINIMUM)
else()
    add_subdirectory(third_party/xbyak_riscv)
endif()

add_library(rvjit INTERFACE)
add_library(rvjit::rvjit ALIAS rvjit)

target_include_directories(rvjit INTERFACE
    "$<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}>"
    "$<INSTALL_INTERFACE:${CMAKE_INSTALL_INCLUDEDIR}>"
)

target_compile_definitions(rvjit INTERFACE XBYAK_RISCV_V=1)
if(RVJIT_DEBUG)
    target_compile_definitions(rvjit INTERFACE RVJIT_DEBUG)
endif()

target_link_libraries(rvjit INTERFACE xbyak_riscv::xbyak_riscv)

# The headers stay C++11 even if the project standard is raised
add_library(rvjit_cxx11_check OBJECT cmake/cxx11_check.cpp)
target_link_libraries(rvjit_cxx11_check PRIVATE rvjit::rvjit)
set_target_properties(rvjit_cxx11_check PROPERTIES
    CXX_STANDARD 11 CXX_STANDARD_REQUIRED ON CXX_EXTENSIONS OFF)

include(GNUInstallDirs)
install(
    TARGETS rvjit
    EXPORT rvjit-targets
    INCLUDES DESTINATION ${CMAKE_INSTALL_INCLUDEDIR}
)
install(DIRECTORY rvjit DESTINATION ${CMAKE_INSTALL_INCLUDEDIR})

include(CMakePackageConfigHelpers)
configure_package_config_file(
    cmake/config.cmake.in
    "${CMAKE_CURRENT_BINARY_DIR}/rvjit-config.cmake"
    INSTALL_DESTINATION ${CMAKE_INSTALL_LIBDIR}/cmake/rvjit
)
# Before 1.0, a minor release may break the interface
if(PROJECT_VERSION_MAJOR EQUAL 0)
    set(RVJIT_COMPATIBILITY SameMinorVersion)
else()
    set(RVJIT_COMPATIBILITY SameMajorVersion)
endif()
write_basic_package_version_file(
    "${CMAKE_CURRENT_BINARY_DIR}/rvjit-config-version.cmake"
    COMPATIBILITY ${RVJIT_COMPATIBILITY}
)
install(
    FILES
        "${CMAKE_CURRENT_BINARY_DIR}/rvjit-config.cmake"
        "${CMAKE_CURRENT_BINARY_DIR}/rvjit-config-version.cmake"
    DESTINATION ${CMAKE_INSTALL_LIBDIR}/cmake/rvjit
)
install(
    EXPORT rvjit-targets
    NAMESPACE rvjit::
    DESTINATION ${CMAKE_INSTALL_LIBDIR}/cmake/rvjit
)

if(RVJIT_BUILD_TESTS)
    enable_testing()
    include(cmake/interface_check.cmake)
    add_subdirectory(test)
endif()

if(RVJIT_BUILD_BENCH)
    add_subdirectory(test/bench)
endif()

if(RVJIT_BUILD_DOCS)
    find_package(Doxygen REQUIRED dot)

    include(FetchContent)
    FetchContent_Declare(
        doxygen-awesome-css
        GIT_REPOSITORY https://github.com/jothepro/doxygen-awesome-css.git
        GIT_TAG v2.4.2
    )
    FetchContent_MakeAvailable(doxygen-awesome-css)
    FetchContent_GetProperties(doxygen-awesome-css SOURCE_DIR AWESOME_CSS_DIR)

    set(DOXYGEN_PROJECT_NUMBER ${PROJECT_VERSION})
    set(DOXYGEN_OUTPUT_DIRECTORY ${CMAKE_CURRENT_BINARY_DIR}/docs)
    set(DOXYGEN_GENERATE_LATEX NO)
    set(DOXYGEN_EXTRACT_ALL NO)
    set(DOXYGEN_WARN_AS_ERROR FAIL_ON_WARNINGS)
    # Per-member coverage isn't the right bar: most flagged members are
    # constructors/fields the project's own style deliberately leaves
    # uncommented. Structural doc errors (broken @ref, a bad \example) still
    # fail the build.
    set(DOXYGEN_WARN_IF_UNDOCUMENTED NO)
    set(DOXYGEN_RECURSIVE YES)
    set(DOXYGEN_HAVE_DOT YES)
    set(DOXYGEN_USE_MDFILE_AS_MAINPAGE ${CMAKE_SOURCE_DIR}/docs/mainpage.md)
    set(DOXYGEN_HTML_EXTRA_STYLESHEET ${AWESOME_CSS_DIR}/doxygen-awesome.css)

    doxygen_add_docs(rvjit_docs
        ${CMAKE_SOURCE_DIR}/docs/mainpage.md
        ${CMAKE_SOURCE_DIR}/rvjit
        COMMENT "Generating rvjit API documentation with Doxygen"
    )
endif()
