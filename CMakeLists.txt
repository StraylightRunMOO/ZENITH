cmake_minimum_required(VERSION 3.16)
project(zenith VERSION 1.0.0 LANGUAGES C)

option(ZENITH_BUILD_TESTS "Build ZENITH correctness and regression tests" ON)
option(ZENITH_BUILD_BENCHMARKS "Build ZENITH benchmark suite" ON)
option(ZENITH_BUILD_EXAMPLES "Build ZENITH examples" ON)
option(ZENITH_ENABLE_SANITIZERS "Enable AddressSanitizer and UndefinedBehaviorSanitizer" OFF)
option(ZENITH_STRICT_WARNINGS "Treat compiler warnings as errors" OFF)
option(ZENITH_ENABLE_NATIVE "Enable -march=native/-mtune=native for local benchmarking" OFF)
option(ZENITH_USE_FFTW "Use FFTW3 for large orthonormal DCT-II projections" OFF)
option(ZENITH_USE_OPENMP "Enable OpenMP parallel build and query" OFF)

add_library(zenith INTERFACE)
add_library(zenith::zenith ALIAS zenith)
target_include_directories(zenith INTERFACE
    $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/include>
    $<INSTALL_INTERFACE:include>)
target_compile_features(zenith INTERFACE c_std_99)

if(UNIX AND NOT APPLE)
    target_link_libraries(zenith INTERFACE m)
elseif(APPLE)
    target_link_libraries(zenith INTERFACE m)
endif()

if(ZENITH_USE_FFTW)
    find_package(PkgConfig REQUIRED)
    pkg_check_modules(FFTW3F REQUIRED IMPORTED_TARGET fftw3f)
    target_compile_definitions(zenith INTERFACE ZENITH_USE_FFTW)
    target_link_libraries(zenith INTERFACE PkgConfig::FFTW3F)
endif()

if(ZENITH_USE_OPENMP)
    find_package(OpenMP REQUIRED)
    target_link_libraries(zenith INTERFACE OpenMP::OpenMP_C)
endif()

function(zenith_configure_target target)
    target_link_libraries(${target} PRIVATE zenith::zenith)
    if(MSVC)
        target_compile_options(${target} PRIVATE /W4)
        if(ZENITH_STRICT_WARNINGS)
            target_compile_options(${target} PRIVATE /WX)
        endif()
    else()
        target_compile_options(${target} PRIVATE -Wall -Wextra -Wpedantic)
        if(ZENITH_STRICT_WARNINGS)
            target_compile_options(${target} PRIVATE -Werror)
        endif()
        if(ZENITH_ENABLE_NATIVE)
            target_compile_options(${target} PRIVATE -march=native -mtune=native)
        endif()
        if(ZENITH_ENABLE_SANITIZERS)
            target_compile_options(${target} PRIVATE -O1 -g -fsanitize=address,undefined -fno-omit-frame-pointer)
            target_link_options(${target} PRIVATE -fsanitize=address,undefined)
        endif()
    endif()
endfunction()

if(ZENITH_BUILD_TESTS)
    enable_testing()
    foreach(test_name IN ITEMS test_core test_features test_persistence test_recall
            test_gaussian_modes test_lloyd_max test_backend)
        add_executable(${test_name} tests/${test_name}.c)
        zenith_configure_target(${test_name})
        add_test(NAME ${test_name} COMMAND ${test_name})
    endforeach()
endif()

if(ZENITH_BUILD_BENCHMARKS)
    add_executable(zenith_bench bench/zenith_bench.c)
    zenith_configure_target(zenith_bench)
    add_custom_target(run_benchmarks
        COMMAND zenith_bench --quick --out ${CMAKE_CURRENT_BINARY_DIR}/zenith_benchmarks.csv
        DEPENDS zenith_bench
        WORKING_DIRECTORY ${CMAKE_CURRENT_BINARY_DIR}
        COMMENT "Running the quick ZENITH benchmark suite")
    add_custom_target(run_perf_benchmarks
        COMMAND ${CMAKE_COMMAND} -E make_directory ${CMAKE_CURRENT_BINARY_DIR}/results
        COMMAND ${CMAKE_COMMAND} -E env
            OMP_PROC_BIND=close OMP_PLACES=cores
            ${CMAKE_CURRENT_SOURCE_DIR}/bench/run_all.sh
            $<TARGET_FILE:zenith_bench>
            ${CMAKE_CURRENT_BINARY_DIR}/results
            perf
        DEPENDS zenith_bench
        WORKING_DIRECTORY ${CMAKE_CURRENT_BINARY_DIR}
        COMMENT "Running the N=8192 D=256 perf suite into the build results/ directory")
endif()

if(ZENITH_BUILD_EXAMPLES)
    foreach(example_name IN ITEMS basic weighted_metrics persistence exact_rerank coefficient_query preprocessing_recall)
        add_executable(example_${example_name} examples/${example_name}.c)
        zenith_configure_target(example_${example_name})
    endforeach()
endif()

include(CMakePackageConfigHelpers)
configure_package_config_file(
    cmake/zenithConfig.cmake.in
    ${CMAKE_CURRENT_BINARY_DIR}/zenithConfig.cmake
    INSTALL_DESTINATION lib/cmake/zenith)
write_basic_package_version_file(
    ${CMAKE_CURRENT_BINARY_DIR}/zenithConfigVersion.cmake
    VERSION ${PROJECT_VERSION}
    COMPATIBILITY SameMajorVersion)

install(FILES include/zenith.h DESTINATION include)
install(DIRECTORY include/zenith DESTINATION include)
install(FILES LICENSE README.md DESTINATION share/doc/zenith)
install(TARGETS zenith EXPORT zenithTargets)
install(EXPORT zenithTargets
    NAMESPACE zenith::
    DESTINATION lib/cmake/zenith)
install(FILES
    ${CMAKE_CURRENT_BINARY_DIR}/zenithConfig.cmake
    ${CMAKE_CURRENT_BINARY_DIR}/zenithConfigVersion.cmake
    DESTINATION lib/cmake/zenith)
