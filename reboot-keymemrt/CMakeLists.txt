cmake_minimum_required(VERSION 3.16)
project(reboot_keymemrt CXX)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)
if(NOT CMAKE_BUILD_TYPE)
  set(CMAKE_BUILD_TYPE Release)
endif()

# {fmt} 12 provides all formatting and logging output.
find_package(fmt 12 REQUIRED)

# The frontend is a plain C++ library: it builds the ReBoot graph, differentiates
# it, lowers the packing to slot operations and prints MLIR.  Nothing here links
# against OpenFHE - the generated MLIR is compiled by KeyMemRT-Compiler, which
# emits the OpenFHE calls and the KeyMemRT key management.
add_library(reboot
  lib/options.cc
  lib/manifest.cc
  lib/ckks_params.cc
  lib/tensor_graph.cc
  lib/autograd.cc
  lib/reboot_model.cc
  lib/slot_graph.cc
  lib/interpreter.cc
  lib/mlir_emitter.cc
)
target_include_directories(reboot PUBLIC ${CMAKE_CURRENT_SOURCE_DIR}/include)
target_link_libraries(reboot PUBLIC fmt::fmt)

add_executable(reboot_emit drivers/reboot_emit.cc)
target_link_libraries(reboot_emit PRIVATE reboot)

add_executable(reboot_eval drivers/reboot_eval.cc)
target_link_libraries(reboot_eval PRIVATE reboot)

enable_testing()
foreach(test test_autograd test_lowering test_emitter test_options)
  add_executable(${test} tests/${test}.cc)
  target_link_libraries(${test} PRIVATE reboot)
  add_test(NAME ${test} COMMAND ${test})
endforeach()

# ---------------------------------------------------------------------------
# The key-memory benchmark needs OpenFHE (the eymay fork) and the KeyMemRT
# runtime headers.  Point KEYMEMRT_DIR at a KeyMemRT checkout to build it; the
# frontend itself never links against either.
#
# The host runner (drivers/reboot_runner.cc) is deliberately not built here: it
# only compiles once keymemrt-translate has produced the generated .cc.  See
# scripts/run_keymemrt.sh.
# ---------------------------------------------------------------------------
set(KEYMEMRT_DIR "" CACHE PATH "Path to a KeyMemRT checkout (include/KeyMemRT.hpp)")

find_package(OpenFHE QUIET)
find_package(OpenMP QUIET)

if(NOT OpenFHE_FOUND)
  message(STATUS "OpenFHE not found - skipping the key-memory benchmark")
elseif(NOT KEYMEMRT_DIR OR NOT EXISTS "${KEYMEMRT_DIR}/include/KeyMemRT.hpp")
  message(STATUS "KEYMEMRT_DIR unset or missing include/KeyMemRT.hpp - "
                 "skipping the key-memory benchmark")
else()
  message(STATUS "Building measure_key_memory against OpenFHE ${OpenFHE_VERSION}")
  add_executable(measure_key_memory tools/measure_key_memory.cc)
  target_link_libraries(measure_key_memory PRIVATE reboot)
  target_include_directories(measure_key_memory PRIVATE
    ${KEYMEMRT_DIR}/include
    ${OpenFHE_INCLUDE}
    ${OpenFHE_INCLUDE}/core
    ${OpenFHE_INCLUDE}/pke
    ${OpenFHE_INCLUDE}/binfhe
    ${OpenFHE_INCLUDE}/third-party/include)
  target_link_directories(measure_key_memory PRIVATE ${OpenFHE_LIBDIR})
  target_link_libraries(measure_key_memory PRIVATE ${OpenFHE_SHARED_LIBRARIES})
  if(OpenMP_CXX_FOUND)
    target_link_libraries(measure_key_memory PRIVATE OpenMP::OpenMP_CXX)
  endif()
endif()
