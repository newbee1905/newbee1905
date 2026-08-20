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
foreach(test test_autograd test_lowering test_emitter)
  add_executable(${test} tests/${test}.cc)
  target_link_libraries(${test} PRIVATE reboot)
  add_test(NAME ${test} COMMAND ${test})
endforeach()
