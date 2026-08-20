cmake_minimum_required(VERSION 3.16)
project(reboot_keymemrt CXX)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)
if(NOT CMAKE_BUILD_TYPE)
  set(CMAKE_BUILD_TYPE Release)
endif()

# ---------------------------------------------------------------------------
# The plaintext driver and the tests have no dependencies at all: the whole
# ReBoot algorithm is backend generic, so it runs on plain slot vectors without
# OpenFHE.  They are always built.
# ---------------------------------------------------------------------------
add_library(reboot INTERFACE)
target_include_directories(reboot INTERFACE ${CMAKE_CURRENT_SOURCE_DIR}/include)

add_executable(reboot_plain drivers/reboot_plain.cpp)
target_link_libraries(reboot_plain PRIVATE reboot)

enable_testing()
foreach(test test_linalg test_training)
  add_executable(${test} tests/${test}.cpp)
  target_link_libraries(${test} PRIVATE reboot)
  add_test(NAME ${test} COMMAND ${test})
endforeach()

# ---------------------------------------------------------------------------
# The encrypted driver needs OpenFHE (the eymay fork, for compressed rotation
# keys) and the KeyMemRT runtime headers.  Point KEYMEMRT_DIR at a checkout of
# https://github.com/eymay/KeyMemRT; OpenFHE is found through its installed
# CMake package.
# ---------------------------------------------------------------------------
set(KEYMEMRT_DIR "" CACHE PATH "Path to a KeyMemRT checkout (provides include/KeyMemRT.hpp)")

find_package(OpenFHE QUIET)
find_package(OpenMP QUIET)

if(NOT OpenFHE_FOUND)
  message(STATUS "OpenFHE not found - building the plaintext driver and tests only")
elseif(NOT KEYMEMRT_DIR OR NOT EXISTS "${KEYMEMRT_DIR}/include/KeyMemRT.hpp")
  message(STATUS "KEYMEMRT_DIR not set or does not contain include/KeyMemRT.hpp - "
                 "building the plaintext driver and tests only")
else()
  message(STATUS "Building the encrypted driver against OpenFHE ${OpenFHE_VERSION}")
  add_executable(reboot_train drivers/reboot_train.cpp)
  target_link_libraries(reboot_train PRIVATE reboot)
  target_include_directories(reboot_train PRIVATE
    ${KEYMEMRT_DIR}/include
    ${OpenFHE_INCLUDE}
    ${OpenFHE_INCLUDE}/core
    ${OpenFHE_INCLUDE}/pke
    ${OpenFHE_INCLUDE}/binfhe
    ${OpenFHE_INCLUDE}/third-party/include)
  target_link_directories(reboot_train PRIVATE ${OpenFHE_LIBDIR})
  target_link_libraries(reboot_train PRIVATE ${OpenFHE_SHARED_LIBRARIES})
  if(OpenMP_CXX_FOUND)
    target_link_libraries(reboot_train PRIVATE OpenMP::OpenMP_CXX)
  endif()
  target_compile_options(reboot_train PRIVATE ${OpenFHE_CXX_FLAGS})
endif()
