cmake_minimum_required(VERSION 3.15 FATAL_ERROR)

set(TARGET_NAME gpu_smash)

project(${TARGET_NAME} DESCRIPTION "GPU-Smash: GPU Compresion Abstraction Library")

# Set the debug or relese mode.
if (CMAKE_BUILD_TYPE MATCHES Debug) # Debug level
  # Defines
  add_definitions(-DDEBUG)

  # Flags
  set(CMAKE_CXX_FLAGS_DEBUG "${CMAKE_CXX_FLAGS_DEBUG} -Wall")
  if(CMAKE_COMPILER_IS_GNUCXX)
    set(CMAKE_CXX_FLAGS_DEBUG "${CMAKE_CXX_FLAGS_DEBUG} -Wextra -Wundef")
  endif(CMAKE_COMPILER_IS_GNUCXX)
elseif (CMAKE_BUILD_TYPE MATCHES Release) # Optimization level
else ()
  message(FATAL_ERROR "Set the build type with -DCMAKE_BUILD_TYPE=<type>")
endif()

# Check and add the C++ version.
include(CheckCXXCompilerFlag)
CHECK_CXX_COMPILER_FLAG("-std=c++20" COMPILER_SUPPORTS_CXX20)
CHECK_CXX_COMPILER_FLAG("-std=c++17" COMPILER_SUPPORTS_CXX17)
CHECK_CXX_COMPILER_FLAG("-std=c++14" COMPILER_SUPPORTS_CXX14)
CHECK_CXX_COMPILER_FLAG("-std=c++11" COMPILER_SUPPORTS_CXX11)
CHECK_CXX_COMPILER_FLAG("-std=c++0x" COMPILER_SUPPORTS_CXX0X)
if(COMPILER_SUPPORTS_CXX20)
  set(CMAKE_CXX_STANDARD 20)
elseif(COMPILER_SUPPORTS_CXX17)
  set(CMAKE_CXX_STANDARD 17)
elseif(COMPILER_SUPPORTS_CXX14)
  set(CMAKE_CXX_STANDARD 14)
elseif(COMPILER_SUPPORTS_CXX11)
  set(CMAKE_CXX_STANDARD 11)
elseif(COMPILER_SUPPORTS_CXX0X)
  set(CMAKE_CXX_STANDARD 0X)
else()
  message(STATUS "The compiler ${CMAKE_CXX_COMPILER} has no C++0x support. Please use a different C++ compiler.")
endif()

SET(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -pthread")

set(CMAKE_ARCHIVE_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}/lib)
set(CMAKE_LIBRARY_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}/lib)
set(CMAKE_RUNTIME_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}/bin)

set(ALLOW_DUPLICATE_CUSTOM_TARGETS TRUE)


find_package(CUDA REQUIRED QUIET)

set(CUDA_NVCC_FLAGS ${CUDA_NVCC_FLAGS} "-Wno-deprecated-gpu-targets")
set(CUDA_NVCC_FLAGS ${CUDA_NVCC_FLAGS} "--disable-warnings")

if(${CUDA_VERSION_MAJOR} VERSION_LESS 8)
  # CUDA < 8
  message(FATAL_ERROR "8.0 is the minimum version of CUDA supported")
elseif(${CUDA_VERSION_MAJOR} VERSION_LESS 9)
  # CUDA 8
  set(CUDA_NVCC_FLAGS ${CUDA_NVCC_FLAGS} "-gencode arch=compute_20,code=[sm_20,compute_20]")
  set(CUDA_NVCC_FLAGS ${CUDA_NVCC_FLAGS} "-gencode arch=compute_30,code=[sm_30,compute_30]")
  set(CUDA_NVCC_FLAGS ${CUDA_NVCC_FLAGS} "-gencode arch=compute_35,code=[sm_35,compute_35]")
  set(CUDA_NVCC_FLAGS ${CUDA_NVCC_FLAGS} "-gencode arch=compute_37,code=[sm_37,compute_37]")
  set(CUDA_NVCC_FLAGS ${CUDA_NVCC_FLAGS} "-gencode arch=compute_50,code=[sm_50,compute_50]")
  set(CUDA_NVCC_FLAGS ${CUDA_NVCC_FLAGS} "-gencode arch=compute_52,code=[sm_52,compute_52]")
  set(CUDA_NVCC_FLAGS ${CUDA_NVCC_FLAGS} "-gencode arch=compute_53,code=[sm_53,compute_53]")
  set(CUDA_NVCC_FLAGS ${CUDA_NVCC_FLAGS} "-gencode arch=compute_60,code=[sm_60,compute_60]")
  set(CUDA_NVCC_FLAGS ${CUDA_NVCC_FLAGS} "-gencode arch=compute_61,code=[sm_61,compute_61]")
  set(CUDA_NVCC_FLAGS ${CUDA_NVCC_FLAGS} "-gencode arch=compute_62,code=[sm_62,compute_62]")
elseif(${CUDA_VERSION_MAJOR} VERSION_LESS 10)
  # CUDA 9
  set(CUDA_NVCC_FLAGS ${CUDA_NVCC_FLAGS} "-gencode arch=compute_20,code=[sm_20,compute_20]")
  set(CUDA_NVCC_FLAGS ${CUDA_NVCC_FLAGS} "-gencode arch=compute_30,code=[sm_30,compute_30]")
  set(CUDA_NVCC_FLAGS ${CUDA_NVCC_FLAGS} "-gencode arch=compute_35,code=[sm_35,compute_35]")
  set(CUDA_NVCC_FLAGS ${CUDA_NVCC_FLAGS} "-gencode arch=compute_37,code=[sm_37,compute_37]")
  set(CUDA_NVCC_FLAGS ${CUDA_NVCC_FLAGS} "-gencode arch=compute_50,code=[sm_50,compute_50]")
  set(CUDA_NVCC_FLAGS ${CUDA_NVCC_FLAGS} "-gencode arch=compute_52,code=[sm_52,compute_52]")
  set(CUDA_NVCC_FLAGS ${CUDA_NVCC_FLAGS} "-gencode arch=compute_53,code=[sm_53,compute_53]")
  set(CUDA_NVCC_FLAGS ${CUDA_NVCC_FLAGS} "-gencode arch=compute_60,code=[sm_60,compute_60]")
  set(CUDA_NVCC_FLAGS ${CUDA_NVCC_FLAGS} "-gencode arch=compute_61,code=[sm_61,compute_61]")
  set(CUDA_NVCC_FLAGS ${CUDA_NVCC_FLAGS} "-gencode arch=compute_62,code=[sm_62,compute_62]")
  set(CUDA_NVCC_FLAGS ${CUDA_NVCC_FLAGS} "-gencode arch=compute_70,code=[sm_70,compute_70]")
  set(CUDA_NVCC_FLAGS ${CUDA_NVCC_FLAGS} "-gencode arch=compute_72,code=[sm_72,compute_72]")
elseif(${CUDA_VERSION_MAJOR} VERSION_LESS 11)
  # CUDA 10
  set(CUDA_NVCC_FLAGS ${CUDA_NVCC_FLAGS} "-gencode arch=compute_30,code=[sm_30,compute_30]")
  set(CUDA_NVCC_FLAGS ${CUDA_NVCC_FLAGS} "-gencode arch=compute_35,code=[sm_35,compute_35]")
  set(CUDA_NVCC_FLAGS ${CUDA_NVCC_FLAGS} "-gencode arch=compute_37,code=[sm_37,compute_37]")
  set(CUDA_NVCC_FLAGS ${CUDA_NVCC_FLAGS} "-gencode arch=compute_50,code=[sm_50,compute_50]")
  set(CUDA_NVCC_FLAGS ${CUDA_NVCC_FLAGS} "-gencode arch=compute_52,code=[sm_52,compute_52]")
  set(CUDA_NVCC_FLAGS ${CUDA_NVCC_FLAGS} "-gencode arch=compute_53,code=[sm_53,compute_53]")
  set(CUDA_NVCC_FLAGS ${CUDA_NVCC_FLAGS} "-gencode arch=compute_60,code=[sm_60,compute_60]")
  set(CUDA_NVCC_FLAGS ${CUDA_NVCC_FLAGS} "-gencode arch=compute_61,code=[sm_61,compute_61]")
  set(CUDA_NVCC_FLAGS ${CUDA_NVCC_FLAGS} "-gencode arch=compute_62,code=[sm_62,compute_62]")
  set(CUDA_NVCC_FLAGS ${CUDA_NVCC_FLAGS} "-gencode arch=compute_70,code=[sm_70,compute_70]")
  set(CUDA_NVCC_FLAGS ${CUDA_NVCC_FLAGS} "-gencode arch=compute_72,code=[sm_72,compute_72]")
  set(CUDA_NVCC_FLAGS ${CUDA_NVCC_FLAGS} "-gencode arch=compute_75,code=[sm_75,compute_75]")
else()
  # CUDA 11
  set(CUDA_NVCC_FLAGS ${CUDA_NVCC_FLAGS} "-gencode arch=compute_35,code=[sm_35,compute_35]")
  set(CUDA_NVCC_FLAGS ${CUDA_NVCC_FLAGS} "-gencode arch=compute_37,code=[sm_37,compute_37]")
  set(CUDA_NVCC_FLAGS ${CUDA_NVCC_FLAGS} "-gencode arch=compute_50,code=[sm_50,compute_50]")
  set(CUDA_NVCC_FLAGS ${CUDA_NVCC_FLAGS} "-gencode arch=compute_52,code=[sm_52,compute_52]")
  set(CUDA_NVCC_FLAGS ${CUDA_NVCC_FLAGS} "-gencode arch=compute_53,code=[sm_53,compute_53]")
  set(CUDA_NVCC_FLAGS ${CUDA_NVCC_FLAGS} "-gencode arch=compute_60,code=[sm_60,compute_60]")
  set(CUDA_NVCC_FLAGS ${CUDA_NVCC_FLAGS} "-gencode arch=compute_61,code=[sm_61,compute_61]")
  set(CUDA_NVCC_FLAGS ${CUDA_NVCC_FLAGS} "-gencode arch=compute_62,code=[sm_62,compute_62]")
  set(CUDA_NVCC_FLAGS ${CUDA_NVCC_FLAGS} "-gencode arch=compute_70,code=[sm_70,compute_70]")
  set(CUDA_NVCC_FLAGS ${CUDA_NVCC_FLAGS} "-gencode arch=compute_72,code=[sm_72,compute_72]")
  set(CUDA_NVCC_FLAGS ${CUDA_NVCC_FLAGS} "-gencode arch=compute_75,code=[sm_75,compute_75]")
  if(${CUDA_VERSION_MINOR} VERSION_LESS 1)
    set(CUDA_NVCC_FLAGS ${CUDA_NVCC_FLAGS} "-gencode arch=compute_80,code=[sm_80,compute_80]")
  elseif(${CUDA_VERSION_MINOR} VERSION_LESS 4)
    set(CUDA_NVCC_FLAGS ${CUDA_NVCC_FLAGS} "-gencode arch=compute_80,code=[sm_80,compute_80]")
    set(CUDA_NVCC_FLAGS ${CUDA_NVCC_FLAGS} "-gencode arch=compute_86,code=[sm_86,compute_86]")
  else()
    set(CUDA_NVCC_FLAGS ${CUDA_NVCC_FLAGS} "-gencode arch=compute_80,code=[sm_80,compute_80]")
    set(CUDA_NVCC_FLAGS ${CUDA_NVCC_FLAGS} "-gencode arch=compute_86,code=[sm_86,compute_86]")
    set(CUDA_NVCC_FLAGS ${CUDA_NVCC_FLAGS} "-gencode arch=compute_87,code=[sm_87,compute_87]")
  endif()
endif()

include(compression_libraries/libraries.cmake)

set(SMASH_SOURCES
  ${SMASH_SOURCES}
  src/gpu_smash.cpp
  src/gpu_compression_libraries.cpp
  src/gpu_options.cpp
  src/gpu_compression_library.cpp
)

include_directories(
  ${CUDA_INCLUDE_DIRS}
  ${CMAKE_CURRENT_SOURCE_DIR}/include
  ${SMASH_INCLUDES}
)

add_library(${TARGET_NAME} STATIC ${SMASH_SOURCES})
target_link_libraries(${TARGET_NAME} ${SMASH_LIBRARIES})
