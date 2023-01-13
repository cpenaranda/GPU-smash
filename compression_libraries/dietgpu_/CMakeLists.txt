cmake_minimum_required(VERSION 3.15 FATAL_ERROR)

add_subdirectory(dietgpu/third_party/glog EXCLUDE_FROM_ALL)

set(DIETGPU_SOURCES
  ${CMAKE_CURRENT_SOURCE_DIR}/dietgpu/dietgpu/utils/DeviceUtils.cpp
  ${CMAKE_CURRENT_SOURCE_DIR}/dietgpu/dietgpu/utils/StackDeviceMemory.cpp
  ${CMAKE_CURRENT_SOURCE_DIR}/dietgpu/dietgpu/float/GpuFloatCompress.cu
  ${CMAKE_CURRENT_SOURCE_DIR}/dietgpu/dietgpu/float/GpuFloatDecompress.cu
  ${CMAKE_CURRENT_SOURCE_DIR}/dietgpu/dietgpu/float/GpuFloatInfo.cu
  ${CMAKE_CURRENT_SOURCE_DIR}/dietgpu/dietgpu/ans/GpuANSDecode.cu
  ${CMAKE_CURRENT_SOURCE_DIR}/dietgpu/dietgpu/ans/GpuANSEncode.cu
  ${CMAKE_CURRENT_SOURCE_DIR}/dietgpu/dietgpu/ans/GpuANSInfo.cu
)

include_directories(
  ${CUDA_INCLUDE_DIRS}
  ${CMAKE_CURRENT_SOURCE_DIR}/dietgpu
)

cuda_add_library(dietgpu STATIC ${DIETGPU_SOURCES})

target_link_libraries(dietgpu
  ${CUDA_LIBRARIES}
  glog::glog
)