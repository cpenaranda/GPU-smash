cmake_minimum_required(VERSION 3.15 FATAL_ERROR)


set(CULZSS_SOURCES
  ${CMAKE_CURRENT_SOURCE_DIR}/CULZSS/gpu_compress.cu
  ${CMAKE_CURRENT_SOURCE_DIR}/CULZSS/gpu_decompress.cu
)

include_directories(
  ${CUDA_INCLUDE_DIRS}
  ${CMAKE_CURRENT_SOURCE_DIR}/CULZSS
)

cuda_add_library(culzss STATIC ${CULZSS_SOURCES})

target_link_libraries(culzss ${CUDA_LIBRARIES})