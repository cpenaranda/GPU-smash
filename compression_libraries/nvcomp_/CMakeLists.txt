cmake_minimum_required(VERSION 3.15 FATAL_ERROR)

SET(NAME_TAR "nvcomp_2.4.1_x86_64_11.x.tgz")
IF (NOT EXISTS ${CMAKE_SOURCE_DIR}/compression_libraries/nvcomp_/nvcomp/${NAME_TAR})
  FILE(DOWNLOAD https://developer.download.nvidia.com/compute/nvcomp/2.4.1/local_installers/${NAME_TAR}
       ${CMAKE_SOURCE_DIR}/compression_libraries/nvcomp_/nvcomp/${NAME_TAR}
  )

  FILE(ARCHIVE_EXTRACT INPUT ${CMAKE_SOURCE_DIR}/compression_libraries/nvcomp_/nvcomp/${NAME_TAR}
       DESTINATION ${CMAKE_SOURCE_DIR}/compression_libraries/nvcomp_/nvcomp/
  )
ENDIF()

set(NVCOMP_UTIL_SOURCES
  ${CMAKE_CURRENT_SOURCE_DIR}/src/nvcomp_util.cu
)

include_directories(
  ${CUDA_INCLUDE_DIRS}
  ${CMAKE_CURRENT_SOURCE_DIR}/include
)

cuda_add_library(nvcomp_util STATIC ${NVCOMP_UTIL_SOURCES})

target_link_libraries(nvcomp_util ${CUDA_LIBRARIES})
