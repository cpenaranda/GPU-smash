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
