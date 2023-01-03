if (NOT NVCOMP MATCHES OFF)
  find_library(NVCOMP_LIBRARY
    NAMES nvcomp
    HINTS "${CMAKE_CURRENT_SOURCE_DIR}/compression_libraries/nvcomp_/nvcomp/lib"
  )
  add_subdirectory(compression_libraries/nvcomp_)
  set(SMASH_LIBRARIES ${SMASH_LIBRARIES} ${NVCOMP_LIBRARY})
  set(SMASH_INCLUDES ${SMASH_INCLUDES}
      ${CMAKE_CURRENT_SOURCE_DIR}/compression_libraries/nvcomp_/include
      ${CMAKE_CURRENT_SOURCE_DIR}/compression_libraries/nvcomp_/nvcomp/include
  )
  set(SMASH_SOURCES ${SMASH_SOURCES}
      ${CMAKE_CURRENT_SOURCE_DIR}/compression_libraries/nvcomp_/src/nvcomp_ans_library.cpp
      ${CMAKE_CURRENT_SOURCE_DIR}/compression_libraries/nvcomp_/src/nvcomp_batch_compressed.cpp
      ${CMAKE_CURRENT_SOURCE_DIR}/compression_libraries/nvcomp_/src/nvcomp_batch_uncompressed.cpp
      ${CMAKE_CURRENT_SOURCE_DIR}/compression_libraries/nvcomp_/src/nvcomp_bitcomp_library.cpp
      ${CMAKE_CURRENT_SOURCE_DIR}/compression_libraries/nvcomp_/src/nvcomp_deflate_library.cpp
      ${CMAKE_CURRENT_SOURCE_DIR}/compression_libraries/nvcomp_/src/nvcomp_gdeflate_library.cpp
      ${CMAKE_CURRENT_SOURCE_DIR}/compression_libraries/nvcomp_/src/nvcomp_lz4_library.cpp
      ${CMAKE_CURRENT_SOURCE_DIR}/compression_libraries/nvcomp_/src/nvcomp_snappy_library.cpp
      ${CMAKE_CURRENT_SOURCE_DIR}/compression_libraries/nvcomp_/src/nvcomp_zstd_library.cpp
  )
  add_definitions(-DNVCOMP)
endif()