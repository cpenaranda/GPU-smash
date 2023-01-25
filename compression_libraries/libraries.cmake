if(NOT CULZSS MATCHES OFF)
  add_subdirectory(compression_libraries/culzss_)
  set(SMASH_LIBRARIES ${SMASH_LIBRARIES} culzss)
  set(SMASH_INCLUDES ${SMASH_INCLUDES}
    ${CMAKE_CURRENT_SOURCE_DIR}/compression_libraries/culzss_/include
    ${CMAKE_CURRENT_SOURCE_DIR}/compression_libraries/culzss_/CULZSS
  )
  set(SMASH_SOURCES ${SMASH_SOURCES}
    ${CMAKE_CURRENT_SOURCE_DIR}/compression_libraries/culzss_/src/culzss_library.cpp
  )
  add_definitions(-DCULZSS)
endif()

if(NOT DIETGPU MATCHES OFF)
  add_subdirectory(compression_libraries/dietgpu_)
  set(SMASH_LIBRARIES ${SMASH_LIBRARIES} dietgpu)
  set(SMASH_INCLUDES ${SMASH_INCLUDES}
    ${CMAKE_CURRENT_SOURCE_DIR}/compression_libraries/dietgpu_/include
    ${CMAKE_CURRENT_SOURCE_DIR}/compression_libraries/dietgpu_/dietgpu
  )
  set(SMASH_SOURCES ${SMASH_SOURCES}
    ${CMAKE_CURRENT_SOURCE_DIR}/compression_libraries/dietgpu_/src/dietgpu_ans_library.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/compression_libraries/dietgpu_/src/dietgpu_float_library.cpp
  )
  add_definitions(-DDIETGPU)
endif()

if(NOT NVCOMP MATCHES OFF)
  find_library(NVCOMP_LIBRARY
    NAMES nvcomp
    HINTS "${CMAKE_CURRENT_SOURCE_DIR}/compression_libraries/nvcomp_/nvcomp/lib"
  )
  add_subdirectory(compression_libraries/nvcomp_)
  set(SMASH_LIBRARIES ${SMASH_LIBRARIES} ${NVCOMP_LIBRARY} nvcomp_util)
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
