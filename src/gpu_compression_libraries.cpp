/*
 * rCUDA: remote CUDA (www.rCUDA.net)
 * Copyright (C) 2016-2022
 * Grupo de Arquitecturas Paralelas
 * Departamento de Informática de Sistemas y Computadores
 * Universidad Politécnica de Valencia (Spain)
 */

#include <iomanip>
#include <iostream>

// SMASH LIBRARIES
#ifdef CULZSS
#include <culzss_library.hpp>
#endif  // CULZSS
#ifdef DIETGPU
#include <dietgpu_ans_library.hpp>
#include <dietgpu_float_library.hpp>
#endif  // DIETGPU
#ifdef NVCOMP
#include <nvcomp_ans_library.hpp>
#include <nvcomp_bitcomp_library.hpp>
#include <nvcomp_deflate_library.hpp>
#include <nvcomp_gdeflate_library.hpp>
#include <nvcomp_lz4_library.hpp>
#include <nvcomp_snappy_library.hpp>
#include <nvcomp_zstd_library.hpp>
#endif  // NVCOMP
#include <gpu_compression_libraries.hpp>

GpuCompressionLibrary *GpuCompressionLibraries::GetCompressionLibrary(
    const std::string &library_name) {
  auto lib = map_.find(library_name);
  if (lib == map_.end()) {
    std::cout << "ERROR: The compression library does not exist" << std::endl;
    exit(EXIT_FAILURE);
  }
  return lib->second();
}

void GpuCompressionLibraries::GetListInformation() {
  GpuCompressionLibrary *library;
  int i = 0;
  for (auto &lib : map_) {
    library = lib.second();
    std::cout << std::right << std::setw(3) << std::setfill(' ') << ++i << ": ";
    library->GetTitle();
    delete library;
  }
}

std::vector<std::string> GpuCompressionLibraries::GetNameLibraries() {
  std::vector<std::string> result;
  for (auto &lib : map_) {
    result.push_back(lib.first);
  }
  return result;
}

GpuCompressionLibraries::GpuCompressionLibraries() {
#ifdef CULZSS
  map_["culzss"] = []() { return new CulzssLibrary(); };
#endif  // CULZSS
#ifdef DIETGPU
  map_["dietgpu-ans"] = []() { return new DietgpuAnsLibrary(); };
  map_["dietgpu-float"] = []() { return new DietgpuFloatLibrary(); };
#endif  // DIETGPU
#ifdef NVCOMP
  map_["nvcomp-ans"] = []() { return new NvcompAnsLibrary(); };
  map_["nvcomp-bitcomp"] = []() { return new NvcompBitcompLibrary(); };
  map_["nvcomp-deflate"] = []() { return new NvcompDeflateLibrary(); };
  map_["nvcomp-gdeflate"] = []() { return new NvcompGdeflateLibrary(); };
  map_["nvcomp-lz4"] = []() { return new NvcompLz4Library(); };
  map_["nvcomp-snappy"] = []() { return new NvcompSnappyLibrary(); };
  map_["nvcomp-zstd"] = []() { return new NvcompZstdLibrary(); };
#endif  // NVCOMP
}

GpuCompressionLibraries::~GpuCompressionLibraries() {}
