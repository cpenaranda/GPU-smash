/*
 * rCUDA: remote CUDA (www.rCUDA.net)
 * Copyright (C) 2016-2022
 * Grupo de Arquitecturas Paralelas
 * Departamento de Informática de Sistemas y Computadores
 * Universidad Politécnica de Valencia (Spain)
 */

#include <cuda_runtime.h>

// SMASH LIBRARIES
#include <gpu_compression_libraries.hpp>
#include <gpu_smash.hpp>

bool GpuSmash::SetOptionsCompressor(GpuOptions *options, cudaStream_t stream) {
  return lib->SetOptionsCompressor(options, stream);
}

bool GpuSmash::SetOptionsDecompressor(GpuOptions *options,
                                      cudaStream_t stream) {
  return lib->SetOptionsDecompressor(options, stream);
}

void GpuSmash::GetCompressedDataSizeFromDeviceMemory(
    char *device_uncompressed_data, uint64_t uncompressed_size,
    uint64_t *compressed_size) {
  lib->GetCompressedDataSizeFromDeviceMemory(
      device_uncompressed_data, uncompressed_size, compressed_size);
}

void GpuSmash::GetCompressedDataSizeFromHostMemory(char *host_uncompressed_data,
                                                   uint64_t uncompressed_size,
                                                   uint64_t *compressed_size) {
  lib->GetCompressedDataSizeFromHostMemory(host_uncompressed_data,
                                           uncompressed_size, compressed_size);
}

bool GpuSmash::CompressDeviceMemory(char *device_uncompressed_data,
                                    uint64_t uncompressed_size,
                                    char *device_compressed_data,
                                    uint64_t *compressed_size) {
  return lib->CompressDeviceMemory(device_uncompressed_data, uncompressed_size,
                                   device_compressed_data, compressed_size);
}

bool GpuSmash::CompressHostMemory(char *host_uncompressed_data,
                                  uint64_t uncompressed_size,
                                  char *host_compressed_data,
                                  uint64_t *compressed_size) {
  return lib->CompressHostMemory(host_uncompressed_data, uncompressed_size,
                                 host_compressed_data, compressed_size);
}

void GpuSmash::GetDecompressedDataSizeFromDeviceMemory(
    char *device_compressed_data, uint64_t compressed_size,
    uint64_t *decompressed_size) {
  lib->GetDecompressedDataSizeFromDeviceMemory(
      device_compressed_data, compressed_size, decompressed_size);
}

void GpuSmash::GetDecompressedDataSizeFromHostMemory(
    char *host_compressed_data, uint64_t compressed_size,
    uint64_t *decompressed_size) {
  lib->GetDecompressedDataSizeFromHostMemory(
      host_compressed_data, compressed_size, decompressed_size);
}

bool GpuSmash::DecompressDeviceMemory(char *device_compressed_data,
                                      uint64_t compressed_size,
                                      char *device_decompressed_data,
                                      uint64_t *decompressed_size) {
  return lib->DecompressDeviceMemory(device_compressed_data, compressed_size,
                                     device_decompressed_data,
                                     decompressed_size);
}

bool GpuSmash::DecompressHostMemory(char *host_compressed_data,
                                    uint64_t compressed_size,
                                    char *host_decompressed_data,
                                    uint64_t *decompressed_size) {
  return lib->DecompressHostMemory(host_compressed_data, compressed_size,
                                   host_decompressed_data, decompressed_size);
}

void GpuSmash::GetTitle() { lib->GetTitle(); }

bool GpuSmash::CompareDataDeviceMemory(char *uncompressed_data,
                                       const uint64_t &uncompressed_size,
                                       char *decompressed_data,
                                       const uint64_t &decompressed_size) {
  return lib->CompareDataDeviceMemory(uncompressed_data, uncompressed_size,
                                      decompressed_data, decompressed_size);
}

bool GpuSmash::CompareDataHostMemory(char *uncompressed_data,
                                     const uint64_t &uncompressed_size,
                                     char *decompressed_data,
                                     const uint64_t &decompressed_size) {
  return lib->CompareDataHostMemory(uncompressed_data, uncompressed_size,
                                    decompressed_data, decompressed_size);
}

bool GpuSmash::GetCompressionLevelInformation(
    std::vector<std::string> *compression_level_information,
    uint8_t *minimum_level, uint8_t *maximum_level) {
  return lib->GetCompressionLevelInformation(compression_level_information,
                                             minimum_level, maximum_level);
}

bool GpuSmash::GetWindowSizeInformation(
    std::vector<std::string> *window_size_information, uint32_t *minimum_size,
    uint32_t *maximum_size) {
  return lib->GetWindowSizeInformation(window_size_information, minimum_size,
                                       maximum_size);
}

bool GpuSmash::GetModeInformation(std::vector<std::string> *mode_information,
                                  uint8_t *minimum_mode, uint8_t *maximum_mode,
                                  const uint8_t &compression_level) {
  return lib->GetModeInformation(mode_information, minimum_mode, maximum_mode,
                                 compression_level);
}

bool GpuSmash::GetWorkFactorInformation(
    std::vector<std::string> *work_factor_information, uint8_t *minimum_factor,
    uint8_t *maximum_factor) {
  return lib->GetWorkFactorInformation(work_factor_information, minimum_factor,
                                       maximum_factor);
}

bool GpuSmash::GetFlagsInformation(std::vector<std::string> *flags_information,
                                   uint8_t *minimum_flags,
                                   uint8_t *maximum_flags) {
  return lib->GetFlagsInformation(flags_information, minimum_flags,
                                  maximum_flags);
}

bool GpuSmash::GetChunkSizeInformation(
    std::vector<std::string> *chunk_size_information,
    uint8_t *minimum_chunk_size, uint8_t *maximum_chunk_size) {
  return lib->GetChunkSizeInformation(chunk_size_information,
                                      minimum_chunk_size, maximum_chunk_size);
}

bool GpuSmash::GetBackReferenceBitsInformation(
    std::vector<std::string> *back_reference_bits_information,
    uint8_t *minimum_bits, uint8_t *maximum_bits) {
  return lib->GetBackReferenceBitsInformation(back_reference_bits_information,
                                              minimum_bits, maximum_bits);
}

std::string GpuSmash::GetModeName(const uint8_t &mode) {
  return lib->GetModeName(mode);
}

std::string GpuSmash::GetFlagsName(const uint8_t &flags) {
  return lib->GetFlagsName(flags);
}

GpuOptions GpuSmash::GetOptions() { return lib->GetOptions(); }

GpuSmash::GpuSmash(const std::string &compression_library_name) {
  lib =
      GpuCompressionLibraries().GetCompressionLibrary(compression_library_name);
}

GpuSmash::~GpuSmash() { delete lib; }
