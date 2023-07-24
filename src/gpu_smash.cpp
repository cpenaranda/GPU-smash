/*
 * rCUDA: remote CUDA (www.rCUDA.net)
 * Copyright (C) 2016-2022
 * Grupo de Arquitecturas Paralelas
 * Departamento de Informática de Sistemas y Computadores
 * Universidad Politécnica de Valencia (Spain)
 */

#include <cuda_runtime.h>

// GPU-SMASH LIBRARIES
#include <gpu_compression_libraries.hpp>
#include <gpu_smash.hpp>

bool GpuSmash::SetOptionsCompressor(GpuOptions *options,
                                    const cudaStream_t &stream) {
  return library_->SetOptionsCompressor(options, stream);
}

bool GpuSmash::SetOptionsDecompressor(GpuOptions *options,
                                      const cudaStream_t &stream) {
  return library_->SetOptionsDecompressor(options, stream);
}

void GpuSmash::GetCompressedDataSizeFromDeviceMemory(
    const char *const device_uncompressed_data,
    const uint64_t &uncompressed_data_size, uint64_t *compressed_data_size) {
  library_->GetCompressedDataSizeFromDeviceMemory(
      device_uncompressed_data, uncompressed_data_size, compressed_data_size);
}

void GpuSmash::GetCompressedDataSizeFromHostMemory(
    const char *const host_uncompressed_data,
    const uint64_t &uncompressed_data_size, uint64_t *compressed_data_size) {
  library_->GetCompressedDataSizeFromHostMemory(
      host_uncompressed_data, uncompressed_data_size, compressed_data_size);
}

bool GpuSmash::Compress(const char *const uncompressed_data,
                        const uint64_t &uncompressed_data_size,
                        char *compressed_data, uint64_t *compressed_data_size,
                        const CompressionType &type) {
  bool result{true};
  switch (type) {
    case CompressionType::DeviceToDevice:
      result = library_->CompressDeviceToDevice(
          uncompressed_data, uncompressed_data_size, compressed_data,
          compressed_data_size);
      break;
    case CompressionType::DeviceToHost:
      result = library_->CompressDeviceToHost(
          uncompressed_data, uncompressed_data_size, compressed_data,
          compressed_data_size);
      break;
    case CompressionType::HostToDevice:
      result = library_->CompressHostToDevice(
          uncompressed_data, uncompressed_data_size, compressed_data,
          compressed_data_size);
      break;
    case CompressionType::HostToHost:
      result = library_->CompressHostToHost(
          uncompressed_data, uncompressed_data_size, compressed_data,
          compressed_data_size);
      break;
  }
  if (result) {
    result = (cudaSuccess == cudaStreamSynchronize(library_->stream_));
  }
  return result;
}

void GpuSmash::GetDecompressedDataSizeFromDeviceMemory(
    const char *const device_compressed_data,
    const uint64_t &compressed_data_size, uint64_t *decompressed_data_size) {
  library_->GetDecompressedDataSizeFromDeviceMemory(
      device_compressed_data, compressed_data_size, decompressed_data_size);
}

void GpuSmash::GetDecompressedDataSizeFromHostMemory(
    const char *const host_compressed_data,
    const uint64_t &compressed_data_size, uint64_t *decompressed_data_size) {
  library_->GetDecompressedDataSizeFromHostMemory(
      host_compressed_data, compressed_data_size, decompressed_data_size);
}

bool GpuSmash::Decompress(const char *const compressed_data,
                          const uint64_t &compressed_data_size,
                          char *decompressed_data,
                          uint64_t *decompressed_data_size,
                          const CompressionType &type) {
  bool result{true};
  switch (type) {
    case CompressionType::DeviceToDevice:
      result = library_->DecompressDeviceToDevice(
          compressed_data, compressed_data_size, decompressed_data,
          decompressed_data_size);
      break;
    case CompressionType::DeviceToHost:
      result = library_->DecompressDeviceToHost(
          compressed_data, compressed_data_size, decompressed_data,
          decompressed_data_size);
      break;
    case CompressionType::HostToDevice:
      result = library_->DecompressHostToDevice(
          compressed_data, compressed_data_size, decompressed_data,
          decompressed_data_size);
      break;
    case CompressionType::HostToHost:
      result = library_->DecompressHostToHost(
          compressed_data, compressed_data_size, decompressed_data,
          decompressed_data_size);
      break;
  }
  return result;
}

void GpuSmash::GetTitle() { library_->GetTitle(); }

bool GpuSmash::CompareDataDeviceMemory(const char *const uncompressed_data,
                                       const uint64_t &uncompressed_data_size,
                                       const char *const decompressed_data,
                                       const uint64_t &decompressed_data_size) {
  return library_->CompareDataDeviceMemory(
      uncompressed_data, uncompressed_data_size, decompressed_data,
      decompressed_data_size);
}

bool GpuSmash::CompareDataHostMemory(const char *const uncompressed_data,
                                     const uint64_t &uncompressed_data_size,
                                     const char *const decompressed_data,
                                     const uint64_t &decompressed_data_size) {
  return library_->CompareDataHostMemory(
      uncompressed_data, uncompressed_data_size, decompressed_data,
      decompressed_data_size);
}

bool GpuSmash::GetCompressionLevelInformation(
    std::vector<std::string> *compression_level_information,
    uint8_t *minimum_level, uint8_t *maximum_level) {
  return library_->GetCompressionLevelInformation(compression_level_information,
                                                  minimum_level, maximum_level);
}

bool GpuSmash::GetWindowSizeInformation(
    std::vector<std::string> *window_size_information, uint32_t *minimum_size,
    uint32_t *maximum_size) {
  return library_->GetWindowSizeInformation(window_size_information,
                                            minimum_size, maximum_size);
}

bool GpuSmash::GetModeInformation(std::vector<std::string> *mode_information,
                                  uint8_t *minimum_mode, uint8_t *maximum_mode,
                                  const uint8_t &compression_level) {
  return library_->GetModeInformation(mode_information, minimum_mode,
                                      maximum_mode, compression_level);
}

bool GpuSmash::GetWorkFactorInformation(
    std::vector<std::string> *work_factor_information, uint8_t *minimum_factor,
    uint8_t *maximum_factor) {
  return library_->GetWorkFactorInformation(work_factor_information,
                                            minimum_factor, maximum_factor);
}

bool GpuSmash::GetFlagsInformation(std::vector<std::string> *flags_information,
                                   uint8_t *minimum_flags,
                                   uint8_t *maximum_flags) {
  return library_->GetFlagsInformation(flags_information, minimum_flags,
                                       maximum_flags);
}

bool GpuSmash::GetChunkSizeInformation(
    std::vector<std::string> *chunk_size_information,
    uint8_t *minimum_chunk_size, uint8_t *maximum_chunk_size) {
  return library_->GetChunkSizeInformation(
      chunk_size_information, minimum_chunk_size, maximum_chunk_size);
}

bool GpuSmash::GetChunkNumberInformation(
    std::vector<std::string> *chunk_number_information, uint8_t *minimum_chunks,
    uint8_t *maximum_chunks) {
  return library_->GetChunkNumberInformation(chunk_number_information,
                                             minimum_chunks, maximum_chunks);
}

bool GpuSmash::GetStreamNumberInformation(
    std::vector<std::string> *stream_number_information,
    uint8_t *minimum_streams, uint8_t *maximum_streams) {
  return library_->GetStreamNumberInformation(stream_number_information,
                                              minimum_streams, maximum_streams);
}

bool GpuSmash::GetBackReferenceInformation(
    std::vector<std::string> *back_reference_information,
    uint8_t *minimum_back_reference, uint8_t *maximum_back_reference) {
  return library_->GetBackReferenceInformation(back_reference_information,
                                               minimum_back_reference,
                                               maximum_back_reference);
}

std::string GpuSmash::GetModeName(const uint8_t &mode) {
  return library_->GetModeName(mode);
}

std::string GpuSmash::GetFlagsName(const uint8_t &flags) {
  return library_->GetFlagsName(flags);
}

GpuOptions GpuSmash::GetOptions() { return library_->GetOptions(); }

GpuSmash::GpuSmash(const std::string &compression_library_name) {
  library_ =
      GpuCompressionLibraries().GetCompressionLibrary(compression_library_name);
}

GpuSmash::~GpuSmash() { delete library_; }
