/*
 * rCUDA: remote CUDA (www.rCUDA.net)
 * Copyright (C) 2016-2022
 * Grupo de Arquitecturas Paralelas
 * Departamento de Informática de Sistemas y Computadores
 * Universidad Politécnica de Valencia (Spain)
 */

#pragma once

#include <cuda_runtime.h>

#include <iostream>
#include <string>
#include <vector>

// GPU-SMASH LIBRARIES
#include <gpu_options.hpp>

class GpuCompressionLibrary {
 private:
  virtual void GetCompressedDataSize(const uint64_t &uncompressed_data_size,
                                     uint64_t *compressed_data_size);

 public:
  GpuOptions options_;
  cudaStream_t stream_;

  bool initialized_compressor_;
  bool initialized_decompressor_;

  virtual bool CheckOptions(GpuOptions *options, const bool &compressor) = 0;

  virtual bool SetOptionsCompressor(GpuOptions *options,
                                    const cudaStream_t &stream);

  virtual bool SetOptionsDecompressor(GpuOptions *options,
                                      const cudaStream_t &stream);

  virtual void GetCompressedDataSizeFromDeviceMemory(
      const char *const device_uncompressed_data,
      const uint64_t &uncompressed_data_size, uint64_t *compressed_data_size);

  virtual void GetCompressedDataSizeFromHostMemory(
      const char *const host_uncompressed_data,
      const uint64_t &uncompressed_data_size, uint64_t *compressed_data_size);

  virtual bool CompressDeviceToDevice(
      const char *const device_uncompressed_data,
      const uint64_t &uncompressed_data_size, char *device_compressed_data,
      uint64_t *compressed_data_size);

  virtual bool CompressDeviceToHost(const char *const device_uncompressed_data,
                                    const uint64_t &uncompressed_data_size,
                                    char *host_compressed_data,
                                    uint64_t *compressed_data_size);

  virtual bool CompressHostToDevice(const char *const host_uncompressed_data,
                                    const uint64_t &uncompressed_data_size,
                                    char *device_compressed_data,
                                    uint64_t *compressed_data_size);

  virtual bool CompressHostToHost(const char *const host_uncompressed_data,
                                  const uint64_t &uncompressed_data_size,
                                  char *host_compressed_data,
                                  uint64_t *compressed_data_size);

  virtual void GetDecompressedDataSizeFromDeviceMemory(
      const char *const device_compressed_data,
      const uint64_t &compressed_data_size, uint64_t *decompressed_data_size);

  virtual void GetDecompressedDataSizeFromHostMemory(
      const char *const host_compressed_data,
      const uint64_t &compressed_data_size, uint64_t *decompressed_data_size);

  virtual bool DecompressDeviceToDevice(
      const char *const device_compressed_data,
      const uint64_t &compressed_data_size, char *device_decompressed_data,
      uint64_t *decompressed_data_size);

  virtual bool DecompressDeviceToHost(const char *const device_compressed_data,
                                      const uint64_t &compressed_data_size,
                                      char *host_decompressed_data,
                                      uint64_t *decompressed_data_size);

  virtual bool DecompressHostToDevice(const char *const host_compressed_data,
                                      const uint64_t &compressed_data_size,
                                      char *device_decompressed_data,
                                      uint64_t *decompressed_data_size);

  virtual bool DecompressHostToHost(const char *const host_compressed_data,
                                    const uint64_t &compressed_data_size,
                                    char *host_decompressed_data,
                                    uint64_t *decompressed_data_size);

  virtual void GetTitle() = 0;

  void GetTitle(const std::string &library_name,
                const std::string &description);

  virtual bool GetCompressionLevelInformation(
      std::vector<std::string> *compression_level_information = nullptr,
      uint8_t *minimum_level = nullptr, uint8_t *maximum_level = nullptr);

  virtual bool GetWindowSizeInformation(
      std::vector<std::string> *window_size_information = nullptr,
      uint32_t *minimum_size = nullptr, uint32_t *maximum_size = nullptr);

  virtual bool GetModeInformation(
      std::vector<std::string> *mode_information = nullptr,
      uint8_t *minimum_mode = nullptr, uint8_t *maximum_mode = nullptr,
      const uint8_t &compression_level = 0);

  virtual bool GetWorkFactorInformation(
      std::vector<std::string> *work_factor_information = nullptr,
      uint8_t *minimum_factor = nullptr, uint8_t *maximum_factor = nullptr);

  virtual bool GetFlagsInformation(
      std::vector<std::string> *flags_information = nullptr,
      uint8_t *minimum_flags = nullptr, uint8_t *maximum_flags = nullptr);

  virtual bool GetChunkSizeInformation(
      std::vector<std::string> *chunk_size_information = nullptr,
      uint8_t *minimum_chunk_size = nullptr,
      uint8_t *maximum_chunk_size = nullptr);

  virtual bool GetChunkNumberInformation(
      std::vector<std::string> *chunk_number_information = nullptr,
      uint8_t *minimum_chunks = nullptr, uint8_t *maximum_chunks = nullptr);

  virtual bool GetStreamNumberInformation(
      std::vector<std::string> *stream_number_information = nullptr,
      uint8_t *minimum_streams = nullptr, uint8_t *maximum_streams = nullptr);

  virtual bool GetBackReferenceInformation(
      std::vector<std::string> *back_reference_information = nullptr,
      uint8_t *minimum_back_reference = nullptr,
      uint8_t *maximum_back_reference = nullptr);

  virtual std::string GetModeName(const uint8_t &mode);

  virtual std::string GetFlagsName(const uint8_t &flags);

  bool CheckCompressionLevel(const std::string &library_name,
                             GpuOptions *options, const uint8_t &minimum_level,
                             const uint8_t &maximum_level);

  bool CheckWindowSize(const std::string &library_name, GpuOptions *options,
                       const uint32_t &minimum_size,
                       const uint32_t &maximum_size);

  bool CheckMode(const std::string &library_name, GpuOptions *options,
                 const uint8_t &minimum_mode, const uint8_t &maximum_mode);

  bool CheckWorkFactor(const std::string &library_name, GpuOptions *options,
                       const uint8_t &minimum_factor,
                       const uint8_t &maximum_factor);

  bool CheckFlags(const std::string &library_name, GpuOptions *options,
                  const uint8_t &minimum_flags, const uint8_t &maximum_flags);

  bool CheckChunkSize(const std::string &library_name, GpuOptions *options,
                      const uint8_t &minimum_chunk_size,
                      const uint8_t &maximum_chunk_size);

  bool CheckChunkNumber(const std::string &library_name, GpuOptions *options,
                        const uint8_t &minimum_chunks,
                        const uint8_t &maximum_chunks);

  bool CheckStreamNumber(const std::string &library_name, GpuOptions *options,
                         const uint8_t &minimum_streams,
                         const uint8_t &maximum_streams);

  bool CheckBackReference(const std::string &library_name, GpuOptions *options,
                          const uint8_t &minimum_back_reference,
                          const uint8_t &maximum_back_reference);

  bool CompareDataDeviceMemory(const char *const device_uncompressed_data,
                               const uint64_t &uncompressed_data_size,
                               const char *const device_decompressed_data,
                               const uint64_t &decompressed_data_size);

  bool CompareDataHostMemory(const char *const host_uncompressed_data,
                             const uint64_t &uncompressed_data_size,
                             const char *const host_decompressed_data,
                             const uint64_t &decompressed_data_size);

  virtual GpuOptions GetOptions();

  GpuCompressionLibrary();
  virtual ~GpuCompressionLibrary();
};
