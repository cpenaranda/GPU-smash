/*
 * rCUDA: remote CUDA (www.rCUDA.net)
 * Copyright (C) 2016-2022
 * Grupo de Arquitecturas Paralelas
 * Departamento de Informática de Sistemas y Computadores
 * Universidad Politécnica de Valencia (Spain)
 */

#include <string.h>

#include <iomanip>

// GPU-SMASH LIBRARIES
#include <gpu_compression_library.hpp>

bool GpuCompressionLibrary::CheckOptions(GpuOptions *options,
                                         const bool &compressor) {
  return true;
}

bool GpuCompressionLibrary::SetOptionsCompressor(GpuOptions *options,
                                                 const cudaStream_t &stream) {
  stream_ = stream;
  if (initialized_decompressor_) initialized_decompressor_ = false;
  initialized_compressor_ = CheckOptions(options, true);
  if (initialized_compressor_) options_ = *options;
  return initialized_compressor_;
}

bool GpuCompressionLibrary::SetOptionsDecompressor(GpuOptions *options,
                                                   const cudaStream_t &stream) {
  stream_ = stream;
  if (initialized_compressor_) initialized_compressor_ = false;
  initialized_decompressor_ = CheckOptions(options, false);
  if (initialized_decompressor_) options_ = *options;
  return initialized_decompressor_;
}

void GpuCompressionLibrary::GetCompressedDataSize(
    const uint64_t &uncompressed_data_size, uint64_t *compressed_data_size) {
  // There is no way to obtain with the library
  if (uncompressed_data_size < 2500) {
    *compressed_data_size = 5000;
  } else {
    *compressed_data_size = uncompressed_data_size * 2;
  }
}

void GpuCompressionLibrary::GetCompressedDataSizeFromDeviceMemory(
    const char *const device_uncompressed_data,
    const uint64_t &uncompressed_data_size, uint64_t *compressed_data_size) {
  GetCompressedDataSize(uncompressed_data_size, compressed_data_size);
}

void GpuCompressionLibrary::GetCompressedDataSizeFromHostMemory(
    const char *const host_uncompressed_data,
    const uint64_t &uncompressed_data_size, uint64_t *compressed_data_size) {
  GetCompressedDataSize(uncompressed_data_size, compressed_data_size);
}

bool GpuCompressionLibrary::CompressDeviceToDevice(
    const char *const device_uncompressed_data,
    const uint64_t &uncompressed_data_size, char *device_compressed_data,
    uint64_t *compressed_data_size) {
  bool result{initialized_compressor_};
  if (result) {
    char *host_uncompressed_data = new char[uncompressed_data_size];
    result = (cudaSuccess == cudaMemcpyAsync(host_uncompressed_data,
                                             device_uncompressed_data,
                                             uncompressed_data_size,
                                             cudaMemcpyDeviceToHost, stream_));
    if (result) {
      result =
          CompressHostToDevice(host_uncompressed_data, uncompressed_data_size,
                               device_compressed_data, compressed_data_size);
    }
    delete[] host_uncompressed_data;
  }
  return result;
}

bool GpuCompressionLibrary::CompressDeviceToHost(
    const char *const device_uncompressed_data,
    const uint64_t &uncompressed_data_size, char *host_compressed_data,
    uint64_t *compressed_data_size) {
  bool result{initialized_compressor_};
  if (result) {
    GetCompressedDataSizeFromDeviceMemory(
        device_uncompressed_data, uncompressed_data_size, compressed_data_size);
    char *device_compressed_data;
    result = (cudaSuccess ==
              cudaMalloc(&device_compressed_data, *compressed_data_size));
    if (result) {
      result = CompressDeviceToDevice(
          device_uncompressed_data, uncompressed_data_size,
          device_compressed_data, compressed_data_size);
      if (result) {
        result = cudaMemcpy(host_compressed_data, device_compressed_data,
                            *compressed_data_size, cudaMemcpyDeviceToHost);
      }
      result &= (cudaSuccess == cudaFree(device_compressed_data));
    }
  }
  return result;
}

bool GpuCompressionLibrary::CompressHostToDevice(
    const char *const host_uncompressed_data,
    const uint64_t &uncompressed_data_size, char *device_compressed_data,
    uint64_t *compressed_data_size) {
  bool result{initialized_compressor_};
  if (result) {
    GetCompressedDataSizeFromHostMemory(
        host_uncompressed_data, uncompressed_data_size, compressed_data_size);
    char *host_compressed_data = new char[*compressed_data_size];
    result = CompressHostToHost(host_uncompressed_data, uncompressed_data_size,
                                host_compressed_data, compressed_data_size);
    if (result) {
      result = (cudaSuccess ==
                cudaMemcpy(device_compressed_data, host_compressed_data,
                           *compressed_data_size, cudaMemcpyHostToDevice));
    }
    delete[] host_compressed_data;
  }
  return result;
}

bool GpuCompressionLibrary::CompressHostToHost(
    const char *const host_uncompressed_data,
    const uint64_t &uncompressed_data_size, char *host_compressed_data,
    uint64_t *compressed_data_size) {
  bool result{initialized_compressor_};
  if (result) {
    char *device_uncompressed_data;
    result = (cudaSuccess ==
              cudaMalloc(&device_uncompressed_data, uncompressed_data_size));
    if (result) {
      result = (cudaSuccess ==
                cudaMemcpyAsync(device_uncompressed_data,
                                host_uncompressed_data, uncompressed_data_size,
                                cudaMemcpyHostToDevice, stream_));
      if (result) {
        result = CompressDeviceToHost(
            device_uncompressed_data, uncompressed_data_size,
            host_compressed_data, compressed_data_size);
      }
      result &= (cudaSuccess == cudaFree(device_uncompressed_data));
    }
  }
  return result;
}

void GpuCompressionLibrary::GetDecompressedDataSizeFromDeviceMemory(
    const char *const device_compressed_data,
    const uint64_t &compressed_data_size, uint64_t *decompressed_data_size) {
  char *host_compressed_data = new char[compressed_data_size];
  cudaMemcpy(host_compressed_data, device_compressed_data, compressed_data_size,
             cudaMemcpyDeviceToHost);
  GetDecompressedDataSizeFromHostMemory(
      host_compressed_data, compressed_data_size, decompressed_data_size);
  delete[] host_compressed_data;
}

void GpuCompressionLibrary::GetDecompressedDataSizeFromHostMemory(
    const char *const host_compressed_data,
    const uint64_t &compressed_data_size, uint64_t *decompressed_data_size) {
  char *device_compressed_data;
  cudaMalloc(&device_compressed_data, compressed_data_size);
  cudaMemcpy(device_compressed_data, host_compressed_data, compressed_data_size,
             cudaMemcpyHostToDevice);
  GetDecompressedDataSizeFromDeviceMemory(
      device_compressed_data, compressed_data_size, decompressed_data_size);
  cudaFree(device_compressed_data);
}

bool GpuCompressionLibrary::DecompressDeviceToDevice(
    const char *const device_compressed_data,
    const uint64_t &compressed_data_size, char *device_decompressed_data,
    uint64_t *decompressed_data_size) {
  bool result{initialized_decompressor_};
  if (result) {
    char *host_compressed_data = new char[compressed_data_size];
    result = (cudaSuccess == cudaMemcpyAsync(host_compressed_data,
                                             device_compressed_data,
                                             compressed_data_size,
                                             cudaMemcpyDeviceToHost, stream_));
    if (result) {
      result = DecompressHostToDevice(
          host_compressed_data, compressed_data_size, device_decompressed_data,
          decompressed_data_size);
    }
    delete[] host_compressed_data;
  }
  return result;
}

bool GpuCompressionLibrary::DecompressDeviceToHost(
    const char *const device_compressed_data,
    const uint64_t &compressed_data_size, char *host_decompressed_data,
    uint64_t *decompressed_data_size) {
  bool result{initialized_decompressor_};
  if (result) {
    GetDecompressedDataSizeFromDeviceMemory(
        device_compressed_data, compressed_data_size, decompressed_data_size);
    char *device_decompressed_data;
    result = (cudaSuccess ==
              cudaMalloc(&device_decompressed_data, *decompressed_data_size));
    if (result) {
      result = DecompressDeviceToDevice(
          device_compressed_data, compressed_data_size,
          device_decompressed_data, decompressed_data_size);
      if (result) {
        result = cudaMemcpy(host_decompressed_data, device_decompressed_data,
                            *decompressed_data_size, cudaMemcpyDeviceToHost);
      }
      result &= (cudaSuccess == cudaFree(device_decompressed_data));
    }
  }
  return result;
}

bool GpuCompressionLibrary::DecompressHostToDevice(
    const char *const host_compressed_data,
    const uint64_t &compressed_data_size, char *device_decompressed_data,
    uint64_t *decompressed_data_size) {
  bool result{initialized_decompressor_};
  if (result) {
    GetDecompressedDataSizeFromHostMemory(
        host_compressed_data, compressed_data_size, decompressed_data_size);
    char *host_decompressed_data = new char[*decompressed_data_size];
    result =
        DecompressHostToHost(host_compressed_data, compressed_data_size,
                             host_decompressed_data, decompressed_data_size);
    if (result) {
      result = (cudaSuccess ==
                cudaMemcpy(device_decompressed_data, host_decompressed_data,
                           *decompressed_data_size, cudaMemcpyHostToDevice));
    }
    delete[] host_decompressed_data;
  }
  return result;
}

bool GpuCompressionLibrary::DecompressHostToHost(
    const char *const host_compressed_data,
    const uint64_t &compressed_data_size, char *host_decompressed_data,
    uint64_t *decompressed_data_size) {
  bool result{initialized_decompressor_};
  if (result) {
    char *device_compressed_data;
    result = (cudaSuccess ==
              cudaMalloc(&device_compressed_data, compressed_data_size));
    if (result) {
      result = (cudaSuccess ==
                cudaMemcpyAsync(device_compressed_data, host_compressed_data,
                                compressed_data_size, cudaMemcpyHostToDevice,
                                stream_));
      if (result) {
        result = DecompressDeviceToHost(
            device_compressed_data, compressed_data_size,
            host_decompressed_data, decompressed_data_size);
      }
      result &= (cudaSuccess == cudaFree(device_compressed_data));
    }
  }
  return result;
}

void GpuCompressionLibrary::GetTitle(const std::string &library_name,
                                     const std::string &description) {
  std::cout << std::left << std::setw(15) << std::setfill(' ') << library_name
            << "- " << description << std::endl;
}

bool GpuCompressionLibrary::GetCompressionLevelInformation(
    std::vector<std::string> *compression_level_information,
    uint8_t *minimum_level, uint8_t *maximum_level) {
  if (minimum_level) *minimum_level = 0;
  if (maximum_level) *maximum_level = 0;
  if (compression_level_information) compression_level_information->clear();
  return false;
}

bool GpuCompressionLibrary::GetWindowSizeInformation(
    std::vector<std::string> *window_size_information, uint32_t *minimum_size,
    uint32_t *maximum_size) {
  if (minimum_size) *minimum_size = 0;
  if (maximum_size) *maximum_size = 0;
  if (window_size_information) window_size_information->clear();
  return false;
}

bool GpuCompressionLibrary::GetModeInformation(
    std::vector<std::string> *mode_information, uint8_t *minimum_mode,
    uint8_t *maximum_mode, const uint8_t &compression_level) {
  if (minimum_mode) *minimum_mode = 0;
  if (maximum_mode) *maximum_mode = 0;
  if (mode_information) mode_information->clear();
  return false;
}

bool GpuCompressionLibrary::GetWorkFactorInformation(
    std::vector<std::string> *work_factor_information, uint8_t *minimum_factor,
    uint8_t *maximum_factor) {
  if (minimum_factor) *minimum_factor = 0;
  if (maximum_factor) *maximum_factor = 0;
  if (work_factor_information) work_factor_information->clear();
  return false;
}

bool GpuCompressionLibrary::GetFlagsInformation(
    std::vector<std::string> *flags_information, uint8_t *minimum_flags,
    uint8_t *maximum_flags) {
  if (minimum_flags) *minimum_flags = 0;
  if (maximum_flags) *maximum_flags = 0;
  if (flags_information) flags_information->clear();
  return false;
}

bool GpuCompressionLibrary::GetChunkSizeInformation(
    std::vector<std::string> *chunk_size_information,
    uint8_t *minimum_chunk_size, uint8_t *maximum_chunk_size) {
  if (minimum_chunk_size) *minimum_chunk_size = 0;
  if (maximum_chunk_size) *maximum_chunk_size = 0;
  if (chunk_size_information) chunk_size_information->clear();
  return false;
}

bool GpuCompressionLibrary::GetChunkNumberInformation(
    std::vector<std::string> *chunk_number_information, uint8_t *minimum_chunks,
    uint8_t *maximum_chunks) {
  if (minimum_chunks) *minimum_chunks = 0;
  if (maximum_chunks) *maximum_chunks = 0;
  if (chunk_number_information) chunk_number_information->clear();
  return false;
}

bool GpuCompressionLibrary::GetStreamNumberInformation(
    std::vector<std::string> *stream_number_information,
    uint8_t *minimum_streams, uint8_t *maximum_streams) {
  if (minimum_streams) *minimum_streams = 0;
  if (maximum_streams) *maximum_streams = 0;
  if (stream_number_information) stream_number_information->clear();
  return false;
}

bool GpuCompressionLibrary::GetBackReferenceInformation(
    std::vector<std::string> *back_reference_information,
    uint8_t *minimum_back_reference, uint8_t *maximum_back_reference) {
  if (minimum_back_reference) *minimum_back_reference = 0;
  if (maximum_back_reference) *maximum_back_reference = 0;
  if (back_reference_information) back_reference_information->clear();
  return false;
}

std::string GpuCompressionLibrary::GetModeName(const uint8_t &mode) {
  return "------------";
}

std::string GpuCompressionLibrary::GetFlagsName(const uint8_t &flags) {
  return "------------";
}

bool GpuCompressionLibrary::CheckCompressionLevel(
    const std::string &library_name, GpuOptions *options,
    const uint8_t &minimum_level, const uint8_t &maximum_level) {
  bool result{true};
  if (options->CompressionLevelIsSet()) {
    if (options->GetCompressionLevel() < minimum_level) {
      std::cout << "ERROR: Compression level can not be lower than "
                << static_cast<uint64_t>(minimum_level) << " using "
                << library_name << std::endl;
      result = false;
    } else if (options->GetCompressionLevel() > maximum_level) {
      std::cout << "ERROR: Compression level can not be higher than "
                << static_cast<uint64_t>(maximum_level) << " using "
                << library_name << std::endl;
      result = false;
    }
  } else {
    options->SetCompressionLevel(minimum_level);
  }
  return result;
}

bool GpuCompressionLibrary::CheckWindowSize(const std::string &library_name,
                                            GpuOptions *options,
                                            const uint32_t &minimum_size,
                                            const uint32_t &maximum_size) {
  bool result{true};
  if (options->WindowSizeIsSet()) {
    if (options->GetWindowSize() < minimum_size) {
      std::cout << "ERROR: Window size can not be lower than "
                << static_cast<uint64_t>(minimum_size) << " using "
                << library_name << std::endl;
      result = false;
    } else if (options->GetWindowSize() > maximum_size) {
      std::cout << "ERROR: Window size can not be higher than "
                << static_cast<uint64_t>(maximum_size) << " using "
                << library_name << std::endl;
      result = false;
    }
  } else {
    options->SetWindowSize(minimum_size);
  }
  return result;
}

bool GpuCompressionLibrary::CheckMode(const std::string &library_name,
                                      GpuOptions *options,
                                      const uint8_t &minimum_mode,
                                      const uint8_t &maximum_mode) {
  bool result{true};
  if (options->ModeIsSet()) {
    if (options->GetMode() < minimum_mode) {
      std::cout << "ERROR: Mode can not be lower than "
                << static_cast<uint64_t>(minimum_mode) << " using "
                << library_name << std::endl;
      result = false;
    } else if (options->GetMode() > maximum_mode) {
      std::cout << "ERROR: Mode can not be higher than "
                << static_cast<uint64_t>(maximum_mode) << " using "
                << library_name << std::endl;
      result = false;
    }
  } else {
    options->SetMode(minimum_mode);
  }
  return result;
}

bool GpuCompressionLibrary::CheckWorkFactor(const std::string &library_name,
                                            GpuOptions *options,
                                            const uint8_t &minimum_factor,
                                            const uint8_t &maximum_factor) {
  bool result{true};
  if (options->WorkFactorIsSet()) {
    if (options->GetWorkFactor() < minimum_factor) {
      std::cout << "ERROR: Work factor can not be lower than "
                << static_cast<uint64_t>(minimum_factor) << " using "
                << library_name << std::endl;
      result = false;
    } else if (options->GetWorkFactor() > maximum_factor) {
      std::cout << "ERROR: Work factor can not be higher than "
                << static_cast<uint64_t>(maximum_factor) << " using "
                << library_name << std::endl;
      result = false;
    }
  } else {
    options->SetWorkFactor(minimum_factor);
  }
  return result;
}

bool GpuCompressionLibrary::CheckFlags(const std::string &library_name,
                                       GpuOptions *options,
                                       const uint8_t &minimum_flags,
                                       const uint8_t &maximum_flags) {
  bool result{true};
  if (options->FlagsIsSet()) {
    if (options->GetFlags() < minimum_flags) {
      std::cout << "ERROR: Flags can not be lower than "
                << static_cast<uint64_t>(minimum_flags) << " using "
                << library_name << std::endl;
      result = false;
    } else if (options->GetFlags() > maximum_flags) {
      std::cout << "ERROR: Flags can not be higher than "
                << static_cast<uint64_t>(maximum_flags) << " using "
                << library_name << std::endl;
      result = false;
    }
  } else {
    options->SetFlags(minimum_flags);
  }
  return result;
}

bool GpuCompressionLibrary::CheckChunkSize(const std::string &library_name,
                                           GpuOptions *options,
                                           const uint8_t &minimum_chunk_size,
                                           const uint8_t &maximum_chunk_size) {
  bool result{true};
  if (options->ChunkSizeIsSet()) {
    if (options->GetChunkSize() < minimum_chunk_size) {
      std::cout << "ERROR: Chunk size can not be lower than "
                << static_cast<uint64_t>(minimum_chunk_size) << " using "
                << library_name << std::endl;
      result = false;
    } else if (options->GetChunkSize() > maximum_chunk_size) {
      std::cout << "ERROR: Chunk size can not be higher than "
                << static_cast<uint64_t>(maximum_chunk_size) << " using "
                << library_name << std::endl;
      result = false;
    }
  } else {
    options->SetChunkSize(minimum_chunk_size);
  }
  return result;
}

bool GpuCompressionLibrary::CheckChunkNumber(const std::string &library_name,
                                             GpuOptions *options,
                                             const uint8_t &minimum_chunks,
                                             const uint8_t &maximum_chunks) {
  bool result{true};
  if (options->ChunkNumberIsSet()) {
    if (options->GetChunkNumber() < minimum_chunks) {
      std::cout << "ERROR: Number of chunks can not be lower than "
                << static_cast<uint64_t>(minimum_chunks) << " using "
                << library_name << std::endl;
      result = false;
    } else if (options->GetChunkNumber() > maximum_chunks) {
      std::cout << "ERROR: Number of chunks can not be higher than "
                << static_cast<uint64_t>(maximum_chunks) << " using "
                << library_name << std::endl;
      result = false;
    }
  } else {
    options->SetChunkNumber(minimum_chunks);
  }
  return result;
}

bool GpuCompressionLibrary::CheckStreamNumber(const std::string &library_name,
                                              GpuOptions *options,
                                              const uint8_t &minimum_streams,
                                              const uint8_t &maximum_streams) {
  bool result{true};
  if (options->StreamNumberIsSet()) {
    if (options->GetStreamNumber() < minimum_streams) {
      std::cout << "ERROR: Number of streams can not be lower than "
                << static_cast<uint64_t>(minimum_streams) << " using "
                << library_name << std::endl;
      result = false;
    } else if (options->GetStreamNumber() > maximum_streams) {
      std::cout << "ERROR: Number of streams can not be higher than "
                << static_cast<uint64_t>(maximum_streams) << " using "
                << library_name << std::endl;
      result = false;
    }
  } else {
    options->SetStreamNumber(minimum_streams);
  }
  return result;
}

bool GpuCompressionLibrary::CheckBackReference(
    const std::string &library_name, GpuOptions *options,
    const uint8_t &minimum_back_reference,
    const uint8_t &maximum_back_reference) {
  bool result{true};
  if (options->BackReferenceIsSet()) {
    if (options->GetBackReference() < minimum_back_reference) {
      std::cout << "ERROR: Back refence bits can not be lower than "
                << static_cast<uint64_t>(minimum_back_reference) << " using "
                << library_name << std::endl;
      result = false;
    } else if (options->GetBackReference() > maximum_back_reference) {
      std::cout << "ERROR: Back refence bits can not be higher than "
                << static_cast<uint64_t>(maximum_back_reference) << " using "
                << library_name << std::endl;
      result = false;
    }
  } else {
    options->SetBackReference(minimum_back_reference);
  }
  return result;
}

bool GpuCompressionLibrary::CompareDataDeviceMemory(
    const char *const device_uncompress_data,
    const uint64_t &uncompressed_data_size,
    const char *const device_decompress_data,
    const uint64_t &decompressed_data_size) {
  bool res = true;
  if ((res = (uncompressed_data_size == decompressed_data_size))) {
    char *uncompressed_data = new char[uncompressed_data_size];
    char *decompressed_data = new char[decompressed_data_size];
    cudaMemcpy(uncompressed_data, device_uncompress_data,
               uncompressed_data_size, cudaMemcpyDeviceToHost);
    cudaMemcpy(decompressed_data, device_decompress_data,
               decompressed_data_size, cudaMemcpyDeviceToHost);
    res = CompareDataHostMemory(uncompressed_data, uncompressed_data_size,
                                decompressed_data, decompressed_data_size);
    delete[] uncompressed_data;
    delete[] decompressed_data;
  }
  return res;
}

bool GpuCompressionLibrary::CompareDataHostMemory(
    const char *const uncompressed_data, const uint64_t &uncompressed_data_size,
    const char *const decompressed_data,
    const uint64_t &decompressed_data_size) {
  return ((uncompressed_data_size == decompressed_data_size) &&
          (memcmp(uncompressed_data, decompressed_data,
                  decompressed_data_size) == 0));
}

GpuOptions GpuCompressionLibrary::GetOptions() { return options_; }

GpuCompressionLibrary::GpuCompressionLibrary() {
  initialized_compressor_ = false;
  initialized_decompressor_ = false;
}

GpuCompressionLibrary::~GpuCompressionLibrary() {}
