/*
 * rCUDA: remote CUDA (www.rCUDA.net)
 * Copyright (C) 2016-2022
 * Grupo de Arquitecturas Paralelas
 * Departamento de Informática de Sistemas y Computadores
 * Universidad Politécnica de Valencia (Spain)
 */

#include <assert.h>
#include <dietgpu/float/GpuFloatCodec.h>
#include <dietgpu/utils/StackDeviceMemory.h>

#include <dietgpu/float/GpuFloatUtils.cuh>

// SMASH LIBRARIES
#include <dietgpu_float_library.hpp>
#include <gpu_options.hpp>

bool DietgpuFloatLibrary::CheckOptions(GpuOptions *options,
                                       const bool &compressor) {
  bool result =
      GpuCompressionLibrary::CheckChunkSize("dietgpu-float", options, 12, 24);
  cudaError_t status = cudaSuccess;
  if ((result = GpuCompressionLibrary::CheckCompressionLevel("dietgpu-float",
                                                             options, 0, 2))) {
    if ((result = GpuCompressionLibrary::CheckMode("dietgpu-float", options, 0,
                                                   1))) {
      if ((result = GpuCompressionLibrary::CheckFlags("dietgpu-float", options,
                                                      0, 2))) {
        uncompressed_chunk_size_ = (1 << options->GetChunkSize());
        configuration_float_ = dietgpu::FloatCompressConfig(
            static_cast<dietgpu::FloatType>(options->GetFlags() + 1),
            dietgpu::ANSCodecConfig(options->GetCompressionLevel() + 9), false);
        float_size_ = getWordSizeFromFloatType(configuration_float_.floatType);
        compressed_chunk_size_ = dietgpu::getMaxFloatCompressedSize(
            configuration_float_.floatType,
            uncompressed_chunk_size_ / float_size_);
        if (reserved_compressed_temporal_memory_) {
          status = cudaFree(host_compressed_ptrs_[0]);
        }
        if (status == cudaSuccess) {
          reserved_compressed_temporal_memory_ = true;
          status = cudaMalloc(&host_compressed_ptrs_[0],
                              batch_size_ * compressed_chunk_size_);
          if (status == cudaSuccess) {
            for (uint64_t i = 1; i < batch_size_; ++i) {
              host_compressed_ptrs_[i] =
                  reinterpret_cast<char *>(host_compressed_ptrs_[0]) +
                  (i * compressed_chunk_size_);
            }
          }
        }
      }
    }
  }
  return result && (status == cudaSuccess);
}

void DietgpuFloatLibrary::GetCompressedDataSize(uint64_t uncompressed_size,
                                                uint64_t *compressed_size) {
  if (initialized_compressor_ || initialized_decompressor_) {
    uint32_t slices = ((uncompressed_size + uncompressed_chunk_size_ - 1) /
                       uncompressed_chunk_size_);
    *compressed_size = sizeof(slices) + (slices * sizeof(slices)) +
                       (slices * compressed_chunk_size_);
  } else {
    *compressed_size = 0;
  }
}

void DietgpuFloatLibrary::GetDecompressedDataSizeFromDeviceMemory(
    char *device_compressed_data, uint64_t compressed_size,
    uint64_t *decompressed_size) {
  if (initialized_compressor_ || initialized_decompressor_) {
    *decompressed_size = 0;
    uint32_t current_batch_size;
    uint32_t min_batch;
    cudaError_t status = cudaSuccess;
    status = cudaMemcpy(&current_batch_size, device_compressed_data,
                        sizeof(current_batch_size), cudaMemcpyDeviceToHost);
    uint32_t *device_compressed_sizes = reinterpret_cast<uint32_t *>(
        device_compressed_data + sizeof(current_batch_size));
    device_compressed_data +=
        sizeof(current_batch_size) +
        sizeof(*device_compressed_sizes) * current_batch_size;
    while ((status == cudaSuccess) && current_batch_size) {
      min_batch =
          current_batch_size < batch_size_ ? current_batch_size : batch_size_;
      status = cudaMemcpy(host_compressed_sizes_, device_compressed_sizes,
                          sizeof(*host_compressed_sizes_) * min_batch,
                          cudaMemcpyDeviceToHost);
      for (uint64_t i = 0; i < min_batch; ++i) {
        status = cudaMemcpyAsync(
            host_compressed_ptrs_[i], device_compressed_data,
            host_compressed_sizes_[i], cudaMemcpyDeviceToDevice, stream_);
        device_compressed_data += host_compressed_sizes_[i];
      }
      floatGetCompressedInfo(
          *temp_memory_, const_cast<const void **>(host_compressed_ptrs_),
          min_batch, device_uncompressed_sizes_, nullptr, nullptr, stream_);
      status = cudaMemcpy(host_uncompressed_sizes_, device_uncompressed_sizes_,
                          sizeof(*host_uncompressed_sizes_) * min_batch,
                          cudaMemcpyDeviceToHost);
      for (uint64_t i = 0; i < min_batch; ++i) {
        *decompressed_size += host_uncompressed_sizes_[i] * float_size_;
      }
    }
  } else {
    *decompressed_size = 0;
  }
}

bool DietgpuFloatLibrary::CompressDeviceMemory(char *device_uncompressed_data,
                                               uint64_t uncompressed_size,
                                               char *device_compressed_data,
                                               uint64_t *compressed_size) {
  bool result{initialized_compressor_};
  if (result) {
    cudaError_t status = cudaSuccess;
    uint32_t min_batch, current_size;
    bool contains_data_without_compress = false;
    char *data_without_compress = nullptr;
    uint32_t size_of_data_without_compress = 0;
    uint32_t current_batch_size =
        (uncompressed_size + uncompressed_chunk_size_ - 1) /
        uncompressed_chunk_size_;
    status = cudaMemcpy(device_compressed_data, &current_batch_size,
                        sizeof(current_batch_size), cudaMemcpyHostToDevice);
    uint32_t *device_compressed_sizes = reinterpret_cast<uint32_t *>(
        device_compressed_data + sizeof(current_batch_size));
    *compressed_size = sizeof(current_batch_size) +
                       sizeof(*device_compressed_sizes) * current_batch_size;
    device_compressed_data += *compressed_size;
    while ((result = (status == cudaSuccess)) && current_batch_size) {
      min_batch =
          current_batch_size < batch_size_ ? current_batch_size : batch_size_;
      for (uint64_t i = 0; i < min_batch; ++i) {
        host_uncompressed_sizes_[i] = uncompressed_chunk_size_ / float_size_;
        host_uncompressed_ptrs_[i] =
            device_uncompressed_data + (i * uncompressed_chunk_size_);
      }
      if ((min_batch == current_batch_size) &&
          (uncompressed_size % uncompressed_chunk_size_)) {
        host_uncompressed_sizes_[min_batch - 1] =
            (uncompressed_size % uncompressed_chunk_size_);
        size_of_data_without_compress =
            host_uncompressed_sizes_[min_batch - 1] % float_size_;
        host_uncompressed_sizes_[min_batch - 1] /= float_size_;
        if (size_of_data_without_compress) {
          contains_data_without_compress = true;
          data_without_compress =
              reinterpret_cast<char *>(host_uncompressed_ptrs_[min_batch - 1]) +
              host_uncompressed_sizes_[min_batch - 1] * float_size_;
        }
      }
      floatCompress(*temp_memory_, configuration_float_, min_batch,
                    const_cast<const void **>(host_uncompressed_ptrs_),
                    host_uncompressed_sizes_, host_compressed_ptrs_,
                    device_compressed_sizes, stream_);
      status = cudaMemcpy(host_compressed_sizes_, device_compressed_sizes,
                          sizeof(*host_compressed_sizes_) * min_batch,
                          cudaMemcpyDeviceToHost);
      current_size = 0;
      for (uint64_t i = 0; status == cudaSuccess && i < min_batch; ++i) {
        status = cudaMemcpyAsync(
            device_compressed_data + current_size, host_compressed_ptrs_[i],
            host_compressed_sizes_[i], cudaMemcpyDeviceToDevice, stream_);
        current_size += host_compressed_sizes_[i];
      }
      *compressed_size += current_size;
      device_compressed_data += current_size;
      device_uncompressed_data += (min_batch * uncompressed_chunk_size_);
      device_compressed_sizes += min_batch;
      current_batch_size -= min_batch;
    }
    if (result && contains_data_without_compress) {
      status = cudaMemcpyAsync(device_compressed_data, data_without_compress,
                               size_of_data_without_compress,
                               cudaMemcpyDeviceToDevice, stream_);
      *compressed_size += size_of_data_without_compress;
    }
    if (status != cudaSuccess || !result) {
      std::cout << "ERROR: dietgpu-float error when compress data" << std::endl;
      result = false;
    }
  }
  result = (cudaStreamSynchronize(stream_) == cudaSuccess);
  return result;
}

bool DietgpuFloatLibrary::DecompressDeviceMemory(char *device_compressed_data,
                                                 uint64_t compressed_size,
                                                 char *device_decompressed_data,
                                                 uint64_t *decompressed_size) {
  bool result{initialized_decompressor_};
  if (result) {
    *decompressed_size = 0;
    cudaError_t status = cudaSuccess;
    uint32_t min_batch, current_size;
    uint32_t current_batch_size;
    status = cudaMemcpy(&current_batch_size, device_compressed_data,
                        sizeof(current_batch_size), cudaMemcpyDeviceToHost);
    uint32_t *device_compressed_sizes = reinterpret_cast<uint32_t *>(
        device_compressed_data + sizeof(current_batch_size));
    device_compressed_data +=
        sizeof(current_batch_size) +
        sizeof(*device_compressed_sizes) * current_batch_size;
    compressed_size -= (sizeof(current_batch_size) +
                        sizeof(*device_compressed_sizes) * current_batch_size);
    while ((result = (status == cudaSuccess)) && current_batch_size) {
      min_batch =
          current_batch_size < batch_size_ ? current_batch_size : batch_size_;
      status = cudaMemcpy(host_compressed_sizes_, device_compressed_sizes,
                          sizeof(*host_compressed_sizes_) * min_batch,
                          cudaMemcpyDeviceToHost);
      for (uint64_t i = 0; i < min_batch; ++i) {
        status = cudaMemcpyAsync(
            host_compressed_ptrs_[i], device_compressed_data,
            host_compressed_sizes_[i], cudaMemcpyDeviceToDevice, stream_);
        host_uncompressed_ptrs_[i] = device_decompressed_data;
        device_compressed_data += host_compressed_sizes_[i];
        compressed_size -= host_compressed_sizes_[i];
        device_decompressed_data += uncompressed_chunk_size_;
      }
      floatGetCompressedInfo(
          *temp_memory_, const_cast<const void **>(host_compressed_ptrs_),
          min_batch, device_uncompressed_sizes_, nullptr, nullptr, stream_);
      status =
          cudaMemcpyAsync(host_uncompressed_sizes_, device_uncompressed_sizes_,
                          sizeof(*host_uncompressed_sizes_) * min_batch,
                          cudaMemcpyDeviceToHost, stream_);
      if (status == cudaSuccess) {
        floatDecompress(*temp_memory_, configuration_float_, min_batch,
                        const_cast<const void **>(host_compressed_ptrs_),
                        host_uncompressed_ptrs_, host_uncompressed_sizes_,
                        nullptr, nullptr, stream_);
        status = cudaStreamSynchronize(stream_);
        current_size = 0;
        for (uint64_t i = 0; i < min_batch; ++i) {
          current_size += host_uncompressed_sizes_[i] * float_size_;
        }
        *decompressed_size += current_size;
        device_compressed_sizes += min_batch;
        current_batch_size -= min_batch;
      }
    }
    if (result && compressed_size) {
      device_decompressed_data -= uncompressed_chunk_size_;
      device_decompressed_data +=
          host_uncompressed_sizes_[min_batch - 1] * float_size_;
      status = cudaMemcpy(device_decompressed_data, device_compressed_data,
                          compressed_size, cudaMemcpyDeviceToDevice);
      *decompressed_size += compressed_size;
    }
    if (status != cudaSuccess || !result) {
      std::cout << "ERROR: dietgpu-float error when decompress data"
                << std::endl;
      result = false;
    }
  }
  return result;
}

void DietgpuFloatLibrary::GetTitle() {
  GpuCompressionLibrary::GetTitle(
      "dietgpu-float",
      "An extension to the above to handle fast lossless compression and "
      "decompression of unstructured floating point data, for use in ML and "
      "HPC applications, especially in communicating over local interconnects "
      "(PCIe / NVLink) and remote interconnects (Ethernet / InfiniBand).");
}

bool DietgpuFloatLibrary::GetChunkSizeInformation(
    std::vector<std::string> *chunk_size_information,
    uint8_t *minimum_chunk_size, uint8_t *maximum_chunk_size) {
  if (minimum_chunk_size) *minimum_chunk_size = 12;
  if (maximum_chunk_size) *maximum_chunk_size = 24;
  if (chunk_size_information) {
    chunk_size_information->clear();
    chunk_size_information->push_back("Available values [12-24]");
    chunk_size_information->push_back("[compression/decompression]");
  }
  return true;
}

bool DietgpuFloatLibrary::GetCompressionLevelInformation(
    std::vector<std::string> *compression_level_information,
    uint8_t *minimum_level, uint8_t *maximum_level) {
  if (minimum_level) *minimum_level = 0;
  if (maximum_level) *maximum_level = 2;
  if (compression_level_information) {
    compression_level_information->clear();
    compression_level_information->push_back("Available values [0-2]");
    compression_level_information->push_back("[compression/decompression]");
  }
  return true;
}

bool DietgpuFloatLibrary::GetFlagsInformation(
    std::vector<std::string> *flags_information, uint8_t *minimum_flags,
    uint8_t *maximum_flags) {
  if (minimum_flags) *minimum_flags = 0;
  if (maximum_flags) *maximum_flags = 2;
  if (flags_information) {
    flags_information->clear();
    flags_information->push_back("Available values [0-2]");
    flags_information->push_back("0: " + flags_[0]);
    flags_information->push_back("1: " + flags_[1]);
    flags_information->push_back("2: " + flags_[2]);
    flags_information->push_back("[compression/decompression]");
  }
  return true;
}

std::string DietgpuFloatLibrary::GetFlagsName(const uint8_t &flags) {
  std::string result = "ERROR";
  if (flags < number_of_flags_) {
    result = flags_[flags];
  }
  return result;
}

DietgpuFloatLibrary::DietgpuFloatLibrary(const uint64_t &batch_size)
    : batch_size_(batch_size), reserved_compressed_temporal_memory_(false) {
  host_uncompressed_ptrs_ = new void *[batch_size];
  cudaMalloc(&device_uncompressed_sizes_,
             sizeof(*device_uncompressed_sizes_) * batch_size);
  host_uncompressed_sizes_ = new uint32_t[batch_size];
  host_compressed_ptrs_ = new void *[batch_size];
  host_compressed_sizes_ = new uint32_t[batch_size];
  temp_memory_ = new dietgpu::StackDeviceMemory(dietgpu::getCurrentDevice(),
                                                256 * 1024 * 1024);
  number_of_flags_ = 3;
  flags_ = new std::string[number_of_flags_];
  flags_[0] = "Float16";
  flags_[1] = "BFloat16";
  flags_[2] = "Float32";
}

DietgpuFloatLibrary::~DietgpuFloatLibrary() {
  if (reserved_compressed_temporal_memory_) {
    cudaFree(host_compressed_ptrs_[0]);
  }
  cudaFree(device_uncompressed_sizes_);
  delete[] host_uncompressed_ptrs_;
  delete[] host_uncompressed_sizes_;
  delete[] host_compressed_ptrs_;
  delete[] host_compressed_sizes_;
  delete temp_memory_;
  delete[] flags_;
}
