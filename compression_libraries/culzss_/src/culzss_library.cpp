/*
 * rCUDA: remote CUDA (www.rCUDA.net)
 * Copyright (C) 2016-2022
 * Grupo de Arquitecturas Paralelas
 * Departamento de Informática de Sistemas y Computadores
 * Universidad Politécnica de Valencia (Spain)
 */

#include <cuda_runtime.h>
#include <gpu_compress.h>
#include <gpu_decompress.h>

// SMASH LIBRARIES
#include <culzss_library.hpp>
#include <gpu_options.hpp>

bool CulzssLibrary::CheckOptions(GpuOptions *options, const bool &compressor) {
  bool result =
      GpuCompressionLibrary::CheckChunkSize("culzss", options, 12, 24);
  if (result) {
    uncompressed_chunk_size_ = (1 << options->GetChunkSize());
  }
  return result;
}

void CulzssLibrary::GetCompressedDataSize(uint64_t uncompressed_size,
                                          uint64_t *compressed_size) {
  if (initialized_compressor_ || initialized_decompressor_) {
    *compressed_size = uncompressed_size * 2;
  } else {
    *compressed_size = 0;
  }
}

bool CulzssLibrary::CompressDeviceMemory(char *device_uncompressed_data,
                                         uint64_t uncompressed_size,
                                         char *device_compressed_data,
                                         uint64_t *compressed_size) {
  bool result{initialized_compressor_};
  if (result) {
    cudaError_t status = cudaSuccess;
    uint32_t min_batch, current_compressed_size, current_uncompressed_size;
    uint32_t current_batch_size =
        (uncompressed_size + uncompressed_chunk_size_ - 1) /
        uncompressed_chunk_size_;
    status = cudaMemcpy(device_compressed_data, &current_batch_size,
                        sizeof(current_batch_size), cudaMemcpyHostToDevice);
    uint32_t *device_compressed_sizes = reinterpret_cast<uint32_t *>(
        device_compressed_data + sizeof(current_batch_size));
    *compressed_size =
        sizeof(current_batch_size) +
        sizeof(*device_compressed_sizes) * (current_batch_size + 1);
    device_compressed_data += *compressed_size;
    while ((result = (status == cudaSuccess)) && current_batch_size) {
      if (current_batch_size < batch_size_) {
        min_batch = current_batch_size;
        current_uncompressed_size = uncompressed_size;
      } else {
        min_batch = batch_size_;
        current_uncompressed_size = batch_size_ * uncompressed_chunk_size_;
      }
      EncodeKernelSmash(min_batch, stream_, device_uncompressed_data,
                        current_uncompressed_size, device_compressed_data,
                        uncompressed_chunk_size_, device_compressed_sizes);
      status = cudaMemcpy(host_compressed_sizes_, device_compressed_sizes,
                          sizeof(*host_compressed_sizes_) * min_batch,
                          cudaMemcpyDeviceToHost);
      current_compressed_size = 0;
      if (host_compressed_sizes_[min_batch - 1] == 0) {
        host_compressed_sizes_[min_batch - 1] =
            uncompressed_size % uncompressed_chunk_size_;
      }
      for (uint64_t i = 0; status == cudaSuccess && i < min_batch; ++i) {
        status = cudaMemcpyAsync(
            device_compressed_data + current_compressed_size,
            device_compressed_data + (i * uncompressed_chunk_size_ * 2),
            host_compressed_sizes_[i], cudaMemcpyDeviceToDevice, stream_);
        current_compressed_size += host_compressed_sizes_[i];
      }
      *compressed_size += current_compressed_size;
      device_compressed_data += current_compressed_size;
      device_uncompressed_data += current_uncompressed_size;
      uncompressed_size -= current_uncompressed_size;
      device_compressed_sizes += min_batch;
      current_batch_size -= min_batch;
    }

    if (status != cudaSuccess || !result) {
      std::cout << "ERROR: culzss error when compress data" << std::endl;
      result = false;
    }
  }
  result = (cudaStreamSynchronize(stream_) == cudaSuccess);
  return result;
}

bool CulzssLibrary::DecompressDeviceMemory(char *device_compressed_data,
                                           uint64_t compressed_size,
                                           char *device_decompressed_data,
                                           uint64_t *decompressed_size) {
  bool result{initialized_decompressor_};
  if (result) {
    *decompressed_size = 0;
    cudaError_t status = cudaSuccess;
    uint32_t min_batch, current_compressed_size, current_uncompressed_size;
    uint32_t current_batch_size;
    status = cudaMemcpy(&current_batch_size, device_compressed_data,
                        sizeof(current_batch_size), cudaMemcpyDeviceToHost);
    uint32_t *device_compressed_sizes = reinterpret_cast<uint32_t *>(
        device_compressed_data + sizeof(current_batch_size));
    device_compressed_data +=
        sizeof(current_batch_size) +
        sizeof(*device_compressed_sizes) * (current_batch_size + 1);
    while ((result = (status == cudaSuccess)) && current_batch_size) {
      min_batch =
          current_batch_size < batch_size_ ? current_batch_size : batch_size_;
      DecodeKernelSmash(min_batch, stream_, device_compressed_data,
                        device_decompressed_data, device_compressed_sizes,
                        device_uncompressed_sizes_, uncompressed_chunk_size_);
      status = cudaMemcpyAsync(host_compressed_sizes_, device_compressed_sizes,
                               sizeof(*host_compressed_sizes_) * min_batch,
                               cudaMemcpyDeviceToHost, stream_);
      if (status == cudaSuccess) {
        status =
            cudaMemcpy(host_uncompressed_sizes_, device_uncompressed_sizes_,
                       sizeof(*host_uncompressed_sizes_) * min_batch,
                       cudaMemcpyDeviceToHost);
        if (status == cudaSuccess) {
          current_uncompressed_size = 0;
          current_compressed_size = 0;
          for (uint64_t i = 0; i < min_batch; ++i) {
            current_uncompressed_size += host_uncompressed_sizes_[i];
            current_compressed_size += host_compressed_sizes_[i];
          }
          *decompressed_size += current_uncompressed_size;
          device_compressed_data += current_compressed_size;
          device_decompressed_data += current_uncompressed_size;
          device_compressed_sizes += min_batch;
          current_batch_size -= min_batch;
        }
      }
    }

    if (status != cudaSuccess || !result) {
      std::cout << "ERROR: culzss error when decompress data" << std::endl;
      result = false;
    }
  }
  return result;
}

void CulzssLibrary::GetTitle() {
  GpuCompressionLibrary::GetTitle("culzss",
                                  "A GPU-based LZSS compression algorithm, "
                                  "highly tuned for NVIDIA GPGPUs "
                                  "and for streaming data, leveraging the "
                                  "respective strengths of CPUs and "
                                  "GPUs together.");
}

bool CulzssLibrary::GetChunkSizeInformation(
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

CulzssLibrary::CulzssLibrary(const uint64_t &batch_size)
    : batch_size_(batch_size) {
  cudaMalloc(&device_uncompressed_sizes_,
             sizeof(*device_uncompressed_sizes_) * batch_size);
  host_uncompressed_sizes_ = new uint32_t[batch_size];
  host_compressed_sizes_ = new uint32_t[batch_size];
}

CulzssLibrary::~CulzssLibrary() {
  cudaFree(device_uncompressed_sizes_);
  delete[] host_uncompressed_sizes_;
  delete[] host_compressed_sizes_;
}
