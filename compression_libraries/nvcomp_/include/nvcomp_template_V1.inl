/*
 * rCUDA: remote CUDA (www.rCUDA.net)
 * Copyright (C) 2016-2022
 * Grupo de Arquitecturas Paralelas
 * Departamento de Informática de Sistemas y Computadores
 * Universidad Politécnica de Valencia (Spain)
 */
#include <unistd.h>

template <typename Opts_t>
inline void NvcompTemplate<Opts_t>::RemoveTemporalMemory() {
  if (device_temporal_memory_) {
    cudaFree(device_temporal_memory_);
    device_temporal_memory_ = nullptr;
  }
}

template <typename Opts_t>
inline bool NvcompTemplate<Opts_t>::InitializeCompression(
    const size_t &chunk_size, const Opts_t &configuration,
    const cudaStream_t &stream) {
  size_t max_chunk_size = 0;
  stream_ = stream;
  options_ = configuration;
  RemoveTemporalMemory();
  get_temporal_size_to_compress_(batch_size_, chunk_size, options_,
                                 &temporal_memory_size_);
  get_max_compressed_chunk_size_(chunk_size, options_, &max_chunk_size);
  compressed_data_->ConfigureCompression(max_chunk_size);
  cudaError_t error =
      cudaMalloc(&device_temporal_memory_, temporal_memory_size_);
  chunk_size_ = chunk_size;
  max_chunk_size_ = max_chunk_size;
  return error == cudaSuccess;
}

template <typename Opts_t>
inline bool NvcompTemplate<Opts_t>::InitializeDecompression(
    const size_t &chunk_size, const cudaStream_t &stream) {
  stream_ = stream;
  RemoveTemporalMemory();
  get_temporal_size_to_decompress_(batch_size_, chunk_size,
                                   &temporal_memory_size_);
  cudaError_t error =
      cudaMalloc(&device_temporal_memory_, temporal_memory_size_);
  chunk_size_ = chunk_size;
  return (error == cudaSuccess);
}

template <typename Opts_t>
inline void NvcompTemplate<Opts_t>::GetCompressedDataSize(
    const uint64_t &uncompressed_data_size, uint64_t *compressed_data_size) {
  *compressed_data_size =
      max_chunk_size_ *
      ((uncompressed_data_size + chunk_size_ - 1) / chunk_size_);
}

template <typename Opts_t>
inline void NvcompTemplate<Opts_t>::GetHeader(
    size_t *current_batch_size, size_t **device_compressed_displacements,
    char **device_compressed_data) {
  cudaMemcpy(current_batch_size, *device_compressed_data,
             sizeof(*current_batch_size), cudaMemcpyDeviceToHost);
  *device_compressed_displacements = reinterpret_cast<size_t *>(
      *device_compressed_data + sizeof(*current_batch_size));
  *device_compressed_data +=
      sizeof(*current_batch_size) +
      sizeof(**device_compressed_displacements) * *current_batch_size;
}

template <typename Opts_t>
inline void NvcompTemplate<Opts_t>::GetHeaderHost(
    size_t *current_batch_size, size_t **host_compressed_displacements,
    char **host_compressed_data) {
  *current_batch_size = *reinterpret_cast<size_t *>(*host_compressed_data);
  *host_compressed_displacements = reinterpret_cast<size_t *>(
      *host_compressed_data + sizeof(*current_batch_size));
  *host_compressed_data +=
      sizeof(*current_batch_size) +
      sizeof(**host_compressed_displacements) * *current_batch_size;
}

template <typename Opts_t>
inline void NvcompTemplate<Opts_t>::GetDecompressedDataSize(
    const char *const device_compressed_data,
    uint64_t *decompressed_data_size) {
  nvcompStatus_t status = nvcompSuccess;
  *decompressed_data_size = 0;
  size_t min_batch;
  size_t current_batch_size;
  size_t *device_compressed_displacements;
  char *compressed_data = const_cast<char *>(device_compressed_data);
  GetHeader(&current_batch_size, &device_compressed_displacements,
            &compressed_data);
  uncompressed_data_->InitializeDecompression(nullptr, stream_);
  compressed_data_->InitializeDecompression(
      compressed_data, device_compressed_displacements, stream_);
  while (status == nvcompSuccess && current_batch_size) {
    min_batch =
        (batch_size_ < current_batch_size) ? batch_size_ : current_batch_size;
    compressed_data_->GetNext(min_batch, stream_);
    status = get_decompressed_size_asynchronously_(
        compressed_data_->d_ptrs(), compressed_data_->d_sizes(),
        uncompressed_data_->d_sizes(), min_batch, stream_);
    if (status == nvcompSuccess) {
      uncompressed_data_->GetNextDecompression(min_batch, chunk_size_, stream_);
      current_batch_size -= min_batch;
    }
  }
  *decompressed_data_size = uncompressed_data_->size();
}

template <typename Opts_t>
inline void NvcompTemplate<Opts_t>::SetHeader(
    size_t *current_batch_size, const uint64_t &uncompressed_data_size,
    size_t **device_compressed_displacements, char **device_compressed_data,
    uint64_t *compressed_data_size) {
  *current_batch_size =
      (uncompressed_data_size + chunk_size_ - 1) / chunk_size_;
  cudaMemcpy(*device_compressed_data, current_batch_size,
             sizeof(*current_batch_size), cudaMemcpyHostToDevice);
  *device_compressed_displacements = reinterpret_cast<size_t *>(
      *device_compressed_data + sizeof(*current_batch_size));
  *compressed_data_size =
      sizeof(*current_batch_size) +
      sizeof(*(*device_compressed_displacements)) * *current_batch_size;
  *device_compressed_data += *compressed_data_size;
}

template <typename Opts_t>
inline void NvcompTemplate<Opts_t>::SetHeaderHost(
    size_t *current_batch_size, const uint64_t &uncompressed_data_size,
    size_t **host_compressed_displacements, char **host_compressed_data,
    uint64_t *compressed_data_size) {
  *current_batch_size =
      (uncompressed_data_size + chunk_size_ - 1) / chunk_size_;
  *reinterpret_cast<size_t *>(*host_compressed_data) = *current_batch_size;
  *host_compressed_displacements = reinterpret_cast<size_t *>(
      *host_compressed_data + sizeof(*current_batch_size));
  *compressed_data_size =
      sizeof(*current_batch_size) +
      sizeof(*(*host_compressed_displacements)) * *current_batch_size;
  *host_compressed_data += *compressed_data_size;
}

template <typename Opts_t>
inline bool NvcompTemplate<Opts_t>::CompressDeviceToDevice(
    const char *const device_uncompressed_data,
    const uint64_t &uncompressed_data_size, char *device_compressed_data,
    uint64_t *compressed_data_size) {
  size_t current_batch_size;
  size_t *device_compressed_displacements;
  SetHeader(&current_batch_size, uncompressed_data_size,
            &device_compressed_displacements, &device_compressed_data,
            compressed_data_size);
  bool result{true};
  size_t current_slices;
  uncompressed_data_->InitializeCompression(device_uncompressed_data,
                                            uncompressed_data_size);
  compressed_data_->InitializeCompression(
      device_compressed_data, device_compressed_displacements, stream_);
  while (result && current_batch_size) {
    current_slices =
        (batch_size_ < current_batch_size) ? batch_size_ : current_batch_size;
    uncompressed_data_->GetNextCompression(current_slices, chunk_size_,
                                           stream_);
    result = (nvcompSuccess ==
              compress_asynchronously_(
                  uncompressed_data_->d_ptrs(), uncompressed_data_->d_sizes(),
                  chunk_size_, current_slices, device_temporal_memory_,
                  temporal_memory_size_, compressed_data_->d_ptrs(),
                  compressed_data_->d_sizes(), options_, stream_));
    if (result) {
      compressed_data_->DumpData(current_slices, stream_);
      current_batch_size -= current_slices;
    }
  }
  if (result) {
    compressed_data_->GetSize(compressed_data_size);
  } else {
    *compressed_data_size = 0;
  }
  return result;
}

template <typename Opts_t>
inline bool NvcompTemplate<Opts_t>::CompressHostToDevice(
    const char *const host_uncompressed_data,
    const uint64_t &uncompressed_data_size, char *device_compressed_data,
    uint64_t *compressed_data_size) {
  char *device_uncompressed_data;
  bool result = (cudaSuccess ==
                 cudaMalloc(&device_uncompressed_data, uncompressed_data_size));
  if (result) {
    result = (cudaSuccess == cudaMemcpyAsync(device_uncompressed_data,
                                             host_uncompressed_data,
                                             uncompressed_data_size,
                                             cudaMemcpyHostToDevice, stream_));
    if (result) {
      result = CompressDeviceToDevice(
          device_uncompressed_data, uncompressed_data_size,
          device_compressed_data, compressed_data_size);
    }
    result &= (cudaSuccess == cudaFree(device_uncompressed_data));
  }
  return result;
}

template <typename Opts_t>
inline bool NvcompTemplate<Opts_t>::CompressDeviceToHost(
    const char *const device_uncompressed_data,
    const uint64_t &uncompressed_data_size, char *host_compressed_data,
    uint64_t *compressed_data_size) {
  char *device_compressed_data;
  GetCompressedDataSize(uncompressed_data_size, compressed_data_size);
  bool result = (cudaSuccess ==
                 cudaMalloc(&device_compressed_data, *compressed_data_size));
  if (result) {
    result =
        CompressDeviceToDevice(device_uncompressed_data, uncompressed_data_size,
                               device_compressed_data, compressed_data_size);
    if (result) {
      result = (cudaSuccess ==
                cudaMemcpy(host_compressed_data, device_compressed_data,
                           *compressed_data_size, cudaMemcpyDeviceToHost));
    }
    result &= (cudaSuccess == cudaFree(device_compressed_data));
  }
  return result;
}

template <typename Opts_t>
inline bool NvcompTemplate<Opts_t>::CompressHostToHost(
    const char *const host_uncompressed_data,
    const uint64_t &uncompressed_data_size, char *host_compressed_data,
    uint64_t *compressed_data_size) {
  char *device_uncompressed_data, *device_compressed_data;
  GetCompressedDataSize(uncompressed_data_size, compressed_data_size);
  bool result = (cudaSuccess ==
                 cudaMalloc(&device_compressed_data, *compressed_data_size));
  if (result) {
    result = (cudaSuccess ==
              cudaMalloc(&device_uncompressed_data, uncompressed_data_size));
    if (result) {
      result = (cudaSuccess ==
                cudaMemcpyAsync(device_uncompressed_data,
                                host_uncompressed_data, uncompressed_data_size,
                                cudaMemcpyHostToDevice, stream_));
      if (result) {
        result = CompressDeviceToDevice(
            device_uncompressed_data, uncompressed_data_size,
            device_compressed_data, compressed_data_size);
        if (result) {
          result = (cudaSuccess ==
                    cudaMemcpy(host_compressed_data, device_compressed_data,
                               *compressed_data_size, cudaMemcpyDeviceToHost));
        }
      }
      result &= (cudaSuccess == cudaFree(device_uncompressed_data));
    }
    result &= (cudaSuccess == cudaFree(device_compressed_data));
  }
  return result;
}

template <typename Opts_t>
inline bool NvcompTemplate<Opts_t>::DecompressDeviceToDevice(
    const char *const device_compressed_data,
    const uint64_t &compressed_data_size, char *device_decompressed_data,
    uint64_t *decompressed_data_size) {
  size_t current_batch_size;
  size_t *device_compressed_displacements;
  char *compressed_data = const_cast<char *>(device_compressed_data);
  GetHeader(&current_batch_size, &device_compressed_displacements,
            &compressed_data);
  bool result{true};
  size_t current_slices;
  uncompressed_data_->InitializeDecompression(device_decompressed_data,
                                              stream_);
  compressed_data_->InitializeDecompression(
      compressed_data, device_compressed_displacements, stream_);
  while (result && current_batch_size) {
    current_slices =
        (batch_size_ < current_batch_size) ? batch_size_ : current_batch_size;
    compressed_data_->GetNext(current_slices, stream_);
    uncompressed_data_->GetNextDecompression(current_slices, chunk_size_,
                                             stream_);
    result =
        (nvcompSuccess ==
         decompress_asynchronously_(
             compressed_data_->d_ptrs(), compressed_data_->d_sizes(),
             uncompressed_data_->d_sizes(), uncompressed_data_->d_sizes(),
             current_slices, device_temporal_memory_, temporal_memory_size_,
             uncompressed_data_->d_ptrs(), statuses_, stream_));
    current_batch_size -= current_slices;
  }
  if (result) {
    *decompressed_data_size = uncompressed_data_->size();
  } else {
    *decompressed_data_size = 0;
  }
  return result;
}

template <typename Opts_t>
inline bool NvcompTemplate<Opts_t>::DecompressDeviceToHost(
    const char *const device_compressed_data,
    const uint64_t &compressed_data_size, char *host_decompressed_data,
    uint64_t *decompressed_data_size) {
  char *device_decompressed_data;
  GetDecompressedDataSize(device_compressed_data, decompressed_data_size);
  bool result = (cudaSuccess == cudaMalloc(&device_decompressed_data,
                                           *decompressed_data_size));
  if (result) {
    result = DecompressDeviceToDevice(
        device_compressed_data, compressed_data_size, device_decompressed_data,
        decompressed_data_size);
    if (result) {
      result = (cudaSuccess ==
                cudaMemcpyAsync(
                    host_decompressed_data, device_decompressed_data,
                    *decompressed_data_size, cudaMemcpyDeviceToHost, stream_));
    }
    result &= (cudaSuccess == cudaFree(device_decompressed_data));
  }
  return result;
}

template <typename Opts_t>
inline bool NvcompTemplate<Opts_t>::DecompressHostToDevice(
    const char *const host_compressed_data,
    const uint64_t &compressed_data_size, char *device_decompressed_data,
    uint64_t *decompressed_data_size) {
  char *device_compressed_data;
  bool result = (cudaSuccess ==
                 cudaMalloc(&device_compressed_data, compressed_data_size));
  if (result) {
    result = (cudaSuccess == cudaMemcpyAsync(device_compressed_data,
                                             host_compressed_data,
                                             compressed_data_size,
                                             cudaMemcpyHostToDevice, stream_));
    if (result) {
      result = DecompressDeviceToDevice(
          device_compressed_data, compressed_data_size,
          device_decompressed_data, decompressed_data_size);
    }
    result &= (cudaSuccess == cudaFree(device_compressed_data));
  }
  return result;
}

template <typename Opts_t>
inline bool NvcompTemplate<Opts_t>::DecompressHostToHost(
    const char *const host_compressed_data,
    const uint64_t &compressed_data_size, char *host_decompressed_data,
    uint64_t *decompressed_data_size) {
  char *device_decompressed_data, *device_compressed_data;
  bool result = (cudaSuccess ==
                 cudaMalloc(&device_compressed_data, compressed_data_size));
  if (result) {
    result = (cudaSuccess == cudaMemcpyAsync(device_compressed_data,
                                             host_compressed_data,
                                             compressed_data_size,
                                             cudaMemcpyHostToDevice, stream_));
    if (result) {
      GetDecompressedDataSize(device_compressed_data, decompressed_data_size);
      result = (cudaSuccess ==
                cudaMalloc(&device_decompressed_data, *decompressed_data_size));
      if (result) {
        result = DecompressDeviceToDevice(
            device_compressed_data, compressed_data_size,
            device_decompressed_data, decompressed_data_size);
        if (result) {
          result =
              (cudaSuccess == cudaMemcpyAsync(host_decompressed_data,
                                              device_decompressed_data,
                                              *decompressed_data_size,
                                              cudaMemcpyDeviceToHost, stream_));
        }
      }
      result &= (cudaSuccess == cudaFree(device_compressed_data));
    }
    result &= (cudaSuccess == cudaFree(device_decompressed_data));
  }
  return result;
}

template <typename Opts_t>
bool NvcompTemplate<Opts_t>::CreateInternalStructures(const size_t &batch_size,
                                                      const uint8_t &streams) {
  bool result{true};
  if (batch_size_ != batch_size) {
    DestroyInternalStructures();
    batch_size_ = batch_size;
    compressed_data_ = new BatchDataCompressed(batch_size_);
    uncompressed_data_ = new BatchDataUncompressed(batch_size_);
    result = (cudaSuccess ==
              cudaMalloc(&statuses_, sizeof(*statuses_) * batch_size_));
  }
  return result;
}

template <typename Opts_t>
void NvcompTemplate<Opts_t>::DestroyInternalStructures() {
  if (compressed_data_) {
    delete compressed_data_;
    delete uncompressed_data_;
    cudaFree(statuses_);
  }
  compressed_data_ = nullptr;
}

template <typename Opts_t>
NvcompTemplate<Opts_t>::NvcompTemplate(
    nvcompStatus_t (*get_temporal_size_to_compress)(size_t batch_size,
                                                    size_t max_chunk_size,
                                                    Opts_t format_ops,
                                                    size_t *temp_bytes),
    nvcompStatus_t (*get_max_compressed_chunk_size)(
        size_t max_chunk_size, Opts_t format_opts, size_t *max_compressed_size),
    nvcompStatus_t (*get_temporal_size_to_decompress)(
        size_t num_chunks, size_t max_uncompressed_chunk_size,
        size_t *temp_bytes),
    nvcompStatus_t (*get_decompressed_size_asynchronously)(
        const void *const *device_compressed_ptrs,
        const size_t *device_compressed_bytes,
        size_t *device_uncompressed_bytes, size_t batch_size,
        cudaStream_t stream),
    nvcompStatus_t (*compress_asynchronously)(
        const void *const *device_uncompressed_ptr,
        const size_t *device_uncompressed_bytes,
        size_t max_uncompressed_chunk_bytes, size_t batch_size,
        void *device_temp_ptr, size_t temp_bytes,
        void *const *device_compressed_ptr, size_t *device_compressed_bytes,
        Opts_t format_ops, cudaStream_t stream),
    nvcompStatus_t (*decompress_asynchronously)(
        const void *const *device_compresed_ptrs,
        const size_t *device_compressed_bytes,
        const size_t *device_uncompressed_bytes,
        size_t *device_actual_uncompressed_bytes, size_t batch_size,
        void *const device_temp_ptr, const size_t temp_bytes,
        void *const *device_uncompressed_ptr, nvcompStatus_t *device_statuses,
        cudaStream_t stream))
    : get_temporal_size_to_compress_(get_temporal_size_to_compress),
      get_max_compressed_chunk_size_(get_max_compressed_chunk_size),
      get_temporal_size_to_decompress_(get_temporal_size_to_decompress),
      get_decompressed_size_asynchronously_(
          get_decompressed_size_asynchronously),
      compress_asynchronously_(compress_asynchronously),
      decompress_asynchronously_(decompress_asynchronously),
      stream_(nullptr),
      statuses_(nullptr),
      device_temporal_memory_(nullptr),
      compressed_data_(nullptr),
      uncompressed_data_(nullptr),
      chunk_size_(0),
      batch_size_(0),
      max_chunk_size_(0),
      temporal_memory_size_(0) {}

template <typename Opts_t>
NvcompTemplate<Opts_t>::~NvcompTemplate() {
  DestroyInternalStructures();
  RemoveTemporalMemory();
}
