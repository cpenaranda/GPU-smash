/*
 * rCUDA: remote CUDA (www.rCUDA.net)
 * Copyright (C) 2016-2022
 * Grupo de Arquitecturas Paralelas
 * Departamento de Informática de Sistemas y Computadores
 * Universidad Politécnica de Valencia (Spain)
 */

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
  stream_ = stream;
  options_ = configuration;
  chunk_size_ = chunk_size;
  RemoveTemporalMemory();
  get_temporal_size_to_compress_(batch_size_, chunk_size_, options_,
                                 &temporal_memory_size_);
  get_max_compressed_chunk_size_(chunk_size_, options_, &max_chunk_size_);
  cudaMalloc(&device_temporal_memory_, temporal_memory_size_);
  return true;
}

template <typename Opts_t>
inline bool NvcompTemplate<Opts_t>::InitializeDecompression(
    const size_t &chunk_size, const cudaStream_t &stream) {
  stream_ = stream;
  chunk_size_ = chunk_size;
  RemoveTemporalMemory();
  get_temporal_size_to_decompress_(batch_size_, chunk_size_,
                                   &temporal_memory_size_);
  cudaMalloc(&device_temporal_memory_, temporal_memory_size_);
  return true;
}

template <typename Opts_t>
inline void NvcompTemplate<Opts_t>::GetCompressedDataSize(
    uint64_t uncompressed_size, uint64_t *compressed_size) {
  *compressed_size =
      max_chunk_size_ * ((uncompressed_size + chunk_size_ - 1) / chunk_size_);
}

template <typename Opts_t>
inline void NvcompTemplate<Opts_t>::GetBatchDataInformationFromCompressedData(
    size_t *current_batch_size, size_t **device_compressed_sizes,
    char *device_compressed_data) {
  cudaMemcpy(current_batch_size, device_compressed_data,
             sizeof(*current_batch_size), cudaMemcpyDeviceToHost);
  *device_compressed_sizes = reinterpret_cast<size_t *>(
      device_compressed_data + sizeof(*current_batch_size));
}

template <typename Opts_t>
inline void NvcompTemplate<Opts_t>::GetDecompressedDataSize(
    char *device_compressed_data, uint64_t *decompressed_size) {
  *decompressed_size = 0;
  size_t min_batch;
  size_t current_batch_size;
  size_t *device_compressed_sizes;
  nvcompStatus_t status = nvcompSuccess;
  GetBatchDataInformationFromCompressedData(
      &current_batch_size, &device_compressed_sizes, device_compressed_data);
  device_compressed_data +=
      sizeof(current_batch_size) +
      sizeof(*device_compressed_sizes) * current_batch_size;
  uncompressed_data_->InitilizeDecompression(nullptr);
  compressed_data_->InitilizeDecompression(device_compressed_data, 0,
                                           device_compressed_sizes);
  while (status == nvcompSuccess && current_batch_size) {
    min_batch =
        (batch_size_ < current_batch_size) ? batch_size_ : current_batch_size;
    compressed_data_->GetNext(min_batch, stream_);
    status = get_decompressed_size_asynchronously_(
        compressed_data_->d_ptrs(), compressed_data_->d_sizes(),
        uncompressed_data_->d_sizes(), min_batch, stream_);
    if (status == nvcompSuccess) {
      uncompressed_data_->GetNext(min_batch, chunk_size_, stream_,
                                  decompressed_size);
    }
  }
}

template <typename Opts_t>
inline void NvcompTemplate<Opts_t>::GetBatchDataInformationFromUncompressedData(
    size_t *current_batch_size, uint64_t uncompressed_size,
    size_t **device_compressed_sizes, char *device_compressed_data,
    uint64_t *compresssed_size) {
  *current_batch_size = (uncompressed_size + chunk_size_ - 1) / chunk_size_;
  cudaMemcpyAsync(device_compressed_data, current_batch_size,
                  sizeof(*current_batch_size), cudaMemcpyHostToDevice, stream_);
  *device_compressed_sizes = reinterpret_cast<size_t *>(
      device_compressed_data + sizeof(*current_batch_size));
  *compresssed_size = sizeof(*current_batch_size) +
                      sizeof(*(*device_compressed_sizes)) * *current_batch_size;
}

template <typename Opts_t>
inline bool NvcompTemplate<Opts_t>::Compress(char *device_uncompressed_data,
                                             uint64_t uncompressed_size,
                                             char *device_compressed_data,
                                             uint64_t *compressed_size) {
  size_t min_batch;
  nvcompStatus_t status = nvcompSuccess;
  size_t current_batch_size;
  size_t *device_compressed_sizes;
  GetBatchDataInformationFromUncompressedData(
      &current_batch_size, uncompressed_size, &device_compressed_sizes,
      device_compressed_data, compressed_size);
  device_compressed_data +=
      sizeof(current_batch_size) +
      sizeof(*device_compressed_sizes) * current_batch_size;
  uncompressed_data_->InitilizeCompression(device_uncompressed_data,
                                           uncompressed_size);
  compressed_data_->InitilizeCompression(device_compressed_data,
                                         device_compressed_sizes,
                                         max_chunk_size_, stream_);
  while ((status == nvcompSuccess) && current_batch_size) {
    min_batch =
        (batch_size_ < current_batch_size) ? batch_size_ : current_batch_size;
    uncompressed_data_->GetNext(min_batch, chunk_size_, stream_);
    status = compress_asynchronously_(
        uncompressed_data_->d_ptrs(), uncompressed_data_->d_sizes(),
        chunk_size_, min_batch, device_temporal_memory_, temporal_memory_size_,
        compressed_data_->d_ptrs(), compressed_data_->d_sizes(), options_,
        stream_);
    if (status == nvcompSuccess) {
      compressed_data_->DumpData(min_batch, stream_);
      current_batch_size -= min_batch;
    }
  }
  *compressed_size = compressed_data_->size();
  cudaStreamSynchronize(stream_);
  return status == nvcompSuccess;
}

template <typename Opts_t>
inline bool NvcompTemplate<Opts_t>::Decompress(char *device_compressed_data,
                                               uint64_t compressed_size,
                                               char *device_decompressed_data,
                                               uint64_t *decompressed_size) {
  nvcompStatus_t status = nvcompSuccess;
  *decompressed_size = 0;
  size_t min_batch;
  size_t current_batch_size;
  size_t *device_compressed_sizes;
  GetBatchDataInformationFromCompressedData(
      &current_batch_size, &device_compressed_sizes, device_compressed_data);
  device_compressed_data +=
      sizeof(current_batch_size) +
      sizeof(*device_compressed_sizes) * current_batch_size;
  uncompressed_data_->InitilizeDecompression(device_decompressed_data);
  compressed_data_->InitilizeDecompression(
      device_compressed_data, compressed_size, device_compressed_sizes);
  while ((status == nvcompSuccess) && current_batch_size) {
    min_batch =
        (batch_size_ < current_batch_size) ? batch_size_ : current_batch_size;
    compressed_data_->GetNext(min_batch, stream_);
    status = get_decompressed_size_asynchronously_(
        compressed_data_->d_ptrs(), compressed_data_->d_sizes(),
        uncompressed_data_->d_sizes(), min_batch, stream_);
    if (status == nvcompSuccess) {
      uncompressed_data_->GetNext(min_batch, chunk_size_, stream_,
                                  decompressed_size);
      status = decompress_asynchronously_(
          compressed_data_->d_ptrs(), compressed_data_->d_sizes(),
          uncompressed_data_->d_sizes(), uncompressed_data_->d_sizes(),
          min_batch, device_temporal_memory_, temporal_memory_size_,
          uncompressed_data_->d_ptrs(), statuses, stream_);
      if (status == nvcompSuccess) {
        compressed_data_->IncrementSizes(min_batch);
        current_batch_size -= min_batch;
      }
    }
  }
  cudaStreamSynchronize(stream_);
  return status == nvcompSuccess;
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
        cudaStream_t stream),
    const size_t &batch_size)
    : get_temporal_size_to_compress_(get_temporal_size_to_compress),
      get_max_compressed_chunk_size_(get_max_compressed_chunk_size),
      get_temporal_size_to_decompress_(get_temporal_size_to_decompress),
      get_decompressed_size_asynchronously_(
          get_decompressed_size_asynchronously),
      compress_asynchronously_(compress_asynchronously),
      decompress_asynchronously_(decompress_asynchronously),
      batch_size_(batch_size),
      device_temporal_memory_(nullptr) {
  compressed_data_ = new BatchDataCompressed(batch_size_);
  uncompressed_data_ = new BatchDataUncompressed(batch_size_);
  cudaMalloc(&statuses, sizeof(*statuses) * batch_size_);
}

template <typename Opts_t>
NvcompTemplate<Opts_t>::~NvcompTemplate() {
  cudaFree(statuses);
  delete compressed_data_;
  delete uncompressed_data_;
  RemoveTemporalMemory();
}
