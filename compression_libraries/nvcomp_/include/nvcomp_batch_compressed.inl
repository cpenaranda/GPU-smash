/*
 * rCUDA: remote CUDA (www.rCUDA.net)
 * Copyright (C) 2016-2022
 * Grupo de Arquitecturas Paralelas
 * Departamento de Informática de Sistemas y Computadores
 * Universidad Politécnica de Valencia (Spain)
 */

inline cudaError_t BatchDataCompressed::InitializeCompression(
    char *data, size_t *displacements, const cudaStream_t &stream) {
  data_ = data;
  cudaError_t error = cudaMemsetAsync(d_size_, 0, sizeof(*d_size_), stream);
  if (error == cudaSuccess) {
    error = cudaMemcpyAsync(d_ptrs_, h_ptrs_compression_,
                            sizeof(*h_ptrs_compression_) * slices_,
                            cudaMemcpyHostToDevice, stream);
  }
  d_displacements_ = displacements;
  last_chunk_ = 0;
  return error;
}

inline cudaError_t BatchDataCompressed::ConfigureCompression(
    const size_t &max_chunk_size) {
  cudaError_t error{cudaSuccess};
  if (max_chunk_size_ < max_chunk_size) {
    if (max_chunk_size_ != 0) {
      for (uint64_t i_chunk = 0; error == cudaSuccess && i_chunk < slices_;
           ++i_chunk) {
        error = cudaFree(h_ptrs_compression_[i_chunk]);
      }
    }
    max_chunk_size_ = max_chunk_size;
    for (uint64_t i_chunk = 0; error == cudaSuccess && i_chunk < slices_;
         ++i_chunk) {
      error = cudaMalloc(&h_ptrs_compression_[i_chunk], max_chunk_size);
    }
  }
  return error;
}

inline cudaError_t BatchDataCompressed::DumpData(const size_t &chunks,
                                                 const cudaStream_t &stream) {
  NvcompUtil::DumpData(data_, d_ptrs_, d_size_, d_sizes_,
                       d_displacements_ + last_chunk_, chunks, stream);
  last_chunk_ += chunks;
  return cudaSuccess;
}

inline cudaError_t BatchDataCompressed::DumpDataPipeline(
    size_t *device_last_batch_size, const size_t &chunks,
    const cudaStream_t &stream) {
  NvcompUtil::DumpDataPipeline(data_, d_ptrs_, device_last_batch_size, d_sizes_,
                               d_displacements_, chunks, stream);
  return cudaMemcpyAsync(device_last_batch_size, &d_displacements_[chunks - 1],
                         sizeof(*device_last_batch_size),
                         cudaMemcpyDeviceToDevice, stream);
}

inline void BatchDataCompressed::DumpDataPipeline2(const size_t &chunks,
                                                   const cudaStream_t &stream) {
  NvcompUtil::DumpDataPipeline2(data_, d_ptrs_, d_sizes_, d_displacements_,
                                chunks, stream);
}

inline cudaError_t BatchDataCompressed::IncrementPipeline(
    size_t *device_last_batch_size, const size_t &chunks,
    const cudaStream_t &stream) {
  NvcompUtil::IncrementPipeline(device_last_batch_size, d_displacements_,
                                chunks, stream);
  return cudaMemcpyAsync(device_last_batch_size, &d_displacements_[chunks - 1],
                         sizeof(*device_last_batch_size),
                         cudaMemcpyDeviceToDevice, stream);
}

inline cudaError_t BatchDataCompressed::IncrementPipeline(
    const size_t &last_batch_size, const size_t &chunks,
    const cudaStream_t &stream) {
  NvcompUtil::IncrementPipeline(last_batch_size, d_displacements_, chunks,
                                stream);
  return cudaSuccess;
}

inline cudaError_t BatchDataCompressed::InitializeDecompression(
    const char *const data, size_t *displacements, const cudaStream_t &stream) {
  data_ = const_cast<char *>(data);
  d_displacements_ = displacements;
  last_chunk_ = 0;
  return cudaMemsetAsync(d_size_, 0, sizeof(*d_size_), stream);
}

inline cudaError_t BatchDataCompressed::GetNext(const size_t &chunks,
                                                const cudaStream_t &stream) {
  NvcompUtil::GetNext(data_, d_ptrs_, d_size_, d_sizes_,
                      d_displacements_ + last_chunk_, chunks, stream);
  last_chunk_ += chunks;
  return cudaMemcpyAsync(d_size_, &d_displacements_[last_chunk_ - 1],
                         sizeof(*d_size_), cudaMemcpyDeviceToDevice, stream);
}

inline cudaError_t BatchDataCompressed::GetNextPipeline(
    const size_t &chunks, const cudaStream_t &stream,
    const size_t &previous_displacement) {
  NvcompUtil::GetNextPipeline(data_, d_ptrs_, previous_displacement, d_sizes_,
                              d_displacements_, chunks, stream);
  return cudaSuccess;
}

inline void *const *BatchDataCompressed::d_ptrs() {
  return reinterpret_cast<void *const *>(d_ptrs_);
}

inline size_t *BatchDataCompressed::d_sizes() { return d_sizes_; }

inline void BatchDataCompressed::GetSize(size_t *size) {
  size_t compressed_data_size = 0;
  cudaMemcpy(&compressed_data_size, d_size_, sizeof(size),
             cudaMemcpyDeviceToHost);
  *size += compressed_data_size;
}
