/*
 * rCUDA: remote CUDA (www.rCUDA.net)
 * Copyright (C) 2016-2022
 * Grupo de Arquitecturas Paralelas
 * Departamento de Informática de Sistemas y Computadores
 * Universidad Politécnica de Valencia (Spain)
 */

inline cudaError_t BatchDataCompressed::InitilizeCompression(
    char *data, size_t *displacements, const cudaStream_t &stream) {
  data_ = data;
  cudaError_t error = cudaMemsetAsync(d_size_, 0, sizeof(*d_size_), stream);
  if (error == cudaSuccess) {
    error = cudaMemcpyAsync(d_ptrs_, h_ptrs_compression_,
                            sizeof(*h_ptrs_compression_) * slices_,
                            cudaMemcpyHostToDevice, stream);
  }
  d_displacements_ = displacements;
  return error;
}

inline cudaError_t BatchDataCompressed::ConfigureCompression(
    const size_t &max_chunk_size, const cudaStream_t &stream) {
  cudaError_t error{cudaSuccess};
  if (max_chunk_size_ < max_chunk_size) {
    if (max_chunk_size_ == 0) {
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
  NvcompUtil::DumpData(data_, d_ptrs_, d_size_, d_sizes_, d_displacements_,
                       chunks, stream);
  cudaError_t error =
      cudaMemcpyAsync(d_size_, &d_displacements_[chunks - 1], sizeof(*d_size_),
                      cudaMemcpyDeviceToDevice, stream);
  d_displacements_ += chunks;
  return error;
}

inline cudaError_t BatchDataCompressed::InitilizeDecompression(
    char *data, size_t *displacements, const cudaStream_t &stream) {
  data_ = data;
  cudaError_t error = cudaMemsetAsync(d_size_, 0, sizeof(*d_size_), stream);
  d_displacements_ = displacements;
  return error;
}

inline cudaError_t BatchDataCompressed::GetNext(const size_t &chunks,
                                                const cudaStream_t &stream) {
  NvcompUtil::GetNext(data_, d_ptrs_, d_size_, d_sizes_, d_displacements_,
                      chunks, stream);
  cudaError_t error =
      cudaMemcpyAsync(d_size_, &d_displacements_[chunks - 1], sizeof(*d_size_),
                      cudaMemcpyDeviceToDevice, stream);
  d_displacements_ += chunks;
  return error;
}

inline void *const *BatchDataCompressed::d_ptrs() {
  return reinterpret_cast<void *const *>(d_ptrs_);
}

inline size_t *BatchDataCompressed::d_sizes() { return d_sizes_; }

inline size_t BatchDataCompressed::size() {
  size_t size = 0;
  cudaMemcpy(&size, d_size_, sizeof(size), cudaMemcpyDeviceToHost);
  return size;
}
