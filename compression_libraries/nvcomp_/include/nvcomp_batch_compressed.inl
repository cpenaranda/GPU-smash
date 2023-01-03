/*
 * rCUDA: remote CUDA (www.rCUDA.net)
 * Copyright (C) 2016-2022
 * Grupo de Arquitecturas Paralelas
 * Departamento de Informática de Sistemas y Computadores
 * Universidad Politécnica de Valencia (Spain)
 */

inline cudaError_t BatchDataCompressed::InitilizeCompression(
    char *data, size_t *sizes, const size_t &max_chunk_size,
    const cudaStream_t &stream) {
  data_ = data;
  size_ = 0;
  d_sizes_ = sizes;
  cudaError_t error{cudaSuccess};
  compressing_ = true;
  for (uint64_t i_chunk = 0; error == cudaSuccess && i_chunk < slices_;
       ++i_chunk) {
    error = cudaMalloc(&h_ptrs_[i_chunk], max_chunk_size);
  }
  if (error == cudaSuccess) {
    error = cudaMemcpyAsync(d_ptrs_, h_ptrs_, sizeof(*h_ptrs_) * slices_,
                            cudaMemcpyHostToDevice, stream);
  }
  return error;
}

inline cudaError_t BatchDataCompressed::DumpData(const size_t &chunks,
                                                 const cudaStream_t &stream) {
  cudaError_t error = cudaMemcpy(h_sizes_, d_sizes_, chunks * sizeof(*d_sizes_),
                                 cudaMemcpyDeviceToHost);

  for (size_t i_chunk = 0; error == cudaSuccess && i_chunk < chunks;
       ++i_chunk) {
    error = cudaMemcpyAsync(data_, h_ptrs_[i_chunk], h_sizes_[i_chunk],
                            cudaMemcpyDeviceToDevice, stream);
    data_ += h_sizes_[i_chunk];
    size_ += h_sizes_[i_chunk];
  }
  d_sizes_ += chunks;
  return error;
}

inline cudaError_t BatchDataCompressed::InitilizeDecompression(
    char *data, const size_t &data_size, size_t *sizes) {
  data_ = data;
  size_ = data_size;
  d_sizes_ = sizes;
  if (compressing_) {
    compressing_ = false;
    cudaError_t error{cudaSuccess};
    for (uint64_t i_chunk = 0; error == cudaSuccess && i_chunk < slices_;
         ++i_chunk) {
      error = cudaFree(h_ptrs_[i_chunk]);
    }
  }
  return cudaSuccess;
}

inline cudaError_t BatchDataCompressed::GetNext(const size_t &chunks,
                                                const cudaStream_t &stream) {
  cudaError_t error = cudaMemcpy(h_sizes_, d_sizes_, chunks * sizeof(*h_sizes_),
                                 cudaMemcpyDeviceToHost);
  for (size_t i_chunk = 0; error == cudaSuccess && i_chunk < chunks;
       ++i_chunk) {
    h_ptrs_[i_chunk] = data_;
    data_ += h_sizes_[i_chunk];
  }
  if (error == cudaSuccess) {
    error = cudaMemcpyAsync(d_ptrs_, h_ptrs_, chunks * sizeof(*h_ptrs_),
                            cudaMemcpyHostToDevice, stream);
  }
  return error;
}

inline cudaError_t BatchDataCompressed::IncrementSizes(const size_t &chunks) {
  d_sizes_ += chunks;
  return cudaSuccess;
}

inline void *const *BatchDataCompressed::d_ptrs() {
  return reinterpret_cast<void *const *>(d_ptrs_);
}

inline size_t *BatchDataCompressed::d_sizes() { return d_sizes_; }

inline size_t BatchDataCompressed::size() { return size_; }
