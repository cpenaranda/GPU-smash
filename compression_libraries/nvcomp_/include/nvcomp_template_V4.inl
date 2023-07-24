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
  for (uint16_t i_stream = 0; i_stream < number_of_streams_; ++i_stream) {
    if (device_temporal_memories_[i_stream]) {
      cudaFree(device_temporal_memories_[i_stream]);
      device_temporal_memories_[i_stream] = nullptr;
    }
  }
}

template <typename Opts_t>
inline cudaError_t NvcompTemplate<Opts_t>::InitializeMemories(
    const size_t &chunk_size, const size_t &max_chunk_size) {
  cudaError_t error = cudaSuccess;
  auxiliar_uncompressed_memory_size_ = batch_size_ * chunk_size;
  auxiliar_compressed_memory_size_ = batch_size_ * max_chunk_size;
  if (chunk_size_ < chunk_size) {
    for (size_t i_memory = 0;
         (error == cudaSuccess) && i_memory < number_of_auxiliar_memories_;
         ++i_memory) {
      error = cudaSetDevice(i_memory % devices_);
      if (error == cudaSuccess) {
        if (list_device_uncompressed_data_[i_memory]) {
          error = cudaFree(list_device_uncompressed_data_[i_memory]);
        }
        if (error == cudaSuccess) {
          error = cudaMalloc(&list_device_uncompressed_data_[i_memory],
                             auxiliar_uncompressed_memory_size_);
        }
      }
    }
  }
  if (max_chunk_size_ < max_chunk_size) {
    for (size_t i_memory = 0;
         (error == cudaSuccess) && i_memory < number_of_auxiliar_memories_;
         ++i_memory) {
      error = cudaSetDevice(i_memory % devices_);
      if (error == cudaSuccess) {
        if (list_device_compressed_data_[i_memory]) {
          error = cudaFree(list_device_compressed_data_[i_memory]);
        }
        if (error == cudaSuccess) {
          error = cudaMalloc(&list_device_compressed_data_[i_memory],
                             auxiliar_compressed_memory_size_);
        }
      }
    }
  }
  return error;
}

template <typename Opts_t>
inline cudaError_t NvcompTemplate<Opts_t>::CompressionH2D(const size_t &batch,
                                                          const uint16_t &id) {
  cudaError_t error = cudaStreamWaitEvent(stream_, list_events_H2D_[id], 0);
  // if (error == cudaSuccess) {
  //   error = static_cast<cudaError_t>(compress_asynchronously_(
  //       list_uncompressed_data_[id]->d_ptrs(),
  //       list_uncompressed_data_[id]->d_sizes(), chunk_size_, batch,
  //       device_temporal_memory_, temporal_memory_size_,
  //       compressed_data_->d_ptrs(), compressed_data_->d_sizes(), options_,
  //       stream_));
  //   if (error == cudaSuccess) {
  //     compressed_data_->DumpData(batch, stream_);
  //     error = cudaEventRecord(main_events_[id], stream_);
  //   }
  // }
  return error;
}

template <typename Opts_t>
inline cudaError_t NvcompTemplate<Opts_t>::CompressionD2H(const size_t &batch,
                                                          const uint16_t &id) {
  cudaError_t error = cudaStreamWaitEvent(stream_, list_events_D2H_[id], 0);
  // if (error == cudaSuccess) {
  //   uncompressed_data_->GetNextCompression(batch, chunk_size_, stream_);
  //   error = static_cast<cudaError_t>(compress_asynchronously_(
  //       uncompressed_data_->d_ptrs(), uncompressed_data_->d_sizes(),
  //       chunk_size_, batch, device_temporal_memory_, temporal_memory_size_,
  //       list_compressed_data_[id]->d_ptrs(),
  //       list_compressed_data_[id]->d_sizes(), options_, stream_));
  //   if (error == cudaSuccess) {
  //     list_compressed_data_[id]->DumpDataPipeline(device_last_batch_size_,
  //                                                 batch, stream_);
  //     error = cudaEventRecord(main_events_[id], stream_);
  //   }
  // }
  return error;
}

template <typename Opts_t>
inline cudaError_t NvcompTemplate<Opts_t>::CompressionH2H(const size_t &batch,
                                                          const uint16_t &id) {
  cudaError_t error = cudaSetDevice(id % devices_);
  if (error == cudaSuccess) {
    error = cudaStreamWaitEvent(stream_kernels_[last_stream_],
                                list_events_H2D_[id], 0);
    if (error == cudaSuccess) {
      error = cudaStreamWaitEvent(stream_kernels_[last_stream_],
                                  list_events_D2H_[id], 0);
      if (error == cudaSuccess) {
        error = static_cast<cudaError_t>(compress_asynchronously_(
            list_uncompressed_data_[id]->d_ptrs(),
            list_uncompressed_data_[id]->d_sizes(), chunk_size_, batch,
            device_temporal_memories_[last_stream_], temporal_memory_size_,
            list_compressed_data_[id]->d_ptrs(),
            list_compressed_data_[id]->d_sizes(), options_,
            stream_kernels_[last_stream_]));
        if (error == cudaSuccess) {
          list_compressed_data_[id]->DumpDataPipeline2(
              batch, stream_kernels_[last_stream_]);
          error = cudaEventRecord(list_events_kernel_[id],
                                  stream_kernels_[last_stream_]);
        }
      }
    }
    ++last_stream_ %= number_of_streams_;
  }
  return error;
}

template <typename Opts_t>
inline cudaError_t NvcompTemplate<Opts_t>::DecompressionH2D(
    const size_t &batch, const uint16_t &id) {
  cudaError_t error = cudaStreamWaitEvent(stream_, list_events_H2D_[id], 0);
  // if (error == cudaSuccess) {
  //   uncompressed_data_->GetNextDecompression(batch, chunk_size_, stream_);
  //   error = static_cast<cudaError_t>(decompress_asynchronously_(
  //       list_compressed_data_[id]->d_ptrs(),
  //       list_compressed_data_[id]->d_sizes(), uncompressed_data_->d_sizes(),
  //       uncompressed_data_->d_sizes(), batch, device_temporal_memory_,
  //       temporal_memory_size_, uncompressed_data_->d_ptrs(), statuses_,
  //       stream_));
  //   if (error == cudaSuccess) {
  //     error = cudaEventRecord(main_events_[id], stream_);
  //   }
  // }
  return error;
}

template <typename Opts_t>
inline cudaError_t NvcompTemplate<Opts_t>::DecompressionD2H(
    const size_t &batch, const uint16_t &id) {
  cudaError_t error = cudaStreamWaitEvent(stream_, list_events_D2H_[id], 0);
  // if (error == cudaSuccess) {
  //   compressed_data_->GetNext(batch, stream_);
  //   error = static_cast<cudaError_t>(decompress_asynchronously_(
  //       compressed_data_->d_ptrs(), compressed_data_->d_sizes(),
  //       list_uncompressed_data_[id]->d_sizes(),
  //       list_uncompressed_data_[id]->d_sizes(), batch,
  //       device_temporal_memory_, temporal_memory_size_,
  //       list_uncompressed_data_[id]->d_ptrs(), statuses_, stream_));
  //   if (error == cudaSuccess) {
  //     error = cudaEventRecord(main_events_[id], stream_);
  //   }
  // }
  return error;
}

template <typename Opts_t>
inline cudaError_t NvcompTemplate<Opts_t>::DecompressionH2H(
    const size_t &batch, const uint16_t &id,
    const size_t &previous_displacement) {
  cudaError_t error = cudaSetDevice(id % devices_);
  if (error == cudaSuccess) {
    error = cudaStreamWaitEvent(stream_kernels_[last_stream_],
                                list_events_H2D_[id], 0);
    list_compressed_data_[id]->GetNextPipeline(
        batch, stream_kernels_[last_stream_], previous_displacement);
    if (error == cudaSuccess) {
      error = cudaStreamWaitEvent(stream_kernels_[last_stream_],
                                  list_events_D2H_[id], 0);
      if (error == cudaSuccess) {
        error = static_cast<cudaError_t>(decompress_asynchronously_(
            list_compressed_data_[id]->d_ptrs(),
            list_compressed_data_[id]->d_sizes(),
            list_uncompressed_data_[id]->d_sizes(),
            list_uncompressed_data_[id]->d_sizes(), batch,
            device_temporal_memories_[last_stream_], temporal_memory_size_,
            list_uncompressed_data_[id]->d_ptrs(), statuses_[id % devices_],
            stream_kernels_[last_stream_]));
        if (error == cudaSuccess) {
          error = cudaEventRecord(list_events_kernel_[id],
                                  stream_kernels_[last_stream_]);
        }
      }
    }
    ++last_stream_ %= number_of_streams_;
  }
  return error;
}

template <typename Opts_t>
inline cudaError_t NvcompTemplate<Opts_t>::CompressionMemcpyH2D(
    size_t *batch, size_t *current_batch_size, char **host_uncompressed_data,
    const uint32_t &uncompressed_data_size, const uint16_t &id) {
  cudaError_t error = cudaSetDevice(id % devices_);
  if (error == cudaSuccess) {
    size_t current_uncompressed_size = auxiliar_uncompressed_memory_size_;
    error = cudaStreamWaitEvent(stream_H2D_[id % devices_],
                                list_events_kernel_[id], 0);
    if (error == cudaSuccess) {
      if (*batch > *current_batch_size) {
        *batch = *current_batch_size;
        current_uncompressed_size =
            uncompressed_data_size % current_uncompressed_size;
        if (current_uncompressed_size % chunk_size_) {
          // Every batch has a 'chunk_size_' size, but the last one may have a
          // different size, so we must update it.
          *host_last_batch_size_ = current_uncompressed_size % chunk_size_;
          error = cudaMemcpyAsync(
              &list_uncompressed_data_[id]->d_sizes()[(*batch) - 1],
              host_last_batch_size_, sizeof(*host_last_batch_size_),
              cudaMemcpyHostToDevice, stream_H2D_[id % devices_]);
        }
      }
      if (error == cudaSuccess) {
        error =
            cudaMemcpyAsync(list_device_uncompressed_data_[id],
                            *host_uncompressed_data, current_uncompressed_size,
                            cudaMemcpyHostToDevice, stream_H2D_[id % devices_]);
        if (error == cudaSuccess) {
          error =
              cudaEventRecord(list_events_H2D_[id], stream_H2D_[id % devices_]);
        }
      }
    }
    *host_uncompressed_data += current_uncompressed_size;
    *current_batch_size -= *batch;
  }
  return error;
}

template <typename Opts_t>
inline cudaError_t NvcompTemplate<Opts_t>::CompressionMemcpyD2H(
    char *host_compressed_data, size_t *host_compressed_displacements,
    const size_t &batch, const uint16_t &id, const bool &no_first_time) {
  cudaError_t error = cudaSetDevice(id % devices_);
  if (error == cudaSuccess) {
    error = cudaStreamWaitEvent(stream_, list_events_kernel_[id], 0);
    if (error == cudaSuccess) {
      size_t previous_displacement =
          (no_first_time) ? *(host_compressed_displacements - 1) : 0;
      error = cudaMemcpyAsync(host_compressed_displacements,
                              list_device_displacements_[id],
                              batch * sizeof(*host_compressed_displacements),
                              cudaMemcpyDeviceToHost, stream_);
      if (error == cudaSuccess) {
        error = cudaStreamSynchronize(stream_);
        if (error == cudaSuccess) {
          error = cudaMemcpyAsync(host_compressed_data + previous_displacement,
                                  list_device_compressed_data_[id],
                                  host_compressed_displacements[batch - 1],
                                  cudaMemcpyDeviceToHost,
                                  stream_D2H_[id % devices_]);
          for (size_t i_batch = 0; i_batch < batch; ++i_batch) {
            host_compressed_displacements[i_batch] += previous_displacement;
          }
          if (error == cudaSuccess) {
            error = cudaEventRecord(list_events_D2H_[id],
                                    stream_D2H_[id % devices_]);
          }
        }
      }
    }
  }
  return error;
}

template <typename Opts_t>
inline cudaError_t NvcompTemplate<Opts_t>::DecompressionMemcpyH2D(
    char *host_compressed_data, size_t *host_compressed_displacements,
    const uint32_t &batch, const uint16_t &id, const bool &no_first_time) {
  cudaError_t error = cudaSetDevice(id % devices_);
  if (error == cudaSuccess) {
    size_t previous_displacement =
        (no_first_time) ? *(host_compressed_displacements - 1) : 0;
    error = cudaStreamWaitEvent(stream_H2D_[id % devices_],
                                list_events_kernel_[id], 0);
    if (error == cudaSuccess) {
      error = cudaMemcpyAsync(
          list_device_displacements_[id], host_compressed_displacements,
          batch * sizeof(*host_compressed_displacements),
          cudaMemcpyHostToDevice, stream_H2D_[id % devices_]);
      if (error == cudaSuccess) {
        error = cudaMemcpyAsync(
            list_device_compressed_data_[id],
            host_compressed_data + previous_displacement,
            host_compressed_displacements[batch - 1] - previous_displacement,
            cudaMemcpyHostToDevice, stream_H2D_[id % devices_]);
        if (error == cudaSuccess) {
          error =
              cudaEventRecord(list_events_H2D_[id], stream_H2D_[id % devices_]);
        }
      }
    }
  }
  return error;
}

template <typename Opts_t>
inline cudaError_t NvcompTemplate<Opts_t>::DecompressionMemcpyD2H(
    char *host_decompressed_data, uint64_t *decompressed_data_size,
    const bool &last_copy, const uint32_t &batch, const uint16_t &id) {
  cudaError_t error = cudaSetDevice(id % devices_);
  if (error == cudaSuccess) {
    error = cudaStreamWaitEvent(stream_D2H_[id % devices_],
                                list_events_kernel_[id], 0);
    if (error == cudaSuccess) {
      size_t size = (last_copy) ? list_uncompressed_data_[id]->size(batch)
                                : batch * chunk_size_;
      error = cudaMemcpyAsync(
          host_decompressed_data, list_device_uncompressed_data_[id], size,
          cudaMemcpyDeviceToHost, stream_D2H_[id % devices_]);
      if (error == cudaSuccess) {
        *decompressed_data_size += size;
        error =
            cudaEventRecord(list_events_D2H_[id], stream_D2H_[id % devices_]);
      }
    }
  }
  return error;
}

template <typename Opts_t>
inline bool NvcompTemplate<Opts_t>::InitializeCompression(
    const size_t &chunk_size, const Opts_t &configuration,
    const cudaStream_t &stream) {
  size_t max_chunk_size = 0;
  int current_device = 0;
  cudaError_t error = cudaGetDevice(&current_device);
  stream_ = stream;
  options_ = configuration;
  RemoveTemporalMemory();
  get_temporal_size_to_compress_(batch_size_, chunk_size, options_,
                                 &temporal_memory_size_);
  get_max_compressed_chunk_size_(chunk_size, options_, &max_chunk_size);
  compressed_data_->ConfigureCompression(max_chunk_size);
  error = cudaMalloc(&device_temporal_memory_, temporal_memory_size_);
  for (uint16_t i_stream = 0;
       (error == cudaSuccess) && (i_stream < number_of_streams_); ++i_stream) {
    cudaSetDevice(i_stream % devices_);
    error =
        cudaMalloc(&device_temporal_memories_[i_stream], temporal_memory_size_);
  }
  if (error == cudaSuccess) {
    error = InitializeMemories(chunk_size, max_chunk_size);
    if (error == cudaSuccess) {
      for (size_t i_memory = 0; i_memory < number_of_auxiliar_memories_;
           ++i_memory) {
        error = cudaSetDevice(i_memory % devices_);
        if (error == cudaSuccess) {
          list_uncompressed_data_[i_memory]->InitializeCompression(
              list_device_uncompressed_data_[i_memory],
              auxiliar_uncompressed_memory_size_);
          list_uncompressed_data_[i_memory]->GetNextCompression(
              batch_size_, chunk_size, stream_kernels_[i_memory % devices_]);

          list_compressed_data_[i_memory]->ConfigureCompression(max_chunk_size);
          list_compressed_data_[i_memory]->InitializeCompression(
              list_device_compressed_data_[i_memory],
              list_device_displacements_[i_memory],
              stream_kernels_[i_memory % devices_]);
        }
      }
      for (int i_device = 0; (error == cudaSuccess) && (i_device < devices_);
           ++i_device) {
        error = cudaStreamSynchronize(stream_kernels_[i_device]);
      }
    }
  }
  chunk_size_ = chunk_size;
  max_chunk_size_ = max_chunk_size;
  cudaSetDevice(current_device);
  return error == cudaSuccess;
}

template <typename Opts_t>
inline bool NvcompTemplate<Opts_t>::InitializeDecompression(
    const size_t &chunk_size, const cudaStream_t &stream) {
  stream_ = stream;
  int current_device = 0;
  cudaError_t error = cudaGetDevice(&current_device);
  RemoveTemporalMemory();
  get_temporal_size_to_decompress_(batch_size_, chunk_size,
                                   &temporal_memory_size_);
  error = cudaMalloc(&device_temporal_memory_, temporal_memory_size_);
  for (uint16_t i_stream = 0;
       (error == cudaSuccess) && (i_stream < number_of_streams_); ++i_stream) {
    cudaSetDevice(i_stream % devices_);
    error =
        cudaMalloc(&device_temporal_memories_[i_stream], temporal_memory_size_);
  }
  if (error == cudaSuccess) {
    size_t max_chunk_size;
    get_max_compressed_chunk_size_(chunk_size, options_, &max_chunk_size);
    error = InitializeMemories(chunk_size, max_chunk_size);
    if (error == cudaSuccess) {
      for (size_t i_memory = 0; i_memory < number_of_auxiliar_memories_;
           ++i_memory) {
        error = cudaSetDevice(i_memory % devices_);
        if (error == cudaSuccess) {
          list_uncompressed_data_[i_memory]->InitializeDecompression(
              list_device_uncompressed_data_[i_memory],
              stream_kernels_[i_memory % devices_]);
          list_uncompressed_data_[i_memory]->GetNextDecompression(
              batch_size_, chunk_size, stream_kernels_[i_memory % devices_]);

          list_compressed_data_[i_memory]->InitializeDecompression(
              list_device_compressed_data_[i_memory],
              list_device_displacements_[i_memory],
              stream_kernels_[i_memory % devices_]);
        }
      }

      for (int i_device = 0; (error == cudaSuccess) && (i_device < devices_);
           ++i_device) {
        error = cudaStreamSynchronize(stream_kernels_[i_device]);
      }
    }
  }
  chunk_size_ = chunk_size;
  cudaSetDevice(current_device);
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
  last_stream_ = 0;
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
  last_stream_ = 0;
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
  cudaError_t result{cudaSuccess};
  size_t current_batch_size;
  size_t *device_compressed_displacements;
  SetHeader(&current_batch_size, uncompressed_data_size,
            &device_compressed_displacements, &device_compressed_data,
            compressed_data_size);
  compressed_data_->InitializeCompression(device_compressed_data,
                                          device_compressed_displacements,
                                          stream_kernels_[last_stream_]);
  size_t min_batch = batch_size_;
  char *uncompressed_data = const_cast<char *>(host_uncompressed_data);
  for (uint16_t i_memory = 0; (result == cudaSuccess) && current_batch_size;
       ++i_memory %= number_of_auxiliar_memories_) {
    // Copy H2D
    result = CompressionMemcpyH2D(&min_batch, &current_batch_size,
                                  &uncompressed_data, uncompressed_data_size,
                                  i_memory);
    if (result != cudaSuccess) break;
    // Kernel
    result = CompressionH2D(min_batch, i_memory);
    if (result != cudaSuccess) break;
  }
  if (result == cudaSuccess) {
    compressed_data_->GetSize(compressed_data_size);
  } else {
    *compressed_data_size = 0;
  }
  return (result == cudaSuccess);
}

template <typename Opts_t>
inline bool NvcompTemplate<Opts_t>::CompressDeviceToHost(
    const char *const device_uncompressed_data,
    const uint64_t &uncompressed_data_size, char *host_compressed_data,
    uint64_t *compressed_data_size) {
  cudaError_t result{cudaSuccess};
  bool no_first_time = false;
  size_t current_batch_size;
  size_t *host_compressed_displacements;
  SetHeaderHost(&current_batch_size, uncompressed_data_size,
                &host_compressed_displacements, &host_compressed_data,
                compressed_data_size);
  uncompressed_data_->InitializeCompression(device_uncompressed_data,
                                            uncompressed_data_size);
  size_t min_batch = batch_size_;
  size_t batches[number_of_auxiliar_memories_];
  uint16_t i_kernel_memory;
  // Kernel
  for (i_kernel_memory = 0; (result == cudaSuccess) && current_batch_size &&
                            (i_kernel_memory < number_of_auxiliar_memories_);
       ++i_kernel_memory) {
    if (min_batch > current_batch_size) min_batch = current_batch_size;
    result = CompressionD2H(min_batch, i_kernel_memory);
    batches[i_kernel_memory] = min_batch;
    current_batch_size -= min_batch;
  }
  uint16_t i_d2h_memory = 0;
  for (uint16_t i_memory = 0;
       (result == cudaSuccess) && (i_kernel_memory > i_d2h_memory);
       i_memory = i_d2h_memory % number_of_auxiliar_memories_) {
    // Copy D2H
    result = CompressionMemcpyD2H(host_compressed_data,
                                  host_compressed_displacements,
                                  batches[i_memory], i_memory, no_first_time);
    host_compressed_displacements += batches[i_memory];
    if (!no_first_time) no_first_time = true;
    if ((result == cudaSuccess) && current_batch_size) {
      // Kernel
      if (min_batch > current_batch_size) min_batch = current_batch_size;
      result = CompressionD2H(min_batch, i_memory);
      batches[i_memory] = min_batch;
      current_batch_size -= min_batch;
      ++i_kernel_memory;
    }
    ++i_d2h_memory;
  }
  for (uint16_t i_device = 0; (result == cudaSuccess) && (i_device < devices_);
       ++i_device) {
    result = cudaStreamSynchronize(stream_D2H_[i_device]);
  }
  *compressed_data_size = *(host_compressed_displacements - 1);
  return (result == cudaSuccess);
}

template <typename Opts_t>
inline bool NvcompTemplate<Opts_t>::CompressHostToHost(
    const char *const host_uncompressed_data,
    const uint64_t &uncompressed_data_size, char *host_compressed_data,
    uint64_t *compressed_data_size) {
  cudaError_t result{cudaSuccess};
  bool no_first_time = false;
  size_t current_batch_size;
  size_t *host_compressed_displacements;
  SetHeaderHost(&current_batch_size, uncompressed_data_size,
                &host_compressed_displacements, &host_compressed_data,
                compressed_data_size);
  size_t min_batch = batch_size_;
  size_t batches[number_of_auxiliar_memories_];
  uint16_t i_kernel_memory;
  int current_device = 0;
  result = cudaGetDevice(&current_device);
  char *uncompressed_data = const_cast<char *>(host_uncompressed_data);
  for (i_kernel_memory = 0; (result == cudaSuccess) && current_batch_size &&
                            (i_kernel_memory < number_of_auxiliar_memories_);
       ++i_kernel_memory) {
    // Copy H2D
    result = CompressionMemcpyH2D(&min_batch, &current_batch_size,
                                  &uncompressed_data, uncompressed_data_size,
                                  i_kernel_memory);
    if (result == cudaSuccess) {
      // Kernel
      result = CompressionH2H(min_batch, i_kernel_memory);
      batches[i_kernel_memory] = min_batch;
    }
  }
  uint16_t i_d2h_memory = 0;
  for (uint16_t i_memory = 0;
       (result == cudaSuccess) && (i_kernel_memory > i_d2h_memory);
       i_memory = i_d2h_memory % number_of_auxiliar_memories_) {
    // Copy D2H
    result = CompressionMemcpyD2H(host_compressed_data,
                                  host_compressed_displacements,
                                  batches[i_memory], i_memory, no_first_time);
    host_compressed_displacements += batches[i_memory];
    if (!no_first_time) no_first_time = true;
    if ((result == cudaSuccess) && current_batch_size) {
      // Copy H2D
      result = CompressionMemcpyH2D(&min_batch, &current_batch_size,
                                    &uncompressed_data, uncompressed_data_size,
                                    i_memory);
      if (result == cudaSuccess) {
        // Kernel
        result = CompressionH2H(min_batch, i_memory);
        batches[i_memory] = min_batch;
      }
      ++i_kernel_memory;
    }
    ++i_d2h_memory;
  }
  for (uint16_t i_device = 0; (result == cudaSuccess) && (i_device < devices_);
       ++i_device) {
    result = cudaStreamSynchronize(stream_D2H_[i_device]);
  }
  *compressed_data_size = *(host_compressed_displacements - 1);
  cudaSetDevice(current_device);
  return (result == cudaSuccess);
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
  int current_device = 0;
  cudaGetDevice(&current_device);
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
             uncompressed_data_->d_ptrs(), statuses_[current_device], stream_));
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
  cudaError_t result{cudaSuccess};
  *decompressed_data_size = 0;
  size_t current_batch_size;
  size_t *device_compressed_displacements;
  char *compressed_data = const_cast<char *>(device_compressed_data);
  GetHeader(&current_batch_size, &device_compressed_displacements,
            &compressed_data);
  compressed_data_->InitializeDecompression(compressed_data,
                                            device_compressed_displacements,
                                            stream_kernels_[last_stream_]);
  size_t min_batch = batch_size_;
  for (uint16_t i_memory = 0; (result == cudaSuccess) && current_batch_size;
       ++i_memory %= number_of_auxiliar_memories_) {
    if (min_batch > current_batch_size) min_batch = current_batch_size;
    // Kernel
    result = DecompressionD2H(min_batch, i_memory);
    if (result != cudaSuccess) break;
    // Copy D2H
    result = DecompressionMemcpyD2H(
        host_decompressed_data + *decompressed_data_size,
        decompressed_data_size, (min_batch == current_batch_size), min_batch,
        i_memory);
    if (result != cudaSuccess) break;
    current_batch_size -= min_batch;
  }
  for (uint16_t i_device = 0; (result == cudaSuccess) && (i_device < devices_);
       ++i_device) {
    result = cudaStreamSynchronize(stream_D2H_[i_device]);
  }
  return (result == cudaSuccess);
}

template <typename Opts_t>
inline bool NvcompTemplate<Opts_t>::DecompressHostToDevice(
    const char *const host_compressed_data,
    const uint64_t &compressed_data_size, char *device_decompressed_data,
    uint64_t *decompressed_data_size) {
  cudaError_t result{cudaSuccess};
  bool no_first_time = false;
  *decompressed_data_size = 0;
  size_t current_batch_size;
  size_t *host_compressed_displacements;
  char *compressed_data = const_cast<char *>(host_compressed_data);
  GetHeaderHost(&current_batch_size, &host_compressed_displacements,
                &compressed_data);
  uncompressed_data_->InitializeDecompression(device_decompressed_data,
                                              stream_kernels_[last_stream_]);
  size_t min_batch = batch_size_;
  for (uint16_t i_memory = 0; (result == cudaSuccess) && current_batch_size;
       ++i_memory %= number_of_auxiliar_memories_) {
    if (min_batch > current_batch_size) min_batch = current_batch_size;
    // Copy H2D
    result =
        DecompressionMemcpyH2D(compressed_data, host_compressed_displacements,
                               min_batch, i_memory, no_first_time);
    if (result != cudaSuccess) break;
    // Kernel
    result = DecompressionH2D(min_batch, i_memory);
    if (result != cudaSuccess) break;
    host_compressed_displacements += min_batch;
    current_batch_size -= min_batch;
    if (!no_first_time) no_first_time = true;
  }
  if (result == cudaSuccess) {
    *decompressed_data_size = uncompressed_data_->size();
  } else {
    *decompressed_data_size = 0;
  }
  return (result == cudaSuccess);
}

template <typename Opts_t>
inline bool NvcompTemplate<Opts_t>::DecompressHostToHost(
    const char *const host_compressed_data,
    const uint64_t &compressed_data_size, char *host_decompressed_data,
    uint64_t *decompressed_data_size) {
  cudaError_t result{cudaSuccess};
  bool no_first_time = false;
  *decompressed_data_size = 0;
  size_t current_batch_size;
  size_t *host_compressed_displacements;
  char *compressed_data = const_cast<char *>(host_compressed_data);
  GetHeaderHost(&current_batch_size, &host_compressed_displacements,
                &compressed_data);
  size_t min_batch = batch_size_;
  int current_device = 0;
  result = cudaGetDevice(&current_device);
  for (uint16_t i_memory = 0; (result == cudaSuccess) && current_batch_size;
       ++i_memory %= number_of_auxiliar_memories_) {
    if (result == cudaSuccess) {
      if (min_batch > current_batch_size) min_batch = current_batch_size;
      // Copy H2D
      result =
          DecompressionMemcpyH2D(compressed_data, host_compressed_displacements,
                                 min_batch, i_memory, no_first_time);
      if (result != cudaSuccess) break;
      // Stream Kernels
      result = DecompressionH2H(
          min_batch, i_memory,
          (no_first_time) ? *(host_compressed_displacements - 1) : 0);
      if (result != cudaSuccess) break;
      // Copy D2H
      result = DecompressionMemcpyD2H(
          host_decompressed_data + *decompressed_data_size,
          decompressed_data_size, (min_batch == current_batch_size), min_batch,
          i_memory);
      if (result != cudaSuccess) break;
      host_compressed_displacements += min_batch;
      current_batch_size -= min_batch;
      if (!no_first_time) no_first_time = true;
    }
  }
  for (uint16_t i_device = 0; (result == cudaSuccess) && (i_device < devices_);
       ++i_device) {
    result = cudaStreamSynchronize(stream_D2H_[i_device]);
  }
  cudaSetDevice(current_device);
  return (result == cudaSuccess);
}

template <typename Opts_t>
bool NvcompTemplate<Opts_t>::CreateInternalStructures(const size_t &batch_size,
                                                      const uint8_t &streams) {
  bool result{true};
  if ((batch_size_ != batch_size) ||
      (number_of_streams_ != (streams * devices_))) {
    DestroyInternalStructures();
    batch_size_ = batch_size;
    number_of_streams_ = streams * devices_;
    number_of_auxiliar_memories_ *= number_of_streams_;
    compressed_data_ = new BatchDataCompressed(batch_size_);
    uncompressed_data_ = new BatchDataUncompressed(batch_size_);
    int current_device = 0;
    cudaGetDevice(&current_device);
    cudaHostAlloc(&host_last_batch_size_, sizeof(*host_last_batch_size_), 0);
    last_stream_ = 0;
    stream_kernels_ = new cudaStream_t[number_of_streams_];
    device_temporal_memories_ = new char *[number_of_streams_];
    for (uint16_t i_stream = 0; i_stream < number_of_streams_; ++i_stream) {
      cudaSetDevice(i_stream % devices_);
      cudaStreamCreate(&stream_kernels_[i_stream]);
      device_temporal_memories_[i_stream] = nullptr;
    }
    list_device_uncompressed_data_ = new char *[number_of_auxiliar_memories_];
    list_device_compressed_data_ = new char *[number_of_auxiliar_memories_];
    list_device_displacements_ = new size_t *[number_of_auxiliar_memories_];
    list_uncompressed_data_ =
        new BatchDataUncompressed *[number_of_auxiliar_memories_];
    list_compressed_data_ =
        new BatchDataCompressed *[number_of_auxiliar_memories_];
    list_events_H2D_ = new cudaEvent_t[number_of_auxiliar_memories_];
    list_events_D2H_ = new cudaEvent_t[number_of_auxiliar_memories_];
    list_events_kernel_ = new cudaEvent_t[number_of_auxiliar_memories_];
    for (size_t i_memory = 0; i_memory < number_of_auxiliar_memories_;
         ++i_memory) {
      cudaSetDevice(i_memory % devices_);
      list_device_uncompressed_data_[i_memory] = nullptr;
      list_device_compressed_data_[i_memory] = nullptr;
      cudaMalloc(&list_device_displacements_[i_memory],
                 sizeof(*list_device_displacements_) * batch_size_);
      list_uncompressed_data_[i_memory] =
          new BatchDataUncompressed(batch_size_);
      list_compressed_data_[i_memory] = new BatchDataCompressed(batch_size_);
      cudaEventCreate(&list_events_kernel_[i_memory]);
      cudaEventCreate(&list_events_H2D_[i_memory]);
      cudaEventCreate(&list_events_D2H_[i_memory]);
    }
    statuses_ = new nvcompStatus_t *[devices_];
    stream_H2D_ = new cudaStream_t[devices_];
    stream_D2H_ = new cudaStream_t[devices_];
    for (uint16_t i_device = 0; i_device < devices_; ++i_device) {
      cudaSetDevice(i_device % devices_);
      cudaStreamCreate(&stream_H2D_[i_device]);
      cudaStreamCreate(&stream_D2H_[i_device]);
      cudaMalloc(&statuses_[i_device],
                 sizeof(*statuses_[i_device]) * batch_size_);
    }
    cudaSetDevice(current_device);
  }
  return result;
}

template <typename Opts_t>
void NvcompTemplate<Opts_t>::DestroyInternalStructures() {
  if (compressed_data_) {
    delete compressed_data_;
    delete uncompressed_data_;
    for (uint16_t i_device = 0; i_device < devices_; ++i_device) {
      cudaFree(statuses_[i_device]);
      cudaStreamDestroy(stream_H2D_[i_device]);
      cudaStreamDestroy(stream_D2H_[i_device]);
    }
    delete[] statuses_;
    delete[] stream_H2D_;
    delete[] stream_D2H_;
    cudaFreeHost(host_last_batch_size_);
    for (size_t i_memory = 0; i_memory < number_of_auxiliar_memories_;
         ++i_memory) {
      if (list_device_uncompressed_data_[i_memory]) {
        cudaFree(list_device_uncompressed_data_[i_memory]);
      }
      if (list_device_compressed_data_[i_memory]) {
        cudaFree(list_device_compressed_data_[i_memory]);
      }
      delete list_uncompressed_data_[i_memory];
      delete list_compressed_data_[i_memory];
      cudaFree(list_device_displacements_[i_memory]);
      cudaEventDestroy(list_events_H2D_[i_memory]);
      cudaEventDestroy(list_events_D2H_[i_memory]);
      cudaEventDestroy(list_events_kernel_[i_memory]);
    }
    for (size_t i_stream = 0; i_stream < number_of_streams_; ++i_stream) {
      cudaStreamDestroy(stream_kernels_[i_stream]);
    }
    delete[] stream_kernels_;
    delete[] device_temporal_memories_;
    delete[] list_device_uncompressed_data_;
    delete[] list_device_compressed_data_;
    delete[] list_device_displacements_;
    delete[] list_uncompressed_data_;
    delete[] list_compressed_data_;
    delete[] list_events_H2D_;
    delete[] list_events_D2H_;
    delete[] list_events_kernel_;
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
      temporal_memory_size_(0),
      stream_H2D_(nullptr),
      stream_D2H_(nullptr),
      host_last_batch_size_(nullptr),
      number_of_streams_(0),
      last_stream_(0),
      stream_kernels_(nullptr),
      device_temporal_memories_(nullptr),
      auxiliar_uncompressed_memory_size_(0),
      auxiliar_compressed_memory_size_(0),
      number_of_auxiliar_memories_(2),  // Per stream
      list_device_uncompressed_data_(nullptr),
      list_device_compressed_data_(nullptr),
      list_device_displacements_(nullptr),
      list_uncompressed_data_(nullptr),
      list_compressed_data_(nullptr),
      list_events_H2D_(nullptr),
      list_events_D2H_(nullptr),
      list_events_kernel_(nullptr) {
  cudaGetDeviceCount(&devices_);
}

template <typename Opts_t>
NvcompTemplate<Opts_t>::~NvcompTemplate() {
  RemoveTemporalMemory();
  DestroyInternalStructures();
}
