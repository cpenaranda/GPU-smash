/*
 * rCUDA: remote CUDA (www.rCUDA.net)
 * Copyright (C) 2016-2022
 * Grupo de Arquitecturas Paralelas
 * Departamento de Informática de Sistemas y Computadores
 * Universidad Politécnica de Valencia (Spain)
 */

#pragma once

#include <functional>
#include <iostream>
#include <map>
#include <string>
#include <vector>

// SMASH LIBRARIES
#include <gpu_compression_library.hpp>

class GpuCompressionLibraries {
 private:
  std::map<std::string, std::function<GpuCompressionLibrary *()>> map_;

 public:
  GpuCompressionLibrary *GetCompressionLibrary(const std::string &library_name);

  std::vector<std::string> GetNameLibraries();

  void GetListInformation();

  GpuCompressionLibraries();
  ~GpuCompressionLibraries();
};
