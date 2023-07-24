/*
 * rCUDA: remote CUDA (www.rCUDA.net)
 * Copyright (C) 2016-2022
 * Grupo de Arquitecturas Paralelas
 * Departamento de Informática de Sistemas y Computadores
 * Universidad Politécnica de Valencia (Spain)
 */

#pragma once
#ifdef GPU_VERSION
  #if GPU_VERSION == 2
    #include <nvcomp_template_V2.hpp>
  #elif GPU_VERSION == 3
    #include <nvcomp_template_V3.hpp>
  #elif GPU_VERSION == 4
    #include <nvcomp_template_V4.hpp>
  #else
    #include <nvcomp_template_V1.hpp>
  #endif
#else
  #include <nvcomp_template_V1.hpp>
#endif
