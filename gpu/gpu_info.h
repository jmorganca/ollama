#ifndef __APPLE__
#ifndef __GPU_INFO_H__
#define __GPU_INFO_H__
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>

#ifndef _WIN32
#include <dlfcn.h>
#define LOAD_LIBRARY(lib, flags) dlopen(lib, flags)
#define LOAD_SYMBOL(handle, sym) dlsym(handle, sym)
#define LOAD_ERR() dlerror()
#define UNLOAD_LIBRARY(handle) dlclose(handle)
#else
#include <windows.h>
#define LOAD_LIBRARY(lib, flags) LoadLibrary(lib)
#define LOAD_SYMBOL(handle, sym) GetProcAddress(handle, sym)
#define UNLOAD_LIBRARY(handle) FreeLibrary(handle)

// TODO - refactor this with proper error message handling on windows
inline static char *LOAD_ERR() {
  static char errbuf[8];
  snprintf(errbuf, 8, "0x%lx", GetLastError());
  return errbuf;
}

#endif

#ifdef __cplusplus
extern "C" {
#endif

typedef struct mem_info {
  uint64_t total;
  uint64_t free;
  char *err;  // If non-nill, caller responsible for freeing
} mem_info_t;

void cpu_check_ram(mem_info_t *resp);

#ifdef __cplusplus
}
#endif

#include "gpu_info_cuda.h"
#include "gpu_info_rocm.h"

#endif  // __GPU_INFO_H__
#endif  // __APPLE__