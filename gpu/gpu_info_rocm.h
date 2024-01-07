#ifndef __APPLE__
#ifndef __GPU_INFO_ROCM_H__
#define __GPU_INFO_ROCM_H__
#include "gpu_info.h"

// Just enough typedef's to dlopen/dlsym for memory information
typedef enum rsmi_status_return {
  RSMI_STATUS_SUCCESS = 0,
  // Other values omitted for now...
} rsmi_status_t;

typedef enum rsmi_memory_type {
  RSMI_MEM_TYPE_VRAM = 0,
  RSMI_MEM_TYPE_VIS_VRAM,
  RSMI_MEM_TYPE_GTT,
} rsmi_memory_type_t;

 typedef struct {
     uint32_t major;     
     uint32_t minor;     
     uint32_t patch;     
     const char *build;  
 } rsmi_version_t;

typedef struct rocm_handle {
  void *handle;
  rsmi_status_t (*initFn)(uint64_t);
  rsmi_status_t (*shutdownFn)(void);
  rsmi_status_t (*totalMemFn)(uint32_t, rsmi_memory_type_t, uint64_t *);
  rsmi_status_t (*usageMemFn)(uint32_t, rsmi_memory_type_t, uint64_t *);
  rsmi_status_t (*versionGetFn) (rsmi_version_t *version);
  // rsmi_status_t (*getHandle)(uint32_t, uint16_t *);
} rocm_handle_t;

typedef struct rocm_init_resp {
  char *err;  // If err is non-null handle is invalid
  rocm_handle_t rh;
} rocm_init_resp_t;

typedef struct rocm_version_resp {
  rsmi_status_t status;
  char *str; // Contains version or error string if status != 0 
} rocm_version_resp_t;

void rocm_init(rocm_init_resp_t *resp);
void rocm_check_vram(rocm_handle_t rh, mem_info_t *resp);
void rocm_get_version(rocm_handle_t rh, rocm_version_resp_t *resp);

#endif  // __GPU_INFO_ROCM_H__
#endif  // __APPLE__