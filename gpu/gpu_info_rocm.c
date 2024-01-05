#ifndef __APPLE__

#include "gpu_info_rocm.h"

#include <string.h>

#ifndef _WIN32
const char *rocm_lib_paths[] = {
    "librocm_smi64.so",
    "/opt/rocm/lib/librocm_smi64.so",
    NULL,
};
#else
// TODO untested
const char *rocm_lib_paths[] = {
    "rocm_smi64.dll",
    "/opt/rocm/lib/rocm_smi64.dll",
    NULL,
};
#endif

#define ROCM_LOOKUP_SIZE 5

void rocm_init(rocm_init_resp_t *resp) {
  rsmi_status_t ret;
  resp->err = NULL;
  const int buflen = 256;
  char buf[buflen + 1];
  int i;
  struct lookup {
    char *s;
    void **p;
  } l[ROCM_LOOKUP_SIZE] = {
      {"rsmi_init", (void *)&resp->rh.initFn},
      {"rsmi_shut_down", (void *)&resp->rh.shutdownFn},
      {"rsmi_dev_memory_total_get", (void *)&resp->rh.totalMemFn},
      {"rsmi_dev_memory_usage_get", (void *)&resp->rh.usageMemFn},
      {"rsmi_version_get", (void *)&resp->rh.versionGetFn},
      // { "rsmi_dev_id_get", (void*)&resp->rh.getHandle },
  };

  for (i = 0; rocm_lib_paths[i] != NULL && resp->rh.handle == NULL; i++) {
    resp->rh.handle = LOAD_LIBRARY(rocm_lib_paths[i], RTLD_LAZY);
  }
  if (!resp->rh.handle) {
    char *msg = LOAD_ERR();
    snprintf(buf, buflen,
             "Unable to load %s library to query for Radeon GPUs: %s\n",
             rocm_lib_paths[0], msg);
    free(msg);
    resp->err = strdup(buf);
    return;
  }

  for (i = 0; i < ROCM_LOOKUP_SIZE; i++) {
    *l[i].p = LOAD_SYMBOL(resp->rh.handle, l[i].s);
    if (!l[i].p) {
      UNLOAD_LIBRARY(resp->rh.handle);
      char *msg = LOAD_ERR();
      snprintf(buf, buflen, "symbol lookup for %s failed: %s", l[i].s,
               msg);
      free(msg);
      resp->err = strdup(buf);
      return;
    }
  }

  ret = (*resp->rh.initFn)(0);
  if (ret != RSMI_STATUS_SUCCESS) {
    snprintf(buf, buflen, "rocm vram init failure: %d", ret);
    resp->err = strdup(buf);
  }

  return;
}

void rocm_check_vram(rocm_handle_t h, mem_info_t *resp) {
  resp->err = NULL;
  // uint32_t num_devices;
  // uint16_t device;
  uint64_t totalMem = 0;
  uint64_t usedMem = 0;
  rsmi_status_t ret;
  const int buflen = 256;
  char buf[buflen + 1];
  int i;

  if (h.handle == NULL) {
    resp->err = strdup("nvml handle not initialized");
    return;
  }

  // TODO - iterate through devices...  ret =
  // rsmi_num_monitor_devices(&num_devices);

  // ret = (*h.getHandle)(0, &device);
  // if (ret != RSMI_STATUS_SUCCESS) {
  //     printf("rocm vram device lookup failure: %d\n", ret);
  //     return -1;
  // }

  // Get total memory - used memory for available memory
  ret = (*h.totalMemFn)(0, RSMI_MEM_TYPE_VRAM, &totalMem);
  if (ret != RSMI_STATUS_SUCCESS) {
    snprintf(buf, buflen, "rocm total mem lookup failure: %d", ret);
    resp->err = strdup(buf);
    return;
  }
  ret = (*h.usageMemFn)(0, RSMI_MEM_TYPE_VRAM, &usedMem);
  if (ret != RSMI_STATUS_SUCCESS) {
    snprintf(buf, buflen, "rocm usage mem lookup failure: %d", ret);
    resp->err = strdup(buf);
    return;
  }

  resp->total = totalMem;
  resp->free = totalMem - usedMem;
  return;
}

void rocm_get_version(rocm_handle_t h, rocm_version_resp_t *resp) {
  const int buflen = 256;
  char buf[buflen + 1];
  if (h.handle == NULL) {
    resp->str = strdup("nvml handle not initialized");
    resp->status = 1;
    return;
  }
  rsmi_version_t ver;
  rsmi_status_t ret;
  ret = h.versionGetFn(&ver);
  if (ret != RSMI_STATUS_SUCCESS) {
    snprintf(buf, buflen, "unexpected response on version lookup %d", ret);
    resp->status = 1;
  } else {
    snprintf(buf, buflen, "%d", ver.major);
    resp->status = 0;
  }
  resp->str = strdup(buf);
}

#endif  // __APPLE__