#include "rocm_shim.h"

#include <stdio.h>
#include <string.h>

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

void rocm_shim_init(const char *libPath, struct rocm_llama_server *s,
                    ext_server_resp_t *err) {
  int i = 0;
  struct lookup {
    char *s;
    void **p;
  } l[] = {
      {"llama_server_init", (void *)&s->llama_server_init},
      {"llama_server_start", (void *)&s->llama_server_start},
      {"llama_server_stop", (void *)&s->llama_server_stop},
      {"llama_server_completion", (void *)&s->llama_server_completion},
      {"llama_server_completion_next_result",
       (void *)&s->llama_server_completion_next_result},
      {"llama_server_completion_cancel",
       (void *)&s->llama_server_completion_cancel},
      {"llama_server_release_task_result",
       (void *)&s->llama_server_release_task_result},
      {"llama_server_tokenize", (void *)&s->llama_server_tokenize},
      {"llama_server_detokenize", (void *)&s->llama_server_detokenize},
      {"llama_server_embedding", (void *)&s->llama_server_embedding},
      {"llama_server_release_json_resp",
       (void *)&s->llama_server_release_json_resp},
      {"", NULL},
  };

  printf("Lazy loading %s library\n", libPath);
  s->handle = LOAD_LIBRARY(libPath, RTLD_LAZY);
  if (!s->handle) {
    err->id = -1;
    snprintf(
        err->msg, err->msg_len,
        "Unable to load rocm server library: %s (If you have a Radeon card, "
        "did you install the ROCM libraries?)",
        LOAD_ERR());
    return;
  }

  for (i = 0; l[i].p != NULL; i++) {
    *l[i].p = LOAD_SYMBOL(s->handle, l[i].s);
    if (!l[i].p) {
      UNLOAD_LIBRARY(s->handle);
      err->id = -1;
      snprintf(err->msg, err->msg_len, "symbol lookup for %s failed: %s",
               l[i].s, LOAD_ERR());
      return;
    }
  }
}

inline void rocm_shim_llama_server_init(struct rocm_llama_server s,
                                        ext_server_params_t *sparams,
                                        ext_server_resp_t *err) {
  s.llama_server_init(sparams, err);
}

inline void rocm_shim_llama_server_start(struct rocm_llama_server s) {
  s.llama_server_start();
}

inline void rocm_shim_llama_server_stop(struct rocm_llama_server s) {
  s.llama_server_stop();
}

inline void rocm_shim_llama_server_completion(struct rocm_llama_server s,
                                              const char *json_req,
                                              ext_server_resp_t *resp) {
  s.llama_server_completion(json_req, resp);
}

inline void rocm_shim_llama_server_completion_next_result(
    struct rocm_llama_server s, const int task_id,
    ext_server_task_result_t *result) {
  s.llama_server_completion_next_result(task_id, result);
}

inline void rocm_shim_llama_server_completion_cancel(struct rocm_llama_server s,
                                                     const int task_id,
                                                     ext_server_resp_t *err) {
  s.llama_server_completion_cancel(task_id, err);
}
inline void rocm_shim_llama_server_release_task_result(
    struct rocm_llama_server s, ext_server_task_result_t *result) {
  s.llama_server_release_task_result(result);
}

inline void rocm_shim_llama_server_tokenize(struct rocm_llama_server s,
                                            const char *json_req,
                                            char **json_resp,
                                            ext_server_resp_t *err) {
  s.llama_server_tokenize(json_req, json_resp, err);
}

inline void rocm_shim_llama_server_detokenize(struct rocm_llama_server s,
                                              const char *json_req,
                                              char **json_resp,
                                              ext_server_resp_t *err) {
  s.llama_server_detokenize(json_req, json_resp, err);
}

inline void rocm_shim_llama_server_embedding(struct rocm_llama_server s,
                                             const char *json_req,
                                             char **json_resp,
                                             ext_server_resp_t *err) {
  s.llama_server_embedding(json_req, json_resp, err);
}

inline void rocm_shim_llama_server_release_json_resp(struct rocm_llama_server s,
                                                     char **json_resp) {
  s.llama_server_release_json_resp(json_resp);
}
