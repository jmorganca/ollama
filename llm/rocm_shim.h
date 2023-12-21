#include <stdlib.h>

#include "server.h"

#ifdef __cplusplus
extern "C" {
#endif
struct rocm_llama_server {
  void *handle;
  void (*llama_server_init)(ext_server_params_t *sparams,
                            ext_server_resp_t *err);
  void (*llama_server_start)();
  void (*llama_server_stop)();
  void (*llama_server_completion)(const char *json_req,
                                  ext_server_resp_t *resp);
  void (*llama_server_completion_next_result)(const int task_id,
                                              ext_server_task_result_t *result);
  void (*llama_server_completion_cancel)(const int task_id,
                                         ext_server_resp_t *err);
  void (*llama_server_release_task_result)(ext_server_task_result_t *result);
  void (*llama_server_tokenize)(const char *json_req, char **json_resp,
                                ext_server_resp_t *err);
  void (*llama_server_detokenize)(const char *json_req, char **json_resp,
                                  ext_server_resp_t *err);
  void (*llama_server_embedding)(const char *json_req, char **json_resp,
                                 ext_server_resp_t *err);
  void (*llama_server_release_json_resp)(char **json_resp);
};

void rocm_shim_init(const char *libPath, struct rocm_llama_server *s,
                    ext_server_resp_t *err);

// No good way to call C function pointers from Go so inline the indirection
void rocm_shim_llama_server_init(struct rocm_llama_server s,
                                 ext_server_params_t *sparams,
                                 ext_server_resp_t *err);

void rocm_shim_llama_server_start(struct rocm_llama_server s);

void rocm_shim_llama_server_stop(struct rocm_llama_server s);

void rocm_shim_llama_server_completion(struct rocm_llama_server s,
                                       const char *json_req,
                                       ext_server_resp_t *resp);

void rocm_shim_llama_server_completion_next_result(
    struct rocm_llama_server s, const int task_id,
    ext_server_task_result_t *result);

void rocm_shim_llama_server_completion_cancel(struct rocm_llama_server s,
                                              const int task_id,
                                              ext_server_resp_t *err);

void rocm_shim_llama_server_release_task_result(
    struct rocm_llama_server s, ext_server_task_result_t *result);

void rocm_shim_llama_server_tokenize(struct rocm_llama_server s,
                                     const char *json_req, char **json_resp,
                                     ext_server_resp_t *err);

void rocm_shim_llama_server_detokenize(struct rocm_llama_server s,
                                       const char *json_req, char **json_resp,
                                       ext_server_resp_t *err);

void rocm_shim_llama_server_embedding(struct rocm_llama_server s,
                                      const char *json_req, char **json_resp,
                                      ext_server_resp_t *err);
void rocm_shim_llama_server_release_json_resp(struct rocm_llama_server s,
                                              char **json_resp);

#ifdef __cplusplus
}
#endif