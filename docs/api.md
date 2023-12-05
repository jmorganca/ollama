# API

## Endpoints

- [Generate a completion](#generate-a-completion)
- [Chat Completions](#chat-completions)
- [Create a Model](#create-a-model)
- [List Local Models](#list-local-models)
- [Show Model Information](#show-model-information)
- [Copy a Model](#copy-a-model)
- [Delete a Model](#delete-a-model)
- [Pull a Model](#pull-a-model)
- [Push a Model](#push-a-model)
- [Generate Embeddings](#generate-embeddings)

## Conventions

### Model names

Model names follow a `model:tag` format. Some examples are `orca-mini:3b-q4_1` and `llama2:70b`. The tag is optional and, if not provided, will default to `latest`. The tag is used to identify a specific version.

### Durations

All durations are returned in nanoseconds.

### Streaming responses

Certain endpoints stream responses as JSON objects delineated with the newline (`\n`) character.

## Generate a completion

```shell
POST /api/generate
```

Generate a response for a given prompt with a provided model. This is a streaming endpoint, so will be a series of responses. The final response object will include statistics and additional data from the request.

### Parameters

- `model`: (required) the [model name](#model-names)
- `prompt`: the prompt to generate a response for

Advanced parameters (optional):

- `format`: the format to return a response in. Currently the only accepted value is `json`
- `options`: additional model parameters listed in the documentation for the [Modelfile](./modelfile.md#valid-parameters-and-values) such as `temperature`
- `system`: system prompt to (overrides what is defined in the `Modelfile`)
- `template`: the full prompt or prompt template (overrides what is defined in the `Modelfile`)
- `context`: the context parameter returned from a previous request to `/generate`, this can be used to keep a short conversational memory
- `stream`: if `false` the response will be returned as a single response object, rather than a stream of objects
- `raw`: if `true` no formatting will be applied to the prompt and no context will be returned. You may choose to use the `raw` parameter if you are specifying a full templated prompt in your request to the API, and are managing history yourself.

### JSON mode

Enable JSON mode by setting the `format` parameter to `json`. This will structure the response as valid JSON. See the JSON mode [example](#request-json-mode) below.

> Note: it's important to instruct the model to use JSON in the `prompt`. Otherwise, the model may generate large amounts whitespace.

### Examples

#### Request

```shell
curl http://localhost:11434/api/generate -d '{
  "model": "llama2",
  "prompt": "Why is the sky blue?"
}'
```

#### Response

A stream of JSON objects is returned:

```json
{
  "model": "llama2",
  "created_at": "2023-08-04T08:52:19.385406455-07:00",
  "response": "The",
  "done": false
}
```

The final response in the stream also includes additional data about the generation:

- `total_duration`: time spent generating the response
- `load_duration`: time spent in nanoseconds loading the model
- `sample_count`: number of samples generated
- `sample_duration`: time spent generating samples
- `prompt_eval_count`: number of tokens in the prompt
- `prompt_eval_duration`: time spent in nanoseconds evaluating the prompt
- `eval_count`: number of tokens the response
- `eval_duration`: time in nanoseconds spent generating the response
- `context`: an encoding of the conversation used in this response, this can be sent in the next request to keep a conversational memory
- `response`: empty if the response was streamed, if not streamed, this will contain the full response

To calculate how fast the response is generated in tokens per second (token/s), divide `eval_count` / `eval_duration`.

```json
{
  "model": "llama2",
  "created_at": "2023-08-04T19:22:45.499127Z",
  "response": "",
  "context": [1, 2, 3],
  "done": true,
  "total_duration": 5589157167,
  "load_duration": 3013701500,
  "sample_count": 114,
  "sample_duration": 81442000,
  "prompt_eval_count": 46,
  "prompt_eval_duration": 1160282000,
  "eval_count": 113,
  "eval_duration": 1325948000
}
```

#### Request (No streaming)

```shell
curl http://localhost:11434/api/generate -d '{
  "model": "llama2",
  "prompt": "Why is the sky blue?",
  "stream": false
}'
```

#### Response

If `stream` is set to `false`, the response will be a single JSON object:

```json
{
  "model": "llama2",
  "created_at": "2023-08-04T19:22:45.499127Z",
  "response": "The sky is blue because it is the color of the sky.",
  "context": [1, 2, 3],
  "done": true,
  "total_duration": 5589157167,
  "load_duration": 3013701500,
  "sample_count": 114,
  "sample_duration": 81442000,
  "prompt_eval_count": 46,
  "prompt_eval_duration": 1160282000,
  "eval_count": 13,
  "eval_duration": 1325948000
}
```

#### Request (Raw mode)

In some cases you may wish to bypass the templating system and provide a full prompt. In this case, you can use the `raw` parameter to disable formatting and context.

```shell
curl http://localhost:11434/api/generate -d '{
  "model": "mistral",
  "prompt": "[INST] why is the sky blue? [/INST]",
  "raw": true,
  "stream": false
}'
```

#### Response

```json
{
  "model": "mistral",
  "created_at": "2023-11-03T15:36:02.583064Z",
  "response": " The sky appears blue because of a phenomenon called Rayleigh scattering.",
  "done": true,
  "total_duration": 14648695333,
  "load_duration": 3302671417,
  "prompt_eval_count": 14,
  "prompt_eval_duration": 286243000,
  "eval_count": 129,
  "eval_duration": 10931424000
}
```

#### Request (JSON mode)

```shell
curl http://localhost:11434/api/generate -d '{
  "model": "llama2",
  "prompt": "What color is the sky at different times of the day? Respond using JSON",
  "format": "json",
  "stream": false
}'
```

#### Response

```json
{
  "model": "llama2",
  "created_at": "2023-11-09T21:07:55.186497Z",
  "response": "{\n\"morning\": {\n\"color\": \"blue\"\n},\n\"noon\": {\n\"color\": \"blue-gray\"\n},\n\"afternoon\": {\n\"color\": \"warm gray\"\n},\n\"evening\": {\n\"color\": \"orange\"\n}\n}\n",
  "done": true,
  "total_duration": 4661289125,
  "load_duration": 1714434500,
  "prompt_eval_count": 36,
  "prompt_eval_duration": 264132000,
  "eval_count": 75,
  "eval_duration": 2112149000
}
```

The value of `response` will be a string containing JSON similar to:

```json
{
  "morning": {
    "color": "blue"
  },
  "noon": {
    "color": "blue-gray"
  },
  "afternoon": {
    "color": "warm gray"
  },
  "evening": {
    "color": "orange"
  }
}
```

#### Request (With options)

If you want to set custom options for the model at runtime rather than in the Modelfile, you can do so with the `options` parameter. This example sets every available option, but you can set any of them individually and omit the ones you do not want to override.

```shell
curl http://localhost:11434/api/generate -d '{
  "model": "llama2",
  "prompt": "Why is the sky blue?",
  "stream": false,
  "options": {
    "num_keep": 5,
    "seed": 42,
    "num_predict": 100,
    "top_k": 20,
    "top_p": 0.9,
    "tfs_z": 0.5,
    "typical_p": 0.7,
    "repeat_last_n": 33,
    "temperature": 0.8,
    "repeat_penalty": 1.2,
    "presence_penalty": 1.5,
    "frequency_penalty": 1.0,
    "mirostat": 1,
    "mirostat_tau": 0.8,
    "mirostat_eta": 0.6,
    "penalize_newline": true,
    "stop": ["\n", "user:"],
    "numa": false,
    "num_ctx": 4,
    "num_batch": 2,
    "num_gqa": 1,
    "num_gpu": 1,
    "main_gpu": 0,
    "low_vram": false,
    "f16_kv": true,
    "logits_all": false,
    "vocab_only": false,
    "use_mmap": true,
    "use_mlock": false,
    "embedding_only": false,
    "rope_frequency_base": 1.1,
    "rope_frequency_scale": 0.8,
    "num_thread": 8
    }
}'
```

#### Response

```json
{
  "model": "llama2",
  "created_at": "2023-08-04T19:22:45.499127Z",
  "response": "The sky is blue because it is the color of the sky.",
  "context": [1, 2, 3],
  "done": true,
  "total_duration": 5589157167,
  "load_duration": 3013701500,
  "sample_count": 114,
  "sample_duration": 81442000,
  "prompt_eval_count": 46,
  "prompt_eval_duration": 1160282000,
  "eval_count": 13,
  "eval_duration": 1325948000
}
```

## Chat Completions

Refrence [OpenAI Chat Completions](https://platform.openai.com/docs/api-reference/chat/create)


```shell
POST /api/chat/completions
```

### Request

```json
{
  "model": "mistral:latest",
  "stream": false,
  "messages": [
    {
      "role": "system",
      "content": "You are a helpful assistant."
    },
    {
      "role": "user",
      "content": "Why is the sky blue?"
    }
  ]
}
```

### Response

```json
{
  "id": "chatcmpl-598319",
  "object": "chat.completion",
  "created": 1701357062,
  "model": "mistral:latest",
  "choices": [
    {
      "index": 0,
      "message": {
        "role": "assistant",
        "content": "\nThe sky appears blue because of a phenomenon known as Rayleigh scattering, which is caused by the interaction between light and Earth's atmosphere. When sunlight enters the atmosphere, it collides with gas molecules and small particles, causing the shorter wavelengths of light (blue) to be scattered in every direction more frequently than the longer wavelengths (red). This scattering effect causes the blue light to be more visible from all directions, giving the sky its characteristic color."
      },
      "finish_reason": "stop"
    }
  ],
  "usage": { "prompt_tokens": 0, "completion_tokens": 0, "total_tokens": 0 }
}

```

## Streaming Response

```shell
{"id":"chatcmpl-855382","object":"chat.completion","created":1701355821,"model":"mistral:latest","choices":[{"index":0,"message":{"role":"assistant","content":"Hello"},"finish_reason":"stop"}],"usage":{"prompt_tokens":0,"completion_tokens":0,"total_tokens":0}}
{"id":"chatcmpl-388597","object":"chat.completion","created":1701355821,"model":"mistral:latest","choices":[{"index":0,"message":{"role":"assistant","content":"!"},"finish_reason":"stop"}],"usage":{"prompt_tokens":0,"completion_tokens":0,"total_tokens":0}}
{"id":"chatcmpl-995697","object":"chat.completion","created":1701355821,"model":"mistral:latest","choices":[{"index":0,"message":{"role":"assistant","content":" How"},"finish_reason":"stop"}],"usage":{"prompt_tokens":0,"completion_tokens":0,"total_tokens":0}}
{"id":"chatcmpl-422675","object":"chat.completion","created":1701355821,"model":"mistral:latest","choices":[{"index":0,"message":{"role":"assistant","content":" can"},"finish_reason":"stop"}],"usage":{"prompt_tokens":0,"completion_tokens":0,"total_tokens":0}}
{"id":"chatcmpl-722314","object":"chat.completion","created":1701355821,"model":"mistral:latest","choices":[{"index":0,"message":{"role":"assistant","content":" I"},"finish_reason":"stop"}],"usage":{"prompt_tokens":0,"completion_tokens":0,"total_tokens":0}}
{"id":"chatcmpl-782491","object":"chat.completion","created":1701355821,"model":"mistral:latest","choices":[{"index":0,"message":{"role":"assistant","content":" help"},"finish_reason":"stop"}],"usage":{"prompt_tokens":0,"completion_tokens":0,"total_tokens":0}}
{"id":"chatcmpl-76807","object":"chat.completion","created":1701355821,"model":"mistral:latest","choices":[{"index":0,"message":{"role":"assistant","content":" you"},"finish_reason":"stop"}],"usage":{"prompt_tokens":0,"completion_tokens":0,"total_tokens":0}}
{"id":"chatcmpl-687363","object":"chat.completion","created":1701355821,"model":"mistral:latest","choices":[{"index":0,"message":{"role":"assistant","content":" today"},"finish_reason":"stop"}],"usage":{"prompt_tokens":0,"completion_tokens":0,"total_tokens":0}}
{"id":"chatcmpl-462789","object":"chat.completion","created":1701355821,"model":"mistral:latest","choices":[{"index":0,"message":{"role":"assistant","content":"?"},"finish_reason":"stop"}],"usage":{"prompt_tokens":0,"completion_tokens":0,"total_tokens":0}}
{"id":"chatcmpl-433223","object":"chat.completion","created":1701355821,"model":"mistral:latest","choices":[{"index":0,"message":{"role":"assistant","content":""},"finish_reason":"stop"}],"usage":{"prompt_tokens":0,"completion_tokens":0,"total_tokens":0}}
```

## Create a Model

```shell
POST /api/create
```

Create a model from a [`Modelfile`](./modelfile.md). It is recommended to set `modelfile` to the content of the Modelfile rather than just set `path`. This is a requirement for remote create. Remote model creation should also create any file blobs, fields such as `FROM` and `ADAPTER`, explicitly with the server using [Create a Blob](#create-a-blob) and the value to the path indicated in the response.

### Parameters

- `name`: name of the model to create
- `modelfile` (optional): contents of the Modelfile
- `stream`: (optional) if `false` the response will be returned as a single response object, rather than a stream of objects
- `path` (optional): path to the Modelfile

### Examples

#### Request

```shell
curl http://localhost:11434/api/create -d '{
  "name": "mario",
  "modelfile": "FROM llama2\nSYSTEM You are mario from Super Mario Bros."
}'
```

#### Response

A stream of JSON objects. When finished, `status` is `success`.

```json
{
  "status": "parsing modelfile"
}
```

### Check if a Blob Exists

```shell
HEAD /api/blobs/:digest
```

Check if a blob is known to the server.

#### Query Parameters

- `digest`: the SHA256 digest of the blob

#### Examples

##### Request

```shell
curl -I http://localhost:11434/api/blobs/sha256:29fdb92e57cf0827ded04ae6461b5931d01fa595843f55d36f5b275a52087dd2
```

##### Response

Return 200 OK if the blob exists, 404 Not Found if it does not.

### Create a Blob

```shell
POST /api/blobs/:digest
```

Create a blob from a file. Returns the server file path.

#### Query Parameters

- `digest`: the expected SHA256 digest of the file

#### Examples

##### Request

```shell
curl -T model.bin -X POST http://localhost:11434/api/blobs/sha256:29fdb92e57cf0827ded04ae6461b5931d01fa595843f55d36f5b275a52087dd2
```

##### Response

Return 201 Created if the blob was successfully created.

## List Local Models

```shell
GET /api/tags
```

List models that are available locally.

### Examples

#### Request

```shell
curl http://localhost:11434/api/tags
```

#### Response

A single JSON object will be returned.

```json
{
  "models": [
    {
      "name": "llama2",
      "modified_at": "2023-08-02T17:02:23.713454393-07:00",
      "size": 3791730596
    },
    {
      "name": "llama2:13b",
      "modified_at": "2023-08-08T12:08:38.093596297-07:00",
      "size": 7323310500
    }
  ]
}
```

## Show Model Information

```shell
POST /api/show
```

Show details about a model including modelfile, template, parameters, license, and system prompt.

### Parameters

- `name`: name of the model to show

### Examples

#### Request

```shell
curl http://localhost:11434/api/show -d '{
  "name": "llama2"
}'
```

#### Response

```json
{
  "license": "<contents of license block>",
  "modelfile": "# Modelfile generated by \"ollama show\"\n# To build a new Modelfile based on this one, replace the FROM line with:\n# FROM llama2:latest\n\nFROM /Users/username/.ollama/models/blobs/sha256:8daa9615cce30c259a9555b1cc250d461d1bc69980a274b44d7eda0be78076d8\nTEMPLATE \"\"\"[INST] {{ if and .First .System }}<<SYS>>{{ .System }}<</SYS>>\n\n{{ end }}{{ .Prompt }} [/INST] \"\"\"\nSYSTEM \"\"\"\"\"\"\nPARAMETER stop [INST]\nPARAMETER stop [/INST]\nPARAMETER stop <<SYS>>\nPARAMETER stop <</SYS>>\n",
  "parameters": "stop                           [INST]\nstop                           [/INST]\nstop                           <<SYS>>\nstop                           <</SYS>>",
  "template": "[INST] {{ if and .First .System }}<<SYS>>{{ .System }}<</SYS>>\n\n{{ end }}{{ .Prompt }} [/INST] "
}
```

## Copy a Model

```shell
POST /api/copy
```

Copy a model. Creates a model with another name from an existing model.

### Examples

#### Request

```shell
curl http://localhost:11434/api/copy -d '{
  "source": "llama2",
  "destination": "llama2-backup"
}'
```

#### Response

The only response is a 200 OK if successful.

## Delete a Model

```shell
DELETE /api/delete
```

Delete a model and its data.

### Parameters

- `name`: model name to delete

### Examples

#### Request

```shell
curl -X DELETE http://localhost:11434/api/delete -d '{
  "name": "llama2:13b"
}'
```

#### Response

If successful, the only response is a 200 OK.

## Pull a Model

```shell
POST /api/pull
```

Download a model from the ollama library. Cancelled pulls are resumed from where they left off, and multiple calls will share the same download progress.

### Parameters

- `name`: name of the model to pull
- `insecure`: (optional) allow insecure connections to the library. Only use this if you are pulling from your own library during development.
- `stream`: (optional) if `false` the response will be returned as a single response object, rather than a stream of objects

### Examples

#### Request

```shell
curl http://localhost:11434/api/pull -d '{
  "name": "llama2"
}'
```

#### Response

If `stream` is not specified, or set to `true`, a stream of JSON objects is returned:

The first object is the manifest:

```json
{
  "status": "pulling manifest"
}
```

Then there is a series of downloading responses. Until any of the download is completed, the `completed` key may not be included. The number of files to be downloaded depends on the number of layers specified in the manifest.

```json
{
  "status": "downloading digestname",
  "digest": "digestname",
  "total": 2142590208,
  "completed": 241970
}
```

After all the files are downloaded, the final responses are:

```json
{
    "status": "verifying sha256 digest"
}
{
    "status": "writing manifest"
}
{
    "status": "removing any unused layers"
}
{
    "status": "success"
}
```

if `stream` is set to false, then the response is a single JSON object:

```json
{
  "status": "success"
}
```

## Push a Model

```shell
POST /api/push
```

Upload a model to a model library. Requires registering for ollama.ai and adding a public key first.

### Parameters

- `name`: name of the model to push in the form of `<namespace>/<model>:<tag>`
- `insecure`: (optional) allow insecure connections to the library. Only use this if you are pushing to your library during development.
- `stream`: (optional) if `false` the response will be returned as a single response object, rather than a stream of objects

### Examples

#### Request

```shell
curl http://localhost:11434/api/push -d '{
  "name": "mattw/pygmalion:latest"
}'
```

#### Response

If `stream` is not specified, or set to `true`, a stream of JSON objects is returned:

```json
{ "status": "retrieving manifest" }
```

and then:

```json
{
  "status": "starting upload",
  "digest": "sha256:bc07c81de745696fdf5afca05e065818a8149fb0c77266fb584d9b2cba3711ab",
  "total": 1928429856
}
```

Then there is a series of uploading responses:

```json
{
  "status": "starting upload",
  "digest": "sha256:bc07c81de745696fdf5afca05e065818a8149fb0c77266fb584d9b2cba3711ab",
  "total": 1928429856
}
```

Finally, when the upload is complete:

```json
{"status":"pushing manifest"}
{"status":"success"}
```

If `stream` is set to `false`, then the response is a single JSON object:

```json
{ "status": "success" }
```

## Generate Embeddings

```shell
POST /api/embeddings
```

Generate embeddings from a model

### Parameters

- `model`: name of model to generate embeddings from
- `prompt`: text to generate embeddings for

Advanced parameters:

- `options`: additional model parameters listed in the documentation for the [Modelfile](./modelfile.md#valid-parameters-and-values) such as `temperature`

### Examples

#### Request

```shell
curl http://localhost:11434/api/embeddings -d '{
  "model": "llama2",
  "prompt": "Here is an article about llamas..."
}'
```

#### Response

```json
{
  "embedding": [
    0.5670403838157654, 0.009260174818336964, 0.23178744316101074, -0.2916173040866852, -0.8924556970596313,
    0.8785552978515625, -0.34576427936553955, 0.5742510557174683, -0.04222835972905159, -0.137906014919281
  ]
}
```
