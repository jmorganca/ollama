import os
import io
import json
import requests
from urllib.parse import urljoin
from pathlib import Path
from hashlib import sha256
from base64 import b64encode


def _do(method, path, stream=False, **kwargs):
  base = os.getenv('OLLAMA_HOST', 'http://127.0.0.1:11434')
  response = requests.request(method, urljoin(base, path), stream=stream, **kwargs)
  response.raise_for_status()
  return response


def _stream(response):
  for lines in response.iter_lines():
    for line in lines.splitlines():
      chunk = json.loads(line)
      if error := chunk.get('error'):
        raise Exception(error)
      yield chunk


def generate(model, prompt='', system='', template='', context=[], stream=False, raw=False, format='', images=[], options={}):
  '''
  Generate a response for a given prompt with a provided model.
  '''

  response = _do('POST', '/api/generate', stream=stream, json={
    'model': model,
    'prompt': prompt,
    'system': system,
    'template': template,
    'context': context,
    'stream': stream,
    'raw': raw,
    'images': [image for image in _encode_images(images)],
    'format': format,
    'options': options,
  })

  return _stream(response) if stream else response.json()


def _encode_images(images):
  '''
  _encode_images takes a list of images and returns a generator of base64 encoded images.
  if the image is a bytes object, it is assumed to be the raw bytes of an image.
  if the image is a string, it is assumed to be a path to a file.
  if the image is a Path object, it is assumed to be a path to a file.
  if the image is a file-like object, it is assumed to be a container to the raw bytes of an image.
  '''
  for image in images:
    if isinstance(image, bytes):
      b64 = b64encode(image)
    elif isinstance(image, str):
      with open(image, 'rb') as f:
        b64 = b64encode(f.read())
    elif isinstance(image, Path):
      with image.open('rb') as f:
        b64 = b64encode(f.read())
    elif isinstance(image, io.BytesIO):
      b64 = b64encode(image.read())
    else:
      raise Exception('images must be a list of bytes, path-like objects, or file-like objects')

    yield b64.decode('utf-8')


def chat(model, messages=[], stream=False, format='', options={}):
  '''
  Generate a response for a chat with the provided model.
  '''
  for message in messages:
    if not isinstance(message, dict):
      raise Exception('messages must be a list of dictionaries')
    if not (role := message.get('role')) or role not in ['system', 'user', 'assistant']:
      raise Exception('messages must contain a role and it must be one of "system", "user", or "assistant"')
    if not message.get('content'):
      raise Exception('messages must contain content')

  response = _do('POST', '/api/chat', stream=stream, json={
    'model': model,
    'messages': messages,
    'stream': stream,
    'format': format,
    'options': options,
  })

  return _stream(response) if stream else response.json()


def pull(model, insecure, stream=False):
  '''
  Pull a model from the model registry.
  '''
  response = _do('POST', '/api/pull', json={
    'model': model,
    'insecure': insecure,
    'stream': stream,
  })

  return _stream(response) if stream else response.json()


def push(model, insecure, stream=False):
  '''
  Push a model to the model registry.
  '''
  response = _do('POST', '/api/push', json={
    'model': model,
    'insecure': insecure,
    'stream': stream,
  })

  return _stream(response) if stream else response.json()


def create(model, path='', modelfile='', stream=False):
  '''
  Create a model from a Modelfile.
  '''
  if (path := Path(path).expanduser()) and not modelfile:
    with path.open() as f:
      modelfile = _preprocess_modelfile(f, base=path.parent)
  elif isinstance(modelfile, str):
    modelfile = _preprocess_modelfile(io.StringIO(modelfile))
  else:
    modelfile = _preprocess_modelfile(modelfile)

  response = _do('POST', '/api/create', stream=stream, json={
    'name': model,
    'modelfile': modelfile,
    'stream': stream,
  })

  return _stream(response) if stream else response.json()


def _preprocess_modelfile(modelfile, base=None):
  base = Path.cwd() if base is None else base

  output = io.StringIO()
  for line in modelfile:
    command, _, args = line.partition(' ')
    if command.upper() in ['FROM', 'ADAPTER']:
      path = Path(args.strip()).expanduser().resolve()
      path = path if path.is_absolute() else base / path
      if path.exists():
        args = f'@{create_blob(path)}'

    print(command, args, file=output)

  return output.getvalue()


def create_blob(path):
  sha256sum = sha256()
  with open(path, 'rb') as r:
    while True:
      chunk = r.read(32*1024)
      if not chunk:
        break

      sha256sum.update(chunk)

  digest = f'sha256:{sha256sum.hexdigest()}'

  try:
    _do('HEAD', f'/api/blobs/{digest}')
  except Exception:
   with open(path, 'rb') as r:
     _do('POST', f'/api/blobs/{digest}', data=r)

  return digest


def list():
  '''
  List models that are available locally.
  '''
  response = _do('GET', '/api/tags')
  return response.json().get('models', [])


def copy(source, destination):
  '''
  Copy a model. Creates a model with another name from an existing model.
  '''
  response = _do('POST', '/api/copy', json={
    'source': source,
    'destination': destination,
  })

  return {'status': 'success' if response.status_code == 200 else 'error'}


def delete(model):
  '''
  Delete a model and its data.
  '''
  response = _do('DELETE', '/api/delete', json={'name': model})
  return {'status': 'success' if response.status_code == 200 else 'error'}


def show(model):
  '''
  Show info about a model.
  '''
  return _do('POST', '/api/show', json={'name': model}).json()
