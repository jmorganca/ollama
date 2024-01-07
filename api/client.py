import os
import json
import requests
import os
import hashlib
import json
from pathlib import Path

BASE_URL = os.environ.get('OLLAMA_HOST', 'http://localhost:11434')

# Generate a response for a given prompt with a provided model. This is a streaming endpoint, so will be a series of responses.
# The final response object will include statistics and additional data from the request. Use the callback function to override
# the default handler.
def generate(model_name, prompt, system=None, template=None, format="", context=None, options=None, callback=None):
    try:
        url = f"{BASE_URL}/api/generate"
        payload = {
            "model": model_name, 
            "prompt": prompt, 
            "system": system, 
            "template": template, 
            "context": context, 
            "options": options,
            "format": format,
        }
        
        # Remove keys with None values
        payload = {k: v for k, v in payload.items() if v is not None}
        
        with requests.post(url, json=payload, stream=True) as response:
            response.raise_for_status()
            
            # Creating a variable to hold the context history of the final chunk
            final_context = None
            
            # Variable to hold concatenated response strings if no callback is provided
            full_response = ""

            # Iterating over the response line by line and displaying the details
            for line in response.iter_lines():
                if line:
                    # Parsing each line (JSON chunk) and extracting the details
                    chunk = json.loads(line)
                    
                    # If a callback function is provided, call it with the chunk
                    if callback:
                        callback(chunk)
                    else:
                        # If this is not the last chunk, add the "response" field value to full_response and print it
                        if not chunk.get("done"):
                            response_piece = chunk.get("response", "")
                            full_response += response_piece
                            print(response_piece, end="", flush=True)
                    
                    # Check if it's the last chunk (done is true)
                    if chunk.get("done"):
                        final_context = chunk.get("context")
            
            # Return the full response and the final context
            return full_response, final_context
    except requests.exceptions.RequestException as e:
        print(f"An error occurred: {e}")
        return None, None
    

# Create a blob file on the server if it doesn't exist.
def create_blob(digest, file_path):
    url = f"{BASE_URL}/api/blobs/{digest}"

    # Check if the blob exists
    response = requests.head(url)
    if response.status_code != 404:
        return  # Blob already exists, no need to upload
    response.raise_for_status()

    # Upload the blob
    with open(file_path, 'rb') as file_data:
        requests.post(url, data=file_data)


# Create a model from a Modelfile. Use the callback function to override the default handler.
def create(model_name, filename, callback=None):
    try:
        file_path = Path(filename).expanduser().resolve()
        processed_lines = []

        # Read and process the modelfile
        with open(file_path, 'r') as f:
            for line in f:            
                # Skip empty or whitespace-only lines
                if not line.strip():
                    continue
            
                command, args = line.split(maxsplit=1)

                if command.upper() in ["FROM", "ADAPTER"]:
                    path = Path(args.strip()).expanduser()

                    # Check if path is relative and resolve it
                    if not path.is_absolute():
                        path = (file_path.parent / path)

                    # Skip if file does not exist for "model", this is handled by the server
                    if not path.exists():
                        processed_lines.append(line)
                        continue

                    # Calculate SHA-256 hash
                    with open(path, 'rb') as bin_file:
                        hash = hashlib.sha256()
                        hash.update(bin_file.read())
                        blob = f"sha256:{hash.hexdigest()}"
                
                    # Add the file to the remote server
                    create_blob(blob, path)

                    # Replace path with digest in the line
                    line = f"{command} @{blob}\n"

                processed_lines.append(line)

        # Combine processed lines back into a single string
        modelfile_content = '\n'.join(processed_lines)

        url = f"{BASE_URL}/api/create"
        payload = {"name": model_name, "modelfile": modelfile_content}

        # Making a POST request with the stream parameter set to True to handle streaming responses
        with requests.post(url, json=payload, stream=True) as response:
            response.raise_for_status()
            # Iterating over the response line by line and displaying the status
            for line in response.iter_lines():
                if line:
                    chunk = json.loads(line)
                    if callback:
                        callback(chunk)
                    else:
                        print(f"Status: {chunk.get('status')}")

    except Exception as e:
        print(f"An error occurred: {e}")


# Pull a model from a the model registry. Cancelled pulls are resumed from where they left off, and multiple
# calls to will share the same download progress. Use the callback function to override the default handler.
def pull(model_name, insecure=False, callback=None):
    try:
        url = f"{BASE_URL}/api/pull"
        payload = {
            "name": model_name,
            "insecure": insecure
        }

        # Making a POST request with the stream parameter set to True to handle streaming responses
        with requests.post(url, json=payload, stream=True) as response:
            response.raise_for_status()

            # Iterating over the response line by line and displaying the details
            for line in response.iter_lines():
                if line:
                    # Parsing each line (JSON chunk) and extracting the details
                    chunk = json.loads(line)

                    # If a callback function is provided, call it with the chunk
                    if callback:
                        callback(chunk)
                    else:
                        # Print the status message directly to the console
                        print(chunk.get('status', ''), end='', flush=True)
                    
                    # If there's layer data, you might also want to print that (adjust as necessary)
                    if 'digest' in chunk:
                        print(f" - Digest: {chunk['digest']}", end='', flush=True)
                        print(f" - Total: {chunk['total']}", end='', flush=True)
                        print(f" - Completed: {chunk['completed']}", end='\n', flush=True)
                    else:
                        print()
    except requests.exceptions.RequestException as e:
        print(f"An error occurred: {e}")

# Push a model to the model registry. Use the callback function to override the default handler.
def push(model_name, insecure=False, callback=None):
    try:
        url = f"{BASE_URL}/api/push"
        payload = {
            "name": model_name,
            "insecure": insecure
        }

        # Making a POST request with the stream parameter set to True to handle streaming responses
        with requests.post(url, json=payload, stream=True) as response:
            response.raise_for_status()

            # Iterating over the response line by line and displaying the details
            for line in response.iter_lines():
                if line:
                    # Parsing each line (JSON chunk) and extracting the details
                    chunk = json.loads(line)

                    # If a callback function is provided, call it with the chunk
                    if callback:
                        callback(chunk)
                    else:
                        # Print the status message directly to the console
                        print(chunk.get('status', ''), end='', flush=True)
                    
                    # If there's layer data, you might also want to print that (adjust as necessary)
                    if 'digest' in chunk:
                        print(f" - Digest: {chunk['digest']}", end='', flush=True)
                        print(f" - Total: {chunk['total']}", end='', flush=True)
                        print(f" - Completed: {chunk['completed']}", end='\n', flush=True)
                    else:
                        print()
    except requests.exceptions.RequestException as e:
        print(f"An error occurred: {e}")

# List models that are available locally.
def list():
    try:
        response = requests.get(f"{BASE_URL}/api/tags")
        response.raise_for_status()
        data = response.json()
        models = data.get('models', [])
        return models

    except requests.exceptions.RequestException as e:
        print(f"An error occurred: {e}")
        return None

# Copy a model. Creates a model with another name from an existing model.
def copy(source, destination):
    try:
        # Create the JSON payload
        payload = {
            "source": source,
            "destination": destination
        }
        
        response = requests.post(f"{BASE_URL}/api/copy", json=payload)
        response.raise_for_status()
        
        # If the request was successful, return a message indicating that the copy was successful
        return "Copy successful"

    except requests.exceptions.RequestException as e:
        print(f"An error occurred: {e}")
        return None

# Delete a model and its data.
def delete(model_name):
    try:
        url = f"{BASE_URL}/api/delete"
        payload = {"name": model_name}
        response = requests.delete(url, json=payload)
        response.raise_for_status()
        return "Delete successful"
    except requests.exceptions.RequestException as e:
        print(f"An error occurred: {e}")
        return None

# Shows info about a model.
def show(model_name):
    try:
        url = f"{BASE_URL}/api/show"
        payload = {"name": model_name}
        response = requests.post(url, json=payload)
        response.raise_for_status()
        
        # Parse the JSON response and return it
        data = response.json()
        return data
    except requests.exceptions.RequestException as e:
        print(f"An error occurred: {e}")
        return None

# Checks whether the model is running.
def heartbeat():
    try:
        url = f"{BASE_URL}/"
        response = requests.head(url)
        response.raise_for_status()
        return "Ollama is running"
    except requests.exceptions.RequestException as e:
        print(f"An error occurred: {e}")
        return "Ollama is not running"
