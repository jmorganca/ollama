# Renaming Files with AI

![airenamer 2024-01-05 09_09_08](https://github.com/jmorganca/ollama/assets/633681/b98df1c8-61a7-4dff-aeb7-b04e034dced0)

This example applies the benefits of the llava models to managing images. It will find any images in your current directory, generate keywords for the image, and then copy the file to a new name based on the keywords.

## Running the example

1. Clone this repo and navigate to the `examples/typescript-airenamer` directory.
2. Install the dependencies with `npm install`.
3. Run `npm run start`.

## Review the Code

The main part of the code is in the `getkeywords` function. It calls the `/api/generate` endpoint passing in the body: 

```json
{
    "model": "llava:13b-v1.5-q5_K_M",
    "prompt": `Describe the image as a collection of keywords. Output in JSON format. Use the following schema: { filename: string, keywords: string[] }`,
    "format": "json",
    "images": [image],
    "stream": false
  }
```

This demonstrates how to use images as well as `format: json` to allow calling another function. The images key takes an array of base64 encoded images. And `format: json` tells the model to output JSON instead of regular text. When using `format: json`, it's important to also say that you expect the output to be JSON in the prompt. Adding the expected schema to the prompt also helps the model understand what you're looking for.

The `main` function calls getkeywords passing it the base64 encoded image. Then it parses the JSON output, formats the keywords into a string, and copies the file to the new name.
