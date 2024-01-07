import fs from 'fs';

export async function getkeywords(image: string): Promise<string[]> {
  const body = {
    "model": "llava:13b-v1.5-q5_K_M",
    "prompt": `Describe the image as a collection of keywords. Output in JSON format. Use the following schema: { filename: string, keywords: string[] }`,
    "format": "json",
    "images": [image],
    "stream": false
  };

  const response = await fetch("http://localhost:11434/api/generate", {
    method: "POST",
    headers: {
      "Content-Type": "application/json",
    },
    body: JSON.stringify(body),
  });

  const json = await response.json();
  const keywords = JSON.parse(json.response);

  return keywords?.keywords || [];
}

async function main() {
  for (const file of fs.readdirSync(".")) {
    if (file.endsWith(".jpg") || file.endsWith(".png")) {
      const currentpath = __dirname;
      const b64 = fs.readFileSync(`${currentpath}/${file}`, { encoding: 'base64' });
      const keywords = await getkeywords(b64.toString());
      const fileparts = keywords.map(k => k.replace(/ /g, "_"));
      const fileext = file.split(".").pop();
      const newfilename = fileparts.join("-") + "." + fileext;
      fs.copyFileSync(`${currentpath}/${file}`, `${currentpath}/${newfilename}`);
      console.log(`Copied ${file} to ${newfilename}`);
    }
  }

}

main();