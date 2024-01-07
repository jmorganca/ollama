import curses
import json
from utils import get_url_for_topic, topic_urls, menu, getUrls, get_summary, getArticleText, knn_search
import requests
from sentence_transformers import SentenceTransformer
from mattsollamatools import chunker

if __name__ == "__main__":
    chosen_topic = curses.wrapper(menu)
    urls = getUrls(chosen_topic, n=5)
    model = SentenceTransformer('all-MiniLM-L6-v2')
    allEmbeddings = []
    failed = False

    for url in urls:
      article={}
      article['embeddings'] = []
      article['url'] = url
      text = getArticleText(url)
      summary = get_summary(text)
      if summary is None:
        failed = True
        print(f"Failed to get summary for {url}")
        break  # Exit the loop if we failed to get a summary

      chunks = chunker(text)  # Use the chunk_text function from web_utils
      embeddings = model.encode(chunks)
      for (chunk, embedding) in zip(chunks, embeddings):
        item = {}
        item['source'] = chunk
        item['embedding'] = embedding.tolist()  # Convert NumPy array to list
        item['sourcelength'] = len(chunk)
        article['embeddings'].append(item)
    
      allEmbeddings.append(article)

      print(f"{summary}\n")

    if failed:
       exit(1)

    print("Here is your news summary:\n")

    while True:
      context = []
      # Input a question from the user
      question = input("Enter your question about the news, or type quit: ")

      if question.lower() == 'quit':
        break

      # Embed the user's question
      question_embedding = model.encode([question])

      # Perform KNN search to find the best matches (indices and source text)
      best_matches = knn_search(question_embedding, allEmbeddings, k=10)


      sourcetext=""
      for i, (index, source_text) in enumerate(best_matches, start=1):
          sourcetext += f"{i}. Index: {index}, Source Text: {source_text}"

      systemPrompt = f"Only use the following information to answer the question. Do not use anything else: {sourcetext}"

      url = "http://localhost:11434/api/generate"

      payload = {
      "model": "mistral-openorca",
      "prompt": question, 
      "system": systemPrompt,
      "stream": False, 
      "context": context
      }

      # Convert the payload to a JSON string
      payload_json = json.dumps(payload)

      # Set the headers to specify JSON content
      headers = {
          "Content-Type": "application/json"
      }

      # Send the POST request
      response = requests.post(url, data=payload_json, headers=headers)

      # Check the response
      if response.status_code == 200:
          output = json.loads(response.text)
          context = output['context']
          print(output['response']+ "\n")
          

      else:
          print(f"Request failed with status code {response.status_code}")

