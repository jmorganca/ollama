import random
import gradio as gr
from langchain.llms import Ollama

ollama = Ollama(base_url='http://localhost:11434', model="llama2")




def random_response(message, history):
    return ollama(message)

demo = gr.ChatInterface(random_response, title="Echo Bot")

if __name__ == "__main__":
    demo.launch()



