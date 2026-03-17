import os
from dotenv import load_dotenv
from langchain_openrouter import ChatOpenRouter

load_dotenv()

llm = ChatOpenRouter(
    model="meta-llama/llama-3.3-70b-instruct",   # Strong free-ish model
    temperature=0.7,
    max_tokens=1024
)

response = llm.invoke("Hello! Confirm you are working correctly for our CityU SDSC4070 Product Feedback Agent project from Hong Kong using OpenRouter.")
print(response.content)