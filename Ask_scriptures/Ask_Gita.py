from openai import OpenAI
from sentence_transformers import SentenceTransformer
import faiss
import numpy as np
import json
import streamlit as st

client = OpenAI(
    api_key="mdb_0JXfDNMjdfKy4BB3mxQWQwr9Ol2jvwSIYfduoeMAiFa1",
    base_url='https://llm.mdb.ai/'
)


def get_data(query):
    index = faiss.read_index("gita_faiss.index")
    with open("gita_chunks.json", "r", encoding="utf-8") as f:
        chunk_data = json.load(f)

    model = SentenceTransformer("all-MiniLM-L6-v2")

    query_vector = model.encode([query])
    D, I = index.search(np.array(query_vector), k=4)

    context = "\n".join([chunk_data[i] for i in I[0]])


    formatted = f"""You are an AI spiritual assistant trained on Bhagavad Gita.
Based on the following Gita verses, answer the question with meaning
from given gita Context only.do not hallucinate
contect:
{context}

Question: {query}
Answer:"""

    completion = client.chat.completions.create(
        model="gpt-3.5-turbo",
        messages=[
            {'role': 'user', 'content': formatted}
        ],
        stream=False
    )

    return completion.choices[0].message.content

while True:
  query = input("Ask question: ")
  answer=get_data(query)
  print("\n🕉️ Gita Answer:\n", answer)
  print("\n")
