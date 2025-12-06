from openai import OpenAI
import streamlit as st

api_key = st.secrets["open_router_key"]

def get_synonyms(word):
    client = OpenAI(
      base_url="https://openrouter.ai/api/v1",
      api_key=api_key
    )

    completion = client.chat.completions.create(
      model="google/gemma-3-27b-it:free",
      messages=[
        {
          "role": "user",
          "content": f"What synonyms are there for {word}? Respond in a python array. Only return the array, nothing else. No explanations. No backticks."
        }
      ]
    )

    return completion.choices[0].message.content