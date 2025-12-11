from openai import OpenAI
import streamlit as st

api_key = st.secrets["open_router_key"]

client = OpenAI(
  base_url="https://openrouter.ai/api/v1",
  api_key=api_key
)

model="google/gemma-3-27b-it:free"

def get_synonyms(word): 

    completion = client.chat.completions.create(
      model=model
      messages=[
        {
          "role": "user",
          "content": f"What synonyms are there for {word}? Respond in a python array. Only return the array, nothing else. No explanations. No backticks."
        }
      ]
    )

    return completion.choices[0].message.content
  
def parse_sentence(sentence):
    completion = client.chat.completions.create(
      model=model,
      messages=[
        {
          "role": "user",
          "content": f"Validate {sentence} for grammatical and semantic correctness.\
          Respond with 'Correct' if correct. \
          Respond with 'Incorrect' if incorrect with a very brief explanation of why, no more than 10 words. \
          Only return 'Correct' or 'Incorrect: <brief explanation>', nothing else. \
"
        }
      ]
    )
    return completion.choices[0].message.content
