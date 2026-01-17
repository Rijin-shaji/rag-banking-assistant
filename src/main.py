import os
from groq import Groq
from retrieval.retrieval import retrieve_documents

LLM_MODEL_NAME = "llama-3.3-70b-versatile"
client = Groq(api_key=os.getenv("GROQ_API_KEY"))

def build_prompt(query, retrieved_chunks):
    context = "\n\n".join(retrieved_chunks)
    prompt = f"""
You are a banking documentation assistant.

Answer ONLY using the provided context.
If the answer is not found, say "Sorry, Not found in the provided documents."

Context:
{context}

Question:
{query}
"""
    return prompt

def call_llama_groq(prompt):
    response = client.chat.completions.create(
        model=LLM_MODEL_NAME,
        messages=[
            {"role": "system", "content": "You are a banking documentation assistant."},
            {"role": "user", "content": prompt}
        ],
        max_completion_tokens=512
    )
    return response.choices[0].message.content

if __name__ == "__main__":
    while True:
        user_query = input("\nAsk a question (or type 'exit'): ")
        if user_query.lower() == "exit":
            break

        retrieved_chunks = retrieve_documents(user_query)
        prompt = build_prompt(user_query, retrieved_chunks)
        answer = call_llama_groq(prompt)

        print("\n---- Answer ----\n")
        print(answer)
        print("-" * 50)
