import os
from groq import Groq
from Query import retrieve_documents

LLM_MODEL_NAME = "llama-3.3-70b-versatile"
client = Groq(api_key=os.getenv("GROQ_API_KEY"))
chat_history = []

def build_prompt(query, retrieved_chunks):
    context = "\n\n".join(retrieved_chunks)

    return f"""
You are a banking documentation assistant.

Answer ONLY using the provided context.

If the answer is not found in the context, reply:
"Sorry, Not found in the provided documents."

Context:
{context}

Question:
{query}
"""


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


def correct_query(query):
    try:
        response = client.chat.completions.create(
            model=LLM_MODEL_NAME,
            messages=[
                {
                    "role": "system",
                    "content": """
You are a helpful assistant that corrects spelling and grammar for banking queries.

Output ONLY the corrected query.

Do not change the meaning.
"""
                },
                {"role": "user", "content": query}
            ],
            max_completion_tokens=128
        )

        corrected = response.choices[0].message.content.strip()

        if corrected.startswith('"') and corrected.endswith('"'):
            corrected = corrected[1:-1]

        return corrected

    except Exception as e:
        print("Correction Error:", e)
        return query

def contextualize_query(query):
    if not chat_history:
        return query

    history = "\n".join(
        [
            f"Question: {item['question']}"
            for item in chat_history[-5:]
        ]
    )

    prompt = f"""
You are a banking assistant.

Based on the conversation history, rewrite the current question
as a complete standalone question.

Conversation History:
{history}

Current Question:
{query}

Return ONLY the rewritten question.
"""

    return call_llama_groq(prompt).strip()

def summarize_conversation():
    if not chat_history:
        return "No conversation history available."

    history_text = "\n\n".join(
        [
            f"Question: {item['question']}\nAnswer: {item['answer']}"
            for item in chat_history
        ]
    )

    summary_prompt = f"""
You are a banking assistant.

Summarize the user's conversation in a concise manner.

Conversation:

{history_text}
"""

    return call_llama_groq(summary_prompt)

def rag_chat(user_query):

    corrected_query = correct_query(user_query)

    standalone_query = contextualize_query(corrected_query)

    retrieved_chunks = retrieve_documents(standalone_query)

    prompt = build_prompt(
        standalone_query,
        retrieved_chunks
    )

    answer = call_llama_groq(prompt)

    chat_history.append(
        {
            "question": corrected_query,
            "answer": answer
        }
    )

    return answer

if __name__ == "__main__":

    print("\nBanking RAG Assistant Started")
    print("Type 'summarize' for conversation summary")
    print("Type 'exit' to quit\n")

    while True:

        user_query = input("\nAsk a question: ").strip()

        if user_query.lower() == "exit":
            chat_history.clear()
            print("\nSession ended.")
            print("Memory cleared successfully.")
            break

        if user_query.lower() == "summarize":
            summary = summarize_conversation()

            print("\nConversation Summary:\n")
            print(summary)
            print("\n" + "-" * 50)

            continue

        corrected_query = correct_query(user_query)

        print(f"\nCorrected Query: {corrected_query}")

        standalone_query = contextualize_query(corrected_query)

        print(f"\nStandalone Query: {standalone_query}")

        retrieved_chunks = retrieve_documents(standalone_query)

        prompt = build_prompt(
            standalone_query,
            retrieved_chunks
        )

        answer = call_llama_groq(prompt)

        chat_history.append(
            {
                "question": corrected_query,
                "answer": answer
            }
        )

        print("\nAnswer:\n")
        print(answer)
        print("\n" + "-" * 50)
