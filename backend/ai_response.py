from openai import AzureOpenAI
from backend.config import DIAL_API_KEY

client = AzureOpenAI(
    api_key=DIAL_API_KEY,
    api_version="2024-02-01",
    azure_endpoint="https://ai-proxy.lab.epam.com"
)

deployment_name = "gpt-35-turbo"

def improve_answer(user_query, retrieved_answer, chat_history):
    """Ensure the improved answer remains relevant to the original query."""

    # print(f"Debug: Using API Key - {DIAL_API_KEY}")

    # STRONG system prompt to prevent repeated answers
    system_message = {
        "role": "system",
        "content": (
            "You are a COVID-19 FAQ assistant. Your job is to refine and clarify the retrieved answer ONLY. "
            "Ensure the improved answer is directly relevant to the user question. "
            "DO NOT reuse or repeat previous responses."
        )
    }

    messages = [system_message]

    # Keep only the last 3 messages for better context
    for msg in chat_history[-3:]:
        messages.append(msg)

    messages.append({"role": "user", "content": f"User Question: {user_query}"})
    messages.append({"role": "assistant", "content": f"Retrieved FAQ Answer: {retrieved_answer}"})
    messages.append({"role": "user", "content": "Can you refine this answer while keeping it directly relevant to the question?"})

    try:
        response = client.chat.completions.create(
            model=deployment_name,
            temperature=0.2,
            messages=messages
        )

        improved_answer = response.choices[0].message.content
        print(f"Refined Answer: {improved_answer}")  # Debug output

        return improved_answer

    except Exception as e:
        print(f"API Error: {str(e)}")
        return f"[API Error] {retrieved_answer}"  # Return original answer if API fails
