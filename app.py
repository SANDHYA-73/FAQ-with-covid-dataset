import streamlit as st
import requests

#Set API URL
API_URL = "http://127.0.0.1:8000/ask"

#Initialize chat history
if "messages" not in st.session_state:
    st.session_state.messages = []

st.title("🤖 COVID-19 FAQ Chatbot")
st.write("Ask any COVID-19 related question, and I'll provide an answer!")

#Display previous messages
for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

#User Input
user_input = st.chat_input("Type your question...")

if user_input:
    #Append user question to history
    st.session_state.messages.append({"role": "user", "content": user_input})

    #Send request to FastAPI with chat history
    response = requests.post(API_URL, json={"query": user_input, "history": st.session_state.messages})

    if response.status_code == 200:
        data = response.json()
        bot_response = data.get("enhanced_answer", "Sorry, I couldn't find an answer.")

        #Append bot response to history
        st.session_state.messages.append({"role": "assistant", "content": bot_response})
        
        #Display user question
        with st.chat_message("user"):
            st.markdown(f"**You:** {user_input}")

        #Display bot response
        with st.chat_message("assistant"):
            st.markdown(f"**Bot:** {bot_response}")
    else:
        st.error("❌ Error fetching response. Try again later.")