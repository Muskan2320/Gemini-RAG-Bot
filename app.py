import streamlit as st
from api import query_rag

st.set_page_config(page_title="Gemini RAG Chatbot", layout="wide")
st.title("Gemini RAG Chatbot")

st.markdown("""
    <style>
    .stTextInput > div > div > input {
        border: 2px solid red !important;
        border-radius: 6px;
        padding: 10px;
        outline: none !important;
        box-shadow: none !important;
    }
    </style>
""", unsafe_allow_html=True)

query = st.text_input(
    "Ask any question related to AI technology, Generative AI, Solar cars, System hardware, or software development. We will answer based on our available data. So, what's your question:"
)

if st.button("Submit"):
    if query.strip() == "":
        st.warning("Please enter a question.")
    else:
        with st.spinner("Thinking..."):
            try:
                response = query_rag(query)
                data = response.json()

                st.subheader("Answer:")
                st.write(data["answer"])
                
                st.subheader("Sources:")
                for source in data["sources parsed"]:
                    st.code(source)
            except Exception as e:
                st.error(f"An error occurred: {e}")