import streamlit as st
from mineral_rag import (
    get_rag_components,
    extract_target_mineral,
    format_docs,
    REJECTION_MESSAGE
)

@st.cache_resource(show_spinner="Initializing RAG components...")
def cached_get_rag_components():
    return get_rag_components()

def get_response_stream(query: str):
    vectorstore, translator_chain, strict_chain, mineral_names = cached_get_rag_components()

    # 1. First stage: Translate/extract formal English mineral name using LLM
    translated_query = translator_chain.invoke({"query": query}).strip()

    # 2. Second stage: Python Gatekeeper matching translated name against CSV records
    target_mineral = extract_target_mineral(translated_query, mineral_names)
    if not target_mineral:
        def empty_response():
            yield REJECTION_MESSAGE
        return empty_response()

    # 3. Third stage: Filter Chroma vector store strictly by Name metadata
    docs = vectorstore.similarity_search(query, k=3, filter={"Name": target_mineral})
    if not docs:
        def empty_response():
            yield REJECTION_MESSAGE
        return empty_response()

    # 4. Fourth stage: Stream RAG answer from strict_chain
    formatted_context = format_docs(docs)
    return strict_chain.stream({"context": formatted_context, "question": query})

def main():
    st.set_page_config(page_title="Mineral Database RAG", page_icon="💎", layout="wide")
    st.title("💎 Mineral Database Local RAG System")
    st.markdown("Explore mineral properties, chemical compositions, and optical attributes using local Ollama (Phi-3), HuggingFace Embeddings, and Chroma DB.")

    if "messages" not in st.session_state:
        st.session_state.messages = []

    for message in st.session_state.messages:
        with st.chat_message(message["role"]):
            st.markdown(message["content"])

    if user_input := st.chat_input("Enter your mineral question..."):
        st.session_state.messages.append({"role": "user", "content": user_input})
        with st.chat_message("user"):
            st.markdown(user_input)

        with st.chat_message("assistant"):
            try:
                response = st.write_stream(get_response_stream(user_input))
                st.session_state.messages.append({"role": "assistant", "content": response})
            except Exception as e:
                error_msg = f"Error generating response: {e}"
                st.error(error_msg)

if __name__ == "__main__":
    main()
