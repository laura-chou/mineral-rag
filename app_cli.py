import sys
from mineral_rag import (
    get_rag_components,
    extract_target_mineral,
    format_docs,
    REJECTION_MESSAGE
)

def main():
    print("Initializing RAG components...")
    vectorstore, translator_chain, strict_chain, mineral_names = get_rag_components()

    print("\n" + "="*50)
    print(" Mineral Database RAG Retrieval System (CLI Debugger)")
    print("="*50)
    print("System ready! You can ask your questions at any time.")

    while True:
        try:
            user_input = input("\nEnter your mineral question (type 'exit' or 'quit' to leave): ").strip()
            if not user_input:
                continue
            if user_input.lower() in ["exit", "quit"]:
                print("\nThank you for using the Mineral RAG system. Goodbye!")
                break

            print("\n[Stage 1] Extracting formal mineral name...")
            translated_query = translator_chain.invoke({"query": user_input}).strip()
            print(f"Extracted English Term: {translated_query}")

            print("[Stage 2] Gatekeeper check against CSV records...")
            target_mineral = extract_target_mineral(translated_query, mineral_names)
            if not target_mineral:
                print("\nResponse:")
                print(REJECTION_MESSAGE)
                continue

            print(f"[Stage 3] Filtering Chroma DB for '{target_mineral}'...")
            docs = vectorstore.similarity_search(user_input, k=3, filter={"Name": target_mineral})
            if not docs:
                print("\nResponse:")
                print(REJECTION_MESSAGE)
                continue

            print("[Stage 4] Generating grounded answer...")
            print("\nResponse:")
            formatted_context = format_docs(docs)
            response = strict_chain.invoke({"context": formatted_context, "question": user_input})
            print(response)

        except KeyboardInterrupt:
            print("\n\nProgram interrupted by user. Goodbye!")
            break
        except Exception as e:
            print(f"\nError occurred: {e}")

if __name__ == "__main__":
    main()
