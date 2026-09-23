from backend.retriever import retrieve_documents
from backend.prompt_builder import build_prompt
from backend.llm import generate_answer


def answer_question(question):


    # Retrieve relevant documents from FAISS
    documents = retrieve_documents(
        question,
        top_k=3
    )

    # Build prompt using retrieved documents
    prompt = build_prompt(
        question,
        documents
    )

    # Generate answer using Gemini
    answer = generate_answer(prompt)

    return {
        "answer": answer,
        "sources": documents
    }