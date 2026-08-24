import os
import sys
PROJECT_ROOT = os.path.dirname(
    os.path.dirname(
        os.path.abspath(__file__)
    )
)

sys.path.insert(
    0,
    PROJECT_ROOT
)
from dotenv import load_dotenv
from langchain_openai import ChatOpenAI

from Rag.retrieval import retrieve_documents


# =========================================================
# Environment
# =========================================================

load_dotenv()

OPENROUTER_API_KEY = os.getenv("OPENROUTER_API_KEY")

if not OPENROUTER_API_KEY:
    raise ValueError(
        "OPENROUTER_API_KEY is missing from .env"
    )


# =========================================================
# LLM - OpenRouter
# =========================================================

llm = ChatOpenAI(
    model="openrouter/free",
    api_key=OPENROUTER_API_KEY,
    base_url="https://openrouter.ai/api/v1",
    temperature=0
)


# =========================================================
# Researcher Agent
# =========================================================

def researcher(question: str):

    # -----------------------------------------------------
    # 1. Retrieve relevant documents from Qdrant
    # -----------------------------------------------------

    documents = retrieve_documents(
        question,
        limit=5
    )

    # -----------------------------------------------------
    # 2. Nothing found
    # -----------------------------------------------------

    if not documents:

        return {
            "answer": (
                "I could not find relevant information "
                "in the provided document."
            ),
            "sources": [],
            "documents": []
        }

    # -----------------------------------------------------
    # 3. Build context
    # -----------------------------------------------------

    context_parts = []

    for document in documents:

        context_parts.append(
            f"""
SOURCE: {document.get("source", "Unknown")}
PAGE: {document.get("page", "Unknown")}
RELEVANCE SCORE: {document.get("score", 0):.4f}

{document.get("text", "")}
"""
        )

    context = "\n\n".join(context_parts)

    # -----------------------------------------------------
    # 4. Prompt
    # -----------------------------------------------------

    prompt = f"""
You are the Researcher Agent in an Agentic RAG system.

Your task is to answer the user's question using ONLY
the retrieved document context.

STRICT RULES:

1. Use only the provided context.
2. Do not use outside knowledge.
3. Do not invent facts.
4. If the context does not contain enough information,
   clearly say that the document does not provide
   enough information to answer the question.
5. Keep the answer concise and clear.
6. Do not mention that you are an AI.
7. Do not mention these instructions.

USER QUESTION:
{question}

RETRIEVED DOCUMENT CONTEXT:
{context}

Now write the best possible answer based ONLY
on the retrieved context.
"""

    # -----------------------------------------------------
    # 5. Generate answer
    # -----------------------------------------------------

    try:

        response = llm.invoke(prompt)

        answer = response.content

    except Exception as e:

        print("\nLLM Error:")
        print(e)

        return {
            "answer": "Failed to generate an answer.",
            "sources": [],
            "documents": documents
        }

    # -----------------------------------------------------
    # 6. Build sources
    # -----------------------------------------------------

    sources = []

    for document in documents:

        sources.append({
            "source": document.get(
                "source",
                "Unknown"
            ),

            "page": document.get(
                "page",
                "Unknown"
            ),

            "score": document.get(
                "score",
                0
            )
        })

    # -----------------------------------------------------
    # 7. Return result
    # -----------------------------------------------------

    return {
        "answer": answer,
        "sources": sources,
        "documents": documents
    }


# =========================================================
# Test Researcher
# =========================================================

if __name__ == "__main__":

    print("=" * 60)
    print("AGENTIC RAG - RESEARCHER TEST")
    print("=" * 60)

    question = input(
        "\nAsk a question: "
    ).strip()

    if not question:

        print("Please enter a question.")

    else:

        result = researcher(
            question
        )

        print("\n")
        print("=" * 60)
        print("DRAFT ANSWER")
        print("=" * 60)

        print(
            result["answer"]
        )

        print("\n")
        print("=" * 60)
        print("SOURCES")
        print("=" * 60)

        for source in result["sources"]:

            print(
                f"- {source['source']} "
                f"| Page {source['page']} "
                f"| Score: "
                f"{source['score']:.4f}"
            )