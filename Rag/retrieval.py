import os

from dotenv import load_dotenv
from sentence_transformers import SentenceTransformer
from qdrant_client import QdrantClient
load_dotenv()

QDRANT_URL = os.getenv("QDRANT_URL")
QDRANT_API_KEY = os.getenv("QDRANT_API_KEY")
COLLECTION_NAME = "rich_dad_poor_dad"

EMBEDDING_MODEL_NAME = "BAAI/bge-small-en-v1.5"
if not QDRANT_URL:
    raise ValueError(
        "QDRANT_URL is missing from .env"
    )

if not QDRANT_API_KEY:
    raise ValueError(
        "QDRANT_API_KEY is missing from .env"
    )
print("Connecting to Qdrant...")

client = QdrantClient(
    url=QDRANT_URL,
    api_key=QDRANT_API_KEY
)

print("Connected to Qdrant.")
print("Loading embedding model...")

embedding_model = SentenceTransformer(
    EMBEDDING_MODEL_NAME
)

print("Embedding model loaded.")
def retrieve_documents(
    question: str,
    limit: int = 5
):

    """
    Convert the user's question into an embedding
    and retrieve the most relevant chunks from Qdrant.
    """

    if not question or not question.strip():

        return []


    query_vector = embedding_model.encode(
        question,
        normalize_embeddings=True
    ).tolist()


    results = client.query_points(
        collection_name=COLLECTION_NAME,
        query=query_vector,
        limit=limit,
        with_payload=True
    ).points


    documents = []

    for result in results:

        payload = result.payload or {}

        documents.append({

            "text": payload.get(
                "text",
                ""
            ),

            "page": payload.get(
                "page"
            ),

            "source": payload.get(
                "source"
            ),

            "chunk_index": payload.get(
                "chunk_index"
            ),

            "score": result.score
        })


    return documents


if __name__ == "__main__":

    print("\n" + "=" * 60)
    print("RAG RETRIEVAL TEST")
    print("=" * 60)

    question = input(
        "\nAsk a question: "
    ).strip()


    if not question:

        print(
            "Question cannot be empty."
        )

    else:

        results = retrieve_documents(
            question,
            limit=5
        )


        print(
            f"\nRetrieved {len(results)} documents."
        )


        for index, document in enumerate(
            results,
            start=1
        ):

            print("\n" + "=" * 60)

            print(
                f"RESULT #{index}"
            )

            print(
                f"Page: {document['page']}"
            )

            print(
                f"Score: {document['score']:.4f}"
            )

            print(
                f"Source: {document['source']}"
            )

            print("\nText:")

            print(
                document["text"]
            )