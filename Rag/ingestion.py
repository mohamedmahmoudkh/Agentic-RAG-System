import os
import uuid
import fitz

from dotenv import load_dotenv
from sentence_transformers import SentenceTransformer

from langchain_text_splitters import RecursiveCharacterTextSplitter

from qdrant_client import QdrantClient
from qdrant_client.models import (
    Distance,
    VectorParams,
    PointStruct
)


# =========================================================
# Load environment variables
# =========================================================

load_dotenv()

QDRANT_URL = os.getenv("QDRANT_URL")
QDRANT_API_KEY = os.getenv("QDRANT_API_KEY")


# =========================================================
# Configuration
# =========================================================

PDF_PATH = "../data/book.pdf"

COLLECTION_NAME = "rich_dad_poor_dad"

# Local embedding model
EMBEDDING_MODEL_NAME = "BAAI/bge-small-en-v1.5"

# bge-small-en-v1.5 produces 384-dimensional vectors
VECTOR_SIZE = 384


# =========================================================
# Validate environment
# =========================================================

if not QDRANT_URL:
    raise ValueError(
        "QDRANT_URL is missing from .env"
    )

if not QDRANT_API_KEY:
    raise ValueError(
        "QDRANT_API_KEY is missing from .env"
    )


# =========================================================
# Connect to Qdrant
# =========================================================

print("=" * 60)
print("AGENTIC RAG - INGESTION")
print("=" * 60)

print("\nConnecting to Qdrant...")

client = QdrantClient(
    url=QDRANT_URL,
    api_key=QDRANT_API_KEY
)

print("Connected to Qdrant.")


# =========================================================
# Load local embedding model
# =========================================================

print("\nLoading embedding model...")

embedding_model = SentenceTransformer(
    EMBEDDING_MODEL_NAME
)

print(
    f"Embedding model loaded: "
    f"{EMBEDDING_MODEL_NAME}"
)


# =========================================================
# Load PDF
# =========================================================

def load_pdf(pdf_path):

    print("\nLoading PDF...")

    if not os.path.exists(pdf_path):

        raise FileNotFoundError(
            f"PDF not found: {pdf_path}"
        )

    pdf = fitz.open(pdf_path)

    documents = []

    for page_number, page in enumerate(pdf):

        text = page.get_text()

        text = text.strip()

        if not text:
            continue

        documents.append({
            "text": text,
            "page": page_number + 1,
            "source": os.path.basename(pdf_path)
        })

    pdf.close()

    print(
        f"Loaded {len(documents)} pages."
    )

    return documents


# =========================================================
# Create chunks
# =========================================================

def create_chunks(documents):

    print("\nCreating chunks...")

    splitter = RecursiveCharacterTextSplitter(
        chunk_size=800,
        chunk_overlap=100,
        separators=[
            "\n\n",
            "\n",
            ". ",
            " ",
            ""
        ]
    )

    chunks = []

    for document in documents:

        split_texts = splitter.split_text(
            document["text"]
        )

        for chunk_index, text in enumerate(
            split_texts
        ):

            text = text.strip()

            if not text:
                continue

            chunks.append({
                "text": text,
                "page": document["page"],
                "source": document["source"],
                "chunk_index": chunk_index
            })

    print(
        f"Created {len(chunks)} chunks."
    )

    return chunks


# =========================================================
# Create embeddings locally
# =========================================================

def create_embeddings(chunks):

    print("\nCreating local embeddings...")

    texts = [
        chunk["text"]
        for chunk in chunks
    ]

    vectors = embedding_model.encode(
        texts,
        batch_size=32,
        show_progress_bar=True,
        normalize_embeddings=True
    )

    vectors = vectors.tolist()

    print(
        f"Created {len(vectors)} embeddings."
    )

    print(
        f"Vector size: {len(vectors[0])}"
    )

    return vectors


# =========================================================
# Create Qdrant collection
# =========================================================

def create_collection():

    print("\nChecking Qdrant collection...")

    collections = client.get_collections()

    existing_collections = [
        collection.name
        for collection in collections.collections
    ]

    if COLLECTION_NAME in existing_collections:

        print(
            f"Collection '{COLLECTION_NAME}' "
            f"already exists."
        )

        return

    client.create_collection(
        collection_name=COLLECTION_NAME,

        vectors_config=VectorParams(
            size=VECTOR_SIZE,
            distance=Distance.COSINE
        )
    )

    print(
        f"Created collection: "
        f"{COLLECTION_NAME}"
    )


# =========================================================
# Upload vectors to Qdrant
# =========================================================

def upload_to_qdrant(chunks, vectors):

    print("\nUploading vectors to Qdrant...")

    points = []

    for index, (chunk, vector) in enumerate(
        zip(chunks, vectors)
    ):

        point_id = str(
            uuid.uuid5(
                uuid.NAMESPACE_URL,

                f"{chunk['source']}-"
                f"{chunk['page']}-"
                f"{index}"
            )
        )

        point = PointStruct(

            id=point_id,

            vector=vector,

            payload={
                "text": chunk["text"],
                "page": chunk["page"],
                "source": chunk["source"],
                "chunk_index": chunk["chunk_index"]
            }
        )

        points.append(point)

    # Upload in batches
    batch_size = 100

    total_points = len(points)

    for start in range(
        0,
        total_points,
        batch_size
    ):

        batch = points[
            start:start + batch_size
        ]

        client.upsert(
            collection_name=COLLECTION_NAME,
            points=batch
        )

        uploaded = min(
            start + batch_size,
            total_points
        )

        print(
            f"Uploaded "
            f"{uploaded}/{total_points}"
        )

    print(
        "\nAll vectors uploaded successfully!"
    )


# =========================================================
# Main
# =========================================================

def main():

    # -----------------------------------------
    # 1. Load PDF
    # -----------------------------------------

    documents = load_pdf(
        PDF_PATH
    )

    if not documents:

        raise ValueError(
            "No text was extracted from the PDF."
        )


    # -----------------------------------------
    # 2. Create chunks
    # -----------------------------------------

    chunks = create_chunks(
        documents
    )

    if not chunks:

        raise ValueError(
            "No chunks were created."
        )


    # -----------------------------------------
    # 3. Create Qdrant collection
    # -----------------------------------------

    create_collection()


    # -----------------------------------------
    # 4. Create embeddings
    # -----------------------------------------

    vectors = create_embeddings(
        chunks
    )


    # -----------------------------------------
    # 5. Validate vector size
    # -----------------------------------------

    actual_vector_size = len(
        vectors[0]
    )

    if actual_vector_size != VECTOR_SIZE:

        raise ValueError(
            f"Wrong vector size. "
            f"Expected {VECTOR_SIZE}, "
            f"got {actual_vector_size}"
        )


    # -----------------------------------------
    # 6. Upload to Qdrant
    # -----------------------------------------

    upload_to_qdrant(
        chunks,
        vectors
    )


    # -----------------------------------------
    # Done
    # -----------------------------------------

    print("\n" + "=" * 60)
    print("INGESTION COMPLETED SUCCESSFULLY!")
    print("=" * 60)


# =========================================================
# Run
# =========================================================

if __name__ == "__main__":
    main()