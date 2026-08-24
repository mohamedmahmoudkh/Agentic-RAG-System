import os

from dotenv import load_dotenv
from langchain_openai import ChatOpenAI

load_dotenv()

OPENROUTER_API_KEY = os.getenv(
    "OPENROUTER_API_KEY"
)

llm = ChatOpenAI(
    model="openrouter/free",
    api_key=OPENROUTER_API_KEY,
    base_url="https://openrouter.ai/api/v1",
    temperature=0
)


def reviewer(
    question: str,
    draft_answer: str,
    documents: list
):

    evidence_parts = []

    for document in documents:

        evidence_parts.append(
            f"""
PAGE: {document['page']}

{document['text']}
"""
        )

    evidence = "\n\n".join(
        evidence_parts
    )
    prompt = f"""
You are the Reviewer Agent in an Agentic RAG system.

Your job is to verify whether the draft answer is
supported by the retrieved evidence.

USER QUESTION:
{question}

DRAFT ANSWER:
{draft_answer}

RETRIEVED EVIDENCE:
{evidence}

Rules:

1. Check every factual claim.
2. Every factual claim must be supported by the evidence.
3. Do not use outside knowledge.
4. If all important claims are supported, verdict = SUPPORTED.
5. If any important claim is not supported, verdict = UNSUPPORTED.

Return exactly this structure:

VERDICT: SUPPORTED

REASON:
<short explanation>

OR:

VERDICT: UNSUPPORTED

REASON:
<short explanation>
"""

    response = llm.invoke(
        prompt
    )

    review_text = response.content.strip()

    if "VERDICT: SUPPORTED" in review_text:

        verdict = "SUPPORTED"

    else:

        verdict = "UNSUPPORTED"

    return {
        "verdict": verdict,
        "reason": review_text
    }