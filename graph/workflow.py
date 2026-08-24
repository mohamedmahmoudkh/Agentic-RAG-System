import os
import sys
from typing import TypedDict
from logs.logger import save_log
PROJECT_ROOT = os.path.dirname(
    os.path.dirname(
        os.path.abspath(__file__)
    )
)

sys.path.insert(
    0,
    PROJECT_ROOT
)
from langgraph.graph import (
    StateGraph,
    START,
    END
)

from agents.researcher import researcher
from agents.reviewer import reviewer


# =========================================================
# State
# =========================================================

class AgentState(TypedDict):

    question: str

    documents: list

    sources: list

    draft_answer: str

    reviewer_verdict: str

    reviewer_reason: str

    final_answer: str


# =========================================================
# Researcher Node
# =========================================================

def researcher_node(
    state: AgentState
):

    result = researcher(
        state["question"]
    )

    return {
        "documents": result["documents"],
        "sources": result["sources"],
        "draft_answer": result["answer"]
    }


# =========================================================
# Reviewer Node
# =========================================================

def reviewer_node(
    state: AgentState
):

    result = reviewer(
        question=state["question"],
        draft_answer=state["draft_answer"],
        documents=state["documents"]
    )

    return {
        "reviewer_verdict": result["verdict"],
        "reviewer_reason": result["reason"]
    }


# =========================================================
# Final Answer Node
# =========================================================

def final_node(
    state: AgentState
):

    if state["reviewer_verdict"] == "SUPPORTED":

        final_answer = state["draft_answer"]

    else:

        final_answer = (
            "I cannot provide a reliable answer "
            "because the retrieved evidence does "
            "not sufficiently support the generated answer."
        )

    return {
        "final_answer": final_answer
    }


# =========================================================
# Build Graph
# =========================================================

workflow = StateGraph(
    AgentState
)


workflow.add_node(
    "researcher",
    researcher_node
)

workflow.add_node(
    "reviewer",
    reviewer_node
)

workflow.add_node(
    "final",
    final_node
)


workflow.add_edge(
    START,
    "researcher"
)

workflow.add_edge(
    "researcher",
    "reviewer"
)

workflow.add_edge(
    "reviewer",
    "final"
)

workflow.add_edge(
    "final",
    END
)


graph = workflow.compile()


# =========================================================
# Test Workflow
# =========================================================

if __name__ == "__main__":

    question = input(
        "\nAsk a question: "
    )

    initial_state = {
        "question": question,
        "documents": [],
        "sources": [],
        "draft_answer": "",
        "reviewer_verdict": "",
        "reviewer_reason": "",
        "final_answer": ""
    }

    result = graph.invoke(
        initial_state
    )

    print("\n")
    print("=" * 60)
    print("FINAL ANSWER")
    print("=" * 60)

    print(
        result["final_answer"]
    )

    print("\n")
    print("=" * 60)
    print("SOURCES")
    print("=" * 60)

    for source in result["sources"]:

        print(
            f"- {source['source']} "
            f"| Page {source['page']} "
            f"| Score {source['score']:.4f}"
        )

    print("\n")
    print("=" * 60)
    print("REVIEWER VERDICT")
    print("=" * 60)

    print(
        result["reviewer_verdict"]
    )

    print(
        result["reviewer_reason"]
    )
    result = graph.invoke(
    initial_state
    )
    save_log(result)