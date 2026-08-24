import json
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
from graph.workflow import graph
from logs.logger import save_log
with open(
    "tests/test_questions.json",
    "r",
    encoding="utf-8"
) as file:

    questions = json.load(file)


print("=" * 60)
print("RUNNING TESTS")
print("=" * 60)
for test in questions:

    print(
        f"\nRunning Test #{test['id']}"
    )

    question = test["question"]

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

    save_log(
        result
    )

    print(
        f"Verdict: "
        f"{result['reviewer_verdict']}"
    )


print("\n")
print("=" * 60)

print(
    f"Completed {len(questions)} tests."
)

print("=" * 60)