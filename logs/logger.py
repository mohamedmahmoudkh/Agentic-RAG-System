import os
import json
from datetime import datetime


LOG_FILE = "logs/results.jsonl"


def save_log(result):

    os.makedirs(
        "logs",
        exist_ok=True
    )

    log_entry = {
        "timestamp": datetime.now().isoformat(),

        "question": result.get(
            "question"
        ),

        "retrieved_chunks": result.get(
            "documents",
            []
        ),

        "sources": result.get(
            "sources",
            []
        ),

        "draft_answer": result.get(
            "draft_answer"
        ),

        "reviewer_verdict": result.get(
            "reviewer_verdict"
        ),

        "reviewer_reason": result.get(
            "reviewer_reason"
        ),

        "final_answer": result.get(
            "final_answer"
        )
    }

    with open(
        LOG_FILE,
        "a",
        encoding="utf-8"
    ) as file:

        file.write(
            json.dumps(
                log_entry,
                ensure_ascii=False
            )
            + "\n"
        )