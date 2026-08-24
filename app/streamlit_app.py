import streamlit as st

from graph.workflow import graph
from logs.logger import save_log


st.set_page_config(
    page_title="Agentic RAG",
    layout="centered"
)


st.title(
    "Agentic RAG Assistant"
)

st.write(
    "Ask questions about the provided document."
)


question = st.text_input(
    "Your question"
)

if st.button("Ask"):

    if not question.strip():

        st.warning(
            "Please enter a question."
        )

    else:

        with st.spinner(
            "Researching and reviewing..."
        ):

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

        st.subheader(
            "Final Answer"
        )

        st.write(
            result["final_answer"]
        )


        st.subheader(
            "Sources"
        )

        for source in result["sources"]:

            st.write(
                f" {source['source']} "
                f"— Page {source['page']} "
                f"— Similarity: "
                f"{source['score']:.3f}"
            )

        st.subheader(
            "Reviewer Verdict"
        )

        if (
            result["reviewer_verdict"]
            == "SUPPORTED"
        ):

            st.success(
                "✓ SUPPORTED"
            )

        else:

            st.error(
                "✗ UNSUPPORTED"
            )

            st.write(
                result["reviewer_reason"]
            )