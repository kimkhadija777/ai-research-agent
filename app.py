import os
from typing import TypedDict

import streamlit as st
from ddgs import DDGS

from langchain_groq import ChatGroq
from langgraph.graph import StateGraph, START, END


# ============================================================
# PAGE CONFIGURATION
# ============================================================

st.set_page_config(
    page_title="AI Research Agent",
    page_icon="🔎",
    layout="centered"
)


# ============================================================
# TITLE
# ============================================================

st.title("🔎 AI Research Agent")

st.write(
    "Ask a research question and the agent will search "
    "the web and summarize the results using Groq."
)


# ============================================================
# GROQ API KEY
# ============================================================

# Streamlit Cloud will provide this through Secrets.
GROQ_API_KEY = st.secrets["GROQ_API_KEY"]

os.environ["GROQ_API_KEY"] = GROQ_API_KEY


# ============================================================
# GROQ MODEL
# ============================================================

MODEL_NAME = "openai/gpt-oss-120b"

llm = ChatGroq(
    model=MODEL_NAME,
    temperature=0.2
)


# ============================================================
# WEB SEARCH
# ============================================================

def web_search(query, max_results=5):

    results = []

    with DDGS() as ddgs:

        search_results = ddgs.text(
            query,
            max_results=max_results
        )

        for result in search_results:

            results.append(
                f"Title: {result.get('title', '')}\n"
                f"URL: {result.get('href', '')}\n"
                f"Snippet: {result.get('body', '')}\n"
            )

    return "\n".join(results)


# ============================================================
# LANGGRAPH STATE
# ============================================================

class ResearchState(TypedDict):

    question: str
    search_results: str
    answer: str


# ============================================================
# SEARCH NODE
# ============================================================

def search_web(state: ResearchState):

    question = state["question"]

    results = web_search(
        question,
        max_results=5
    )

    return {
        "search_results": results
    }


# ============================================================
# ANSWER NODE
# ============================================================

def generate_answer(state: ResearchState):

    question = state["question"]

    search_results = state["search_results"]

    prompt = f"""
You are a helpful AI research assistant.

The user asked:

{question}

Below are web search results:

{search_results}

Answer the user's question using the search results.

Instructions:

1. Give a clear and useful answer.
2. Summarize the most important information.
3. Use simple language.
4. Do not invent facts.
5. If the search results are insufficient, say so.
6. Mention important sources or URLs when useful.
7. Keep the answer reasonably concise.
"""

    response = llm.invoke(prompt)

    return {
        "answer": response.content
    }


# ============================================================
# BUILD LANGGRAPH
# ============================================================

graph_builder = StateGraph(ResearchState)

graph_builder.add_node(
    "search_web",
    search_web
)

graph_builder.add_node(
    "generate_answer",
    generate_answer
)

graph_builder.add_edge(
    START,
    "search_web"
)

graph_builder.add_edge(
    "search_web",
    "generate_answer"
)

graph_builder.add_edge(
    "generate_answer",
    END
)

research_agent = graph_builder.compile()


# ============================================================
# STREAMLIT UI
# ============================================================

question = st.text_area(
    "Enter your research question:",
    placeholder="Example: What is Retrieval Augmented Generation?",
    height=120
)


# ============================================================
# RESEARCH BUTTON
# ============================================================

if st.button("🔎 Research", type="primary"):

    if not question.strip():

        st.warning(
            "Please enter a research question."
        )

    else:

        with st.spinner(
            "🔎 Searching the web and generating your answer..."
        ):

            try:

                result = research_agent.invoke({

                    "question": question,

                    "search_results": "",

                    "answer": ""

                })

                st.subheader("📚 Research Answer")

                st.markdown(
                    result["answer"]
                )

            except Exception as e:

                st.error(
                    f"Something went wrong: {str(e)}"
)
