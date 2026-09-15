# -*- coding: utf-8 -*-

import os
import sys
import io
import traceback
from typing import TypedDict, List, Optional

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel

from langchain_core.messages import BaseMessage, HumanMessage
from langchain_core.tools import tool
from langchain_google_genai import ChatGoogleGenerativeAI
from langgraph.graph import StateGraph, START, END


# ============================================================
# FASTAPI APPLICATION
# ============================================================

app = FastAPI(
    title="LangGraph Agentic AI",
    description="Developer, Tester and Manager Agent workflow",
    version="1.0.0"
)


# Allow frontend / browser requests
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


# ============================================================
# ENVIRONMENT / GEMINI API KEY
# ============================================================

api_key = os.getenv("GEMINI_API_KEY")

if not api_key:
    raise ValueError(
        "GEMINI_API_KEY environment variable is not set"
    )


# ============================================================
# LLM INITIALIZATION
# ============================================================

llm_flash = ChatGoogleGenerativeAI(
    model="gemini-3.1-flash-lite-preview",
    google_api_key=api_key
)

llm = llm_flash


# ============================================================
# STATE
# ============================================================

class CrewState(TypedDict):
    messages: List[BaseMessage]
    next_step: Optional[str]
    code: Optional[str]
    report: Optional[str]


# ============================================================
# REQUEST / RESPONSE MODELS
# ============================================================

class TaskRequest(BaseModel):
    task: str


class TaskResponse(BaseModel):
    task: str
    generated_code: str
    test_cases: str
    execution_output: str
    report: str


# ============================================================
# TOOL 1 — RUN PYTHON CODE
# ============================================================

@tool
def run_python_code(code: str) -> str:
    """
    Execute generated Python code and return output or error.
    """

    if not isinstance(code, str):
        code = str(code)

    clean_code = (
        code
        .replace("```python", "")
        .replace("```", "")
        .strip()
    )

    old_stdout = sys.stdout
    new_stdout = io.StringIO()

    sys.stdout = new_stdout

    try:

        local_scope = {}

        exec(
            clean_code,
            {},
            local_scope
        )

        result = new_stdout.getvalue()

    except Exception:

        result = (
            "Execution Error:\n"
            + traceback.format_exc()
        )

    finally:

        sys.stdout = old_stdout

    if not result.strip():
        return "Success (no terminal output)"

    return result.strip()


# ============================================================
# TOOL 2 — GENERATE TEST CASES
# ============================================================

@tool
def generate_test_cases(task_description: str) -> str:
    """
    Generate 3 to 5 test scenarios for a coding task.
    """

    prompt = f"""
You are a Senior QA Engineer.

Generate 3 to 5 highly specific test scenarios
for the following coding task:

{task_description}

Include:
1. Normal test cases
2. Edge cases
3. Boundary cases where applicable

Return only a numbered list.
"""

    response = llm.invoke(prompt)

    if hasattr(response, "content"):
        return str(response.content)

    return str(response)


# ============================================================
# NODE 1 — TASK INPUT
# ============================================================

def task_input_node(state: CrewState):

    task = state["messages"][-1].content

    return {
        "next_step": "developer",
        "messages": state["messages"]
    }


# ============================================================
# NODE 2 — DEVELOPER AGENT
# ============================================================

def real_time_developer(state: CrewState):

    task = state["messages"][-1].content

    dev_prompt = f"""
You are the Developer Agent.

Write a clean Python program to solve this coding task:

{task}

Requirements:
- Write valid Python code.
- Keep the code simple and readable.
- Do not include explanations.
- Do not use Markdown.
- Return ONLY the Python code.
"""

    response = llm_flash.invoke(dev_prompt)

    content = response.content

    if isinstance(content, list):

        code_parts = []

        for item in content:

            if isinstance(item, dict):

                if "text" in item:
                    code_parts.append(
                        str(item["text"])
                    )

            else:

                code_parts.append(
                    str(item)
                )

        code_str = "\n".join(code_parts)

    else:

        code_str = str(content)

    code_str = (
        code_str
        .replace("```python", "")
        .replace("```", "")
        .strip()
    )

    return {
        "code": code_str,
        "next_step": "tester"
    }


# ============================================================
# NODE 3 — TESTER AGENT
# ============================================================

def real_time_tester(state: CrewState):

    task = state["messages"][-1].content

    code = state.get("code", "")

    # Generate test cases
    test_cases = generate_test_cases.invoke(
        task
    )

    # Execute generated code
    execution_result = run_python_code.invoke(
        {
            "code": code
        }
    )

    report = (
        "### EXECUTION OUTPUT\n\n"
        f"{execution_result}\n\n"
        "### TEST SCENARIOS EVALUATED\n\n"
        f"{test_cases}"
    )

    return {
        "report": report,
        "next_step": "manager"
    }


# ============================================================
# NODE 4 — MANAGER
# ============================================================

def manager_decision_node(state: CrewState):

    return {
        "next_step": "complete"
    }


# ============================================================
# GRAPH
# ============================================================

workflow = StateGraph(CrewState)


workflow.add_node(
    "task_input",
    task_input_node
)

workflow.add_node(
    "developer",
    real_time_developer
)

workflow.add_node(
    "tester",
    real_time_tester
)

workflow.add_node(
    "manager",
    manager_decision_node
)


# START → TASK INPUT

workflow.add_edge(
    START,
    "task_input"
)


# TASK INPUT → DEVELOPER

workflow.add_edge(
    "task_input",
    "developer"
)


# DEVELOPER → TESTER

workflow.add_edge(
    "developer",
    "tester"
)


# TESTER → MANAGER

workflow.add_edge(
    "tester",
    "manager"
)


# MANAGER → END

workflow.add_edge(
    "manager",
    END
)


# Compile LangGraph

rt_app = workflow.compile()


# ============================================================
# API ENDPOINTS
# ============================================================

@app.get("/")
def home():

    return {
        "status": "running",
        "message": "LangGraph Agentic AI is running successfully!",
        "docs": "/docs"
    }


@app.get("/health")
def health():

    return {
        "status": "healthy"
    }


@app.post("/run", response_model=TaskResponse)
def run_task(request: TaskRequest):

    task = request.task.strip()

    if not task:

        return {
            "task": "",
            "generated_code": "",
            "test_cases": "",
            "execution_output": "Task cannot be empty.",
            "report": "Please provide a coding task."
        }

    # Initial state
    initial_state: CrewState = {
        "messages": [
            HumanMessage(
                content=task
            )
        ],
        "next_step": "developer",
        "code": None,
        "report": None
    }

    # Execute LangGraph
    result = rt_app.invoke(
        initial_state
    )

    # Extract generated code
    generated_code = result.get(
        "code",
        ""
    )

    # Extract complete report
    report = result.get(
        "report",
        ""
    )

    # Extract execution output
    execution_output = ""

    if "### EXECUTION OUTPUT" in report:

        execution_output = (
            report
            .split(
                "### EXECUTION OUTPUT",
                1
            )[1]
            .split(
                "### TEST SCENARIOS EVALUATED",
                1
            )[0]
            .strip()
        )

    # Extract test cases
    test_cases = ""

    if "### TEST SCENARIOS EVALUATED" in report:

        test_cases = (
            report
            .split(
                "### TEST SCENARIOS EVALUATED",
                1
            )[1]
            .strip()
        )

    return TaskResponse(
        task=task,
        generated_code=generated_code,
        test_cases=test_cases,
        execution_output=execution_output,
        report=report
    )


# ============================================================
# LOCAL DEVELOPMENT
# ============================================================

if __name__ == "__main__":

    import uvicorn

    port = int(
        os.getenv(
            "PORT",
            "8000"
        )
    )

    uvicorn.run(
        app,
        host="0.0.0.0",
        port=port
    )
