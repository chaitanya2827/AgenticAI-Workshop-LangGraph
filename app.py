# -*- coding: utf-8 -*-

import os
import sys
import io
import traceback
import html

from typing import TypedDict, List, Optional

from fastapi import FastAPI
from fastapi.responses import HTMLResponse
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel

from langchain_core.messages import BaseMessage, HumanMessage
from langchain_core.tools import tool
from langchain_google_genai import ChatGoogleGenerativeAI
from langgraph.graph import StateGraph, START, END


# ============================================================
# FASTAPI APP
# ============================================================

app = FastAPI(
    title="Agentic AI LangGraph",
    description="Developer - Tester - Manager AI workflow",
    version="1.0"
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"]
)


# ============================================================
# GEMINI API KEY
# ============================================================

api_key = os.getenv("GEMINI_API_KEY")

if not api_key:
    raise ValueError(
        "GEMINI_API_KEY environment variable is not set"
    )


# ============================================================
# GEMINI MODEL
# ============================================================

llm = ChatGoogleGenerativeAI(
    model="gemini-3.1-flash-lite-preview",
    google_api_key=api_key
)

llm_flash = llm


# ============================================================
# LANGGRAPH STATE
# ============================================================

class CrewState(TypedDict):
    messages: List[BaseMessage]
    next_step: Optional[str]
    code: Optional[str]
    report: Optional[str]


# ============================================================
# REQUEST MODEL
# ============================================================

class TaskRequest(BaseModel):
    task: str


# ============================================================
# TOOL: EXECUTE PYTHON CODE
# ============================================================

@tool
def run_python_code(code: str) -> str:
    """
    Execute generated Python code and return the result.
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
# TOOL: GENERATE TEST CASES
# ============================================================

@tool
def generate_test_cases(task_description: str) -> str:
    """
    Generate test cases for the coding task.
    """

    prompt = f"""
You are a Senior QA Engineer.

Generate 3 to 5 highly specific test scenarios
for this coding task:

{task_description}

Include:
- Normal cases
- Edge cases
- Boundary cases when applicable

Return only a numbered list.
"""

    response = llm.invoke(prompt)

    if hasattr(response, "content"):
        content = response.content

        if isinstance(content, list):

            parts = []

            for item in content:

                if isinstance(item, dict):
                    parts.append(
                        str(item.get("text", ""))
                    )
                else:
                    parts.append(str(item))

            return "\n".join(parts)

        return str(content)

    return str(response)


# ============================================================
# NODE 1: TASK INPUT
# ============================================================

def task_input_node(state: CrewState):

    return {
        "messages": state["messages"],
        "next_step": "developer"
    }


# ============================================================
# NODE 2: DEVELOPER AGENT
# ============================================================

def real_time_developer(state: CrewState):

    task = state["messages"][-1].content

    prompt = f"""
You are the Developer Agent.

Solve this coding task:

{task}

Requirements:

1. Write clean Python code.
2. Make the code executable.
3. Keep it simple and readable.
4. Do not explain the code.
5. Do not use Markdown.
6. Return ONLY Python code.

Coding task:
{task}
"""

    response = llm_flash.invoke(prompt)

    content = response.content

    if isinstance(content, list):

        parts = []

        for item in content:

            if isinstance(item, dict):
                parts.append(
                    str(item.get("text", ""))
                )
            else:
                parts.append(str(item))

        code = "\n".join(parts)

    else:

        code = str(content)

    code = (
        code
        .replace("```python", "")
        .replace("```", "")
        .strip()
    )

    return {
        "code": code,
        "next_step": "tester"
    }


# ============================================================
# NODE 3: TESTER AGENT
# ============================================================

def real_time_tester(state: CrewState):

    task = state["messages"][-1].content

    code = state.get("code", "")

    # Generate test cases
    test_cases = generate_test_cases.invoke(
        task
    )

    # Execute generated code
    execution_output = run_python_code.invoke(
        {
            "code": code
        }
    )

    report = (
        "### EXECUTION OUTPUT\n\n"
        + execution_output
        + "\n\n"
        + "### TEST SCENARIOS EVALUATED\n\n"
        + str(test_cases)
    )

    return {
        "report": report,
        "next_step": "manager"
    }


# ============================================================
# NODE 4: MANAGER
# ============================================================

def manager_node(state: CrewState):

    return {
        "next_step": "complete"
    }


# ============================================================
# BUILD LANGGRAPH
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
    manager_node
)


workflow.add_edge(
    START,
    "task_input"
)

workflow.add_edge(
    "task_input",
    "developer"
)

workflow.add_edge(
    "developer",
    "tester"
)

workflow.add_edge(
    "tester",
    "manager"
)

workflow.add_edge(
    "manager",
    END
)


rt_app = workflow.compile()


# ============================================================
# WEBPAGE
# ============================================================

HTML_PAGE = """
<!DOCTYPE html>

<html>

<head>

<meta name="viewport" content="width=device-width, initial-scale=1.0">

<title>Agentic AI - LangGraph</title>

<style>

* {
    box-sizing: border-box;
}

body {

    margin: 0;

    font-family:
        Arial,
        Helvetica,
        sans-serif;

    background:
        linear-gradient(
            135deg,
            #0f172a,
            #1e293b
        );

    color: white;

    min-height: 100vh;
}


.container {

    width: 92%;

    max-width: 1100px;

    margin: auto;

    padding: 40px 0;
}


.header {

    text-align: center;

    margin-bottom: 35px;
}


.header h1 {

    font-size: 42px;

    margin-bottom: 10px;
}


.header p {

    color: #cbd5e1;

    font-size: 17px;
}


.card {

    background: rgba(
        255,
        255,
        255,
        0.07
    );

    border: 1px solid rgba(
        255,
        255,
        255,
        0.12
    );

    border-radius: 18px;

    padding: 25px;

    margin-bottom: 25px;

    backdrop-filter: blur(10px);
}


label {

    display: block;

    margin-bottom: 10px;

    font-weight: bold;

    font-size: 16px;
}


textarea {

    width: 100%;

    min-height: 130px;

    resize: vertical;

    padding: 15px;

    border-radius: 12px;

    border: 1px solid #475569;

    background: #0f172a;

    color: white;

    font-size: 15px;

    outline: none;
}


textarea:focus {

    border-color: #38bdf8;
}


button {

    width: 100%;

    margin-top: 18px;

    padding: 15px;

    border: none;

    border-radius: 12px;

    background:
        linear-gradient(
            90deg,
            #2563eb,
            #7c3aed
        );

    color: white;

    font-size: 16px;

    font-weight: bold;

    cursor: pointer;
}


button:hover {

    opacity: 0.9;
}


button:disabled {

    opacity: 0.5;

    cursor: not-allowed;
}


.loading {

    display: none;

    text-align: center;

    padding: 20px;

    color: #38bdf8;
}


.result {

    display: none;
}


.section {

    margin-top: 25px;
}


.section h3 {

    margin-bottom: 10px;

    color: #38bdf8;
}


pre {

    background: #020617;

    border: 1px solid #334155;

    padding: 18px;

    border-radius: 12px;

    overflow-x: auto;

    white-space: pre-wrap;

    word-wrap: break-word;

    color: #e2e8f0;

    line-height: 1.5;
}


.status {

    text-align: center;

    color: #86efac;

    margin-top: 15px;

    font-weight: bold;
}


.footer {

    text-align: center;

    margin-top: 35px;

    color: #94a3b8;

    font-size: 13px;
}

</style>

</head>


<body>

<div class="container">


<div class="header">

<h1>🤖 Agentic AI</h1>

<p>
LangGraph Developer → Tester → Manager Workflow
</p>

</div>


<div class="card">

<label>
Enter your coding task
</label>


<textarea
id="task"
placeholder="Example: Write a Python program to find the largest number in a list..."
></textarea>


<button
id="runButton"
onclick="runAgent()"
>

🚀 Run Agent

</button>


<div
class="loading"
id="loading"
>

⏳ AI agents are working... Please wait.

</div>

</div>


<div
class="card result"
id="result"
>


<div class="status">
✅ Workflow completed successfully
</div>


<div class="section">

<h3>🧑‍💻 Developer Agent — Generated Code</h3>

<pre id="code"></pre>

</div>


<div class="section">

<h3>🧪 Tester Agent — Test Cases</h3>

<pre id="tests"></pre>

</div>


<div class="section">

<h3>▶️ Execution Output</h3>

<pre id="output"></pre>

</div>


<div class="section">

<h3>📋 Manager Report</h3>

<pre id="report"></pre>

</div>


</div>


<div class="footer">

Powered by FastAPI + LangGraph + Gemini

</div>


</div>


<script>

async function runAgent() {

    const task =
        document
        .getElementById("task")
        .value
        .trim();

    if (!task) {

        alert(
            "Please enter a coding task."
        );

        return;
    }


    const button =
        document
        .getElementById("runButton");

    const loading =
        document
        .getElementById("loading");

    const result =
        document
        .getElementById("result");


    button.disabled = true;

    loading.style.display = "block";

    result.style.display = "none";


    try {

        const response =
            await fetch(
                "/run",
                {

                    method: "POST",

                    headers: {
                        "Content-Type":
                            "application/json"
                    },

                    body: JSON.stringify({
                        task: task
                    })

                }
            );


        const data =
            await response.json();


        if (!response.ok) {

            throw new Error(
                data.detail ||
                "Something went wrong"
            );

        }


        document
            .getElementById("code")
            .textContent =
                data.generated_code;


        document
            .getElementById("tests")
            .textContent =
                data.test_cases;


        document
            .getElementById("output")
            .textContent =
                data.execution_output;


        document
            .getElementById("report")
            .textContent =
                data.report;


        result.style.display =
            "block";


        result.scrollIntoView({
            behavior: "smooth"
        });


    }

    catch (error) {

        alert(
            "Error: " +
            error.message
        );

    }

    finally {

        button.disabled = false;

        loading.style.display =
            "none";

    }

}

</script>


</body>

</html>
"""


# ============================================================
# HOME PAGE
# ============================================================

@app.get(
    "/",
    response_class=HTMLResponse
)
def home():

    return HTML_PAGE


# ============================================================
# HEALTH CHECK
# ============================================================

@app.get("/health")
def health():

    return {
        "status": "healthy",
        "service": "Agentic AI LangGraph"
    }


# ============================================================
# RUN LANGGRAPH
# ============================================================

@app.post("/run")
def run_task(request: TaskRequest):

    task = request.task.strip()

    if not task:

        return {
            "error": "Task cannot be empty"
        }


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


    # Run LangGraph

    result = rt_app.invoke(
        initial_state
    )


    generated_code = result.get(
        "code",
        ""
    )


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


    return {

        "task": task,

        "generated_code":
            generated_code,

        "test_cases":
            test_cases,

        "execution_output":
            execution_output,

        "report":
            report

    }


# ============================================================
# START SERVER LOCALLY
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
