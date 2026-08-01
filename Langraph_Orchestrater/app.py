from fastapi import FastAPI
from pydantic import BaseModel
from graph import graph

app = FastAPI(title="LangGraph Orchestrator API")


class RequestData(BaseModel):
    input: str


@app.get("/")
def home():
    return {
        "status": "Running",
        "message": "LangGraph Orchestrator API"
    }


@app.post("/run")
def run_graph(data: RequestData):

    initial_state = {
        "input": data.input
    }

    result = graph.invoke(initial_state)

    return result