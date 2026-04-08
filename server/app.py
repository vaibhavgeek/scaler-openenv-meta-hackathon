"""FastAPI server exposing the PricingEnv via HTTP for HF Spaces / OpenEnv.

Endpoints:
  POST /reset          → reset the environment, returns initial observation
  POST /step           → take an action, returns observation + reward + done
  GET  /state          → current environment state
  GET  /tasks          → list available tasks
  POST /reset/{task}   → reset with a specific task configuration
  GET  /               → health check
"""

from __future__ import annotations

from typing import Optional

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel

try:
    from pricing_env.env import PricingEnv, PricingAction, Result as EnvResult
    from pricing_env.tasks import TASKS, TASK_MAP, TaskDef, reset_for_task, grade_episode
except ImportError:
    from env import PricingEnv, PricingAction, Result as EnvResult
    from tasks import TASKS, TASK_MAP, TaskDef, reset_for_task, grade_episode

app = FastAPI(
    title="Pricing Negotiation OpenEnv",
    description="AI agent pricing negotiation environment",
    version="1.0.0",
)

# Global env instance
env = PricingEnv(list_price=100.0, cost=50.0, max_turns=6, seed=42)
current_task: Optional[TaskDef] = None
episode_turns: int = 0
episode_sold: bool = False
episode_final_price: Optional[float] = None


class ActionRequest(BaseModel):
    message: str


class ResetResponse(BaseModel):
    observation: dict
    reward: float
    done: bool
    info: dict


class StepResponse(BaseModel):
    observation: dict
    reward: float
    done: bool
    info: dict


class TaskInfo(BaseModel):
    name: str
    difficulty: str
    description: str


@app.get("/")
def health():
    return {"status": "ok", "env": "pricing-negotiation", "version": "1.0.0"}


@app.get("/tasks")
def list_tasks():
    return [
        TaskInfo(name=t.name, difficulty=t.difficulty, description=t.description)
        for t in TASKS
    ]


@app.post("/reset")
def reset():
    global current_task, episode_turns, episode_sold, episode_final_price
    current_task = None
    episode_turns = 0
    episode_sold = False
    episode_final_price = None

    result = env.reset()
    return ResetResponse(
        observation=result.observation.model_dump(),
        reward=result.reward,
        done=result.done,
        info=result.info,
    )


@app.post("/reset/{task_name}")
def reset_task(task_name: str):
    global current_task, episode_turns, episode_sold, episode_final_price

    if task_name not in TASK_MAP:
        raise HTTPException(status_code=404, detail=f"Task '{task_name}' not found. Available: {list(TASK_MAP.keys())}")

    current_task = TASK_MAP[task_name]
    episode_turns = 0
    episode_sold = False
    episode_final_price = None

    result = reset_for_task(env, current_task)
    return ResetResponse(
        observation=result.observation.model_dump(),
        reward=result.reward,
        done=result.done,
        info=result.info,
    )


@app.post("/step")
def step(action: ActionRequest):
    global episode_turns, episode_sold, episode_final_price

    if env.done:
        raise HTTPException(status_code=400, detail="Episode is done. Call /reset first.")

    result = env.step(PricingAction(message=action.message))
    episode_turns += 1

    if result.info.get("final_outcome") == "sold":
        episode_sold = True
        episode_final_price = result.info.get("offer")

    response = StepResponse(
        observation=result.observation.model_dump(),
        reward=result.reward,
        done=result.done,
        info=result.info,
    )

    # If done and we have a task, include the graded score
    if result.done and current_task:
        score = grade_episode(current_task, episode_sold, episode_final_price, episode_turns)
        response.info["graded_score"] = score

    return response


@app.get("/state")
def state():
    return env.state().model_dump()


def main():
    import uvicorn
    uvicorn.run("server.app:app", host="0.0.0.0", port=7860)
