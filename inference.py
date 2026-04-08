"""Baseline inference script for pricing negotiation environment.

Uses the OpenAI API client to run an LLM agent against all 3 tasks.
Emits structured stdout logs in the required [START], [STEP], [END] format.

Environment variables:
  API_BASE_URL  - The API endpoint for the LLM
  MODEL_NAME    - The model identifier to use for inference
  HF_TOKEN      - Your Hugging Face / API key (used as API key)
"""

from __future__ import annotations

import json
import os
import sys
import time
from typing import List, Optional

try:
    from dotenv import load_dotenv
    load_dotenv()
except ImportError:
    pass

from openai import OpenAI

import sys, os
# Support running from inside the package directory or from a parent
_here = os.path.dirname(os.path.abspath(__file__))
if os.path.exists(os.path.join(_here, "env.py")):
    # Running from inside the package dir — add parent so 'pricing_env' resolves
    # only if the directory is actually named pricing_env
    _parent = os.path.dirname(_here)
    _pkg_name = os.path.basename(_here)
    if _parent not in sys.path:
        sys.path.insert(0, _parent)
    # Also support direct imports when dir isn't named pricing_env
    try:
        from env import PricingEnv, PricingAction, Observation
        from tasks import TASKS, TaskDef, reset_for_task, grade_episode
    except ImportError:
        from pricing_env.env import PricingEnv, PricingAction, Observation
        from pricing_env.tasks import TASKS, TaskDef, reset_for_task, grade_episode
else:
    from pricing_env.env import PricingEnv, PricingAction, Observation
    from pricing_env.tasks import TASKS, TaskDef, reset_for_task, grade_episode

# ---------- config from env vars ----------

API_BASE_URL = os.environ.get("API_BASE_URL", "https://generativelanguage.googleapis.com/v1beta/openai/")
MODEL_NAME = os.environ.get("MODEL_NAME", "gemini-2.0-flash-lite")
API_KEY = os.environ.get("HF_TOKEN") or os.environ.get("GEMINI_API_KEY") or os.environ.get("OPENAI_API_KEY", "")

MAX_STEPS = 6
MAX_TOKENS = 256
BENCHMARK = "pricing-negotiation"
SUCCESS_SCORE_THRESHOLD = 0.5
MAX_TOTAL_REWARD = 1.0

SYSTEM_PROMPT = """You are an expert sales negotiator. Your goal is to maximize revenue while closing deals.

## RULES
1. You are selling a product with a known list price and a cost floor.
2. NEVER offer below your cost floor.
3. You have a limited number of turns before the buyer walks away.
4. Each response MUST contain exactly one offer in the format: [offer: $X.XX]
5. After the offer, you may include a short (1-2 sentence) justification.

## STRATEGY
- Anchor high on turn 1 (near list price) to leave room for concessions.
- Concede gradually; large drops signal desperation.
- Read the buyer's tone: eager buyers will pay more, bargain hunters need work.
- If the buyer counter-offers, don't just split the difference mechanically.
- Closing a deal at a moderate margin beats no deal at all.

## RESPONSE FORMAT
[offer: $XX.XX] <one sentence justification>
"""


# ---------- logging helpers ----------

def log_start(task: str, env: str, model: str) -> None:
    print(json.dumps({
        "type": "START",
        "task": task,
        "env": env,
        "model": model,
        "timestamp": time.time(),
    }), flush=True)


def log_step(step: int, action: str, reward: float, done: bool, error: Optional[str] = None) -> None:
    entry = {
        "type": "STEP",
        "step": step,
        "action": action,
        "reward": reward,
        "done": done,
        "timestamp": time.time(),
    }
    if error:
        entry["error"] = error
    print(json.dumps(entry), flush=True)


def log_end(success: bool, steps: int, score: float, rewards: List[float]) -> None:
    print(json.dumps({
        "type": "END",
        "success": success,
        "steps": steps,
        "score": score,
        "rewards": rewards,
        "timestamp": time.time(),
    }), flush=True)


# ---------- agent ----------

def build_user_prompt(obs: Observation) -> str:
    lines = [
        f"List price: ${obs.list_price:.2f}",
        f"Your cost floor: ${obs.cost:.2f}",
        f"Turn: {obs.turn + 1} / {obs.max_turns}",
        "",
        "Conversation so far:",
    ]
    if not obs.messages:
        lines.append("[BUYER] (waiting for your opening offer)")
    else:
        for m in obs.messages:
            lines.append(f"[{m.category}] {m.content}")
    lines.append("")
    lines.append("Reply with a single offer in the format: [offer: $X.XX] <justification>")
    return "\n".join(lines)


def get_model_message(
    client: OpenAI,
    obs: Observation,
    history: List[str],
) -> str:
    user_prompt = build_user_prompt(obs)

    messages = [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content": user_prompt},
    ]

    try:
        completion = client.chat.completions.create(
            model=MODEL_NAME,
            messages=messages,
            max_tokens=MAX_TOKENS,
            stream=False,
        )
        text = (completion.choices[0].message.content or "").strip()
        return text if text else "[offer: $75.00] Let's make a deal."
    except Exception as exc:
        print(f"[DEBUG] Model request failed: {exc}", flush=True)
        return "[offer: $75.00] Let's make a deal."


# ---------- run one task ----------

def run_task(client: OpenAI, task: TaskDef, seed: int = 42) -> dict:
    env = PricingEnv(
        list_price=task.list_price,
        cost=task.cost,
        max_turns=task.max_turns,
        seed=seed,
    )

    history: List[str] = []
    rewards: List[float] = []
    steps_taken = 0
    sold = False
    final_price = None

    log_start(task=task.name, env=BENCHMARK, model=MODEL_NAME)

    try:
        result = reset_for_task(env, task)

        for step in range(1, MAX_STEPS + 1):
            if result.done:
                break

            message = get_model_message(client, result.observation, history)
            result = env.step(PricingAction(message=message))

            reward = result.reward or 0.0
            done = result.done
            error = None

            rewards.append(reward)
            steps_taken = step

            log_step(step=step, action=message, reward=reward, done=done, error=error)
            history.append(f"Step {step}: {message!r} -> reward {reward:+.2f}")

            if result.info.get("final_outcome") == "sold":
                sold = True
                final_price = result.info.get("offer")

            if done:
                break

        score = grade_episode(task, sold, final_price, steps_taken)
        success = score >= SUCCESS_SCORE_THRESHOLD

    except Exception as exc:
        print(f"[DEBUG] Task {task.name} failed: {exc}", flush=True)
        score = 0.0
        success = False

    log_end(success=success, steps=steps_taken, score=score, rewards=rewards)

    return {
        "task": task.name,
        "difficulty": task.difficulty,
        "score": score,
        "success": success,
        "sold": sold,
        "final_price": final_price,
        "steps": steps_taken,
    }


# ---------- main ----------

def main() -> None:
    client = OpenAI(base_url=API_BASE_URL, api_key=API_KEY)

    print(f"[INFO] Running baseline inference on {len(TASKS)} tasks", flush=True)
    print(f"[INFO] Model: {MODEL_NAME}", flush=True)
    print(f"[INFO] API base: {API_BASE_URL}", flush=True)

    results = []
    for task in TASKS:
        print(f"\n{'='*60}", flush=True)
        print(f"[INFO] Task: {task.name} ({task.difficulty})", flush=True)
        print(f"{'='*60}", flush=True)

        result = run_task(client, task, seed=42)
        results.append(result)

    # Summary
    print(f"\n{'='*60}", flush=True)
    print("[SUMMARY] Baseline Results:", flush=True)
    print(f"{'='*60}", flush=True)
    for r in results:
        status = "PASS" if r["success"] else "FAIL"
        print(
            f"  [{status}] {r['task']} ({r['difficulty']}): "
            f"score={r['score']:.3f} sold={r['sold']} "
            f"price={r['final_price']} steps={r['steps']}",
            flush=True,
        )

    avg_score = sum(r["score"] for r in results) / len(results)
    print(f"\n  Average score: {avg_score:.3f}", flush=True)


if __name__ == "__main__":
    main()
