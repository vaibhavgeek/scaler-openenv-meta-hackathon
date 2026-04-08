"""
Inference Script — Pricing Negotiation OpenEnv
===================================
MANDATORY
- Before submitting, ensure the following variables are defined in your environment configuration:
    API_BASE_URL   The API endpoint for the LLM.
    MODEL_NAME     The model identifier to use for inference.
    HF_TOKEN       Your Hugging Face / API key.
    LOCAL_IMAGE_NAME The name of the local image to use for the environment if you are using from_docker_image()

- Defaults are set only for API_BASE_URL and MODEL_NAME:
    API_BASE_URL = os.getenv("API_BASE_URL", "https://generativelanguage.googleapis.com/v1beta/openai/")
    MODEL_NAME = os.getenv("MODEL_NAME", "gemini-2.0-flash")

STDOUT FORMAT
- The script must emit exactly three line types to stdout, in this order:

    [START] task=<task_name> env=<benchmark> model=<model_name>
    [STEP]  step=<n> action=<action_str> reward=<0.00> done=<true|false> error=<msg|null>
    [END]   success=<true|false> steps=<n> score=<score> rewards=<r1,r2,...,rn>
"""

import asyncio
import os
import sys
import textwrap
from typing import List, Optional

from openai import OpenAI

# --- local imports (support running from inside the package dir or as a module) ---
_here = os.path.dirname(os.path.abspath(__file__))
if _here not in sys.path:
    sys.path.insert(0, _here)

try:
    from env import PricingEnv, PricingAction, Observation
    from tasks import TASKS, TASK_MAP, TaskDef, reset_for_task, grade_episode
except ImportError:
    from pricing_env.env import PricingEnv, PricingAction, Observation
    from pricing_env.tasks import TASKS, TASK_MAP, TaskDef, reset_for_task, grade_episode

# ---------- config from env vars (strictly per spec) ----------

API_KEY = os.getenv("HF_TOKEN")  # no default — mandatory
API_BASE_URL = os.getenv("API_BASE_URL") or "https://generativelanguage.googleapis.com/v1beta/openai/"
MODEL_NAME = os.getenv("MODEL_NAME") or "gemini-2.0-flash"
LOCAL_IMAGE_NAME = os.getenv("LOCAL_IMAGE_NAME")  # required only if using Docker

TASK_NAME = os.getenv("PRICING_ENV_TASK", "eager_buyer")
BENCHMARK = "pricing-negotiation"
MAX_STEPS = 8  # matches max_turns of easiest task
TEMPERATURE = 0.3
MAX_TOKENS = 256
SUCCESS_SCORE_THRESHOLD = 0.5

# Max possible reward: ~1.0 normalized per step, up to 8 steps
MAX_TOTAL_REWARD = 8.0

SYSTEM_PROMPT = textwrap.dedent("""
    You are an expert sales negotiator. Your goal is to maximize revenue while closing deals.

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
""").strip()

TASK_GUIDANCE = {
    "eager_buyer": (
        "\n\n## TASK CONTEXT\n"
        "This buyer is EAGER with high willingness-to-pay and low price sensitivity. "
        "Anchor firmly at or near list price. Concede slowly — "
        "they will likely accept a price well above your cost. "
        "Prioritize maximizing sale price over closing speed."
    ),
    "neutral_buyer": (
        "\n\n## TASK CONTEXT\n"
        "This buyer is MODERATE. They will negotiate but walk if pushed too hard. "
        "Start near list price but be prepared to concede 15-25%. "
        "Balance price maximization with deal closure risk."
    ),
    "bargain_hunter": (
        "\n\n## TASK CONTEXT\n"
        "This buyer is a BARGAIN HUNTER with low willingness-to-pay and high sensitivity. "
        "Your margin is tight — cost floor is high relative to list price. "
        "Concede strategically but protect your margin. "
        "Closing above cost is more important than maximizing price."
    ),
}


def get_task_prompt(task_name: str) -> str:
    """Return SYSTEM_PROMPT with task-specific strategy appended."""
    return SYSTEM_PROMPT + TASK_GUIDANCE.get(task_name, "")


# ---------- logging helpers (exact [START]/[STEP]/[END] format) ----------

def log_start(task: str, env: str, model: str) -> None:
    print(f"[START] task={task} env={env} model={model}", flush=True)


def log_step(step: int, action: str, reward: float, done: bool, error: Optional[str]) -> None:
    error_val = error if error else "null"
    done_val = str(done).lower()
    print(
        f"[STEP] step={step} action={action} reward={reward:.2f} done={done_val} error={error_val}",
        flush=True,
    )


def log_end(success: bool, steps: int, score: float, rewards: List[float]) -> None:
    rewards_str = ",".join(f"{r:.2f}" for r in rewards)
    print(
        f"[END] success={str(success).lower()} steps={steps} score={score:.2f} rewards={rewards_str}",
        flush=True,
    )


# ---------- agent ----------

def build_user_prompt(step: int, obs: Observation, last_reward: float, history: List[str]) -> str:
    history_block = "\n".join(history[-4:]) if history else "None"
    conversation = ""
    if obs.messages:
        conversation = "\n".join(f"[{m.category}] {m.content}" for m in obs.messages)
    else:
        conversation = "[BUYER] (waiting for your opening offer)"

    return textwrap.dedent(f"""
        Step: {step}
        List price: ${obs.list_price:.2f}
        Your cost floor: ${obs.cost:.2f}
        Turn: {obs.turn} / {obs.max_turns}
        Last reward: {last_reward:.2f}

        Conversation:
        {conversation}

        Previous steps:
        {history_block}

        Send your next offer in the format: [offer: $X.XX] <justification>
    """).strip()


def get_model_message(
    client: OpenAI,
    step: int,
    obs: Observation,
    last_reward: float,
    history: List[str],
    task_prompt: Optional[str] = None,
) -> str:
    user_prompt = build_user_prompt(step, obs, last_reward, history)
    try:
        completion = client.chat.completions.create(
            model=MODEL_NAME,
            messages=[
                {"role": "system", "content": task_prompt or SYSTEM_PROMPT},
                {"role": "user", "content": user_prompt},
            ],
            temperature=TEMPERATURE,
            max_tokens=MAX_TOKENS,
            stream=False,
        )
        text = (completion.choices[0].message.content or "").strip()
        return text if text else "[offer: $75.00] Let's make a deal."
    except Exception as exc:
        print(f"[DEBUG] Model request failed: {exc}", flush=True)
        return "[offer: $75.00] Let's make a deal."


# ---------- run one task ----------

async def run_task(client: OpenAI, task: TaskDef, seed: int = 42) -> dict:
    env = PricingEnv(
        list_price=task.list_price,
        cost=task.cost,
        max_turns=task.max_turns,
        seed=seed,
    )

    history: List[str] = []
    rewards: List[float] = []
    steps_taken = 0
    score = 0.0
    success = False

    log_start(task=task.name, env=BENCHMARK, model=MODEL_NAME)

    try:
        result = reset_for_task(env, task)
        last_reward = 0.0
        task_prompt = get_task_prompt(task.name)

        for step in range(1, MAX_STEPS + 1):
            if result.done:
                break

            message = get_model_message(client, step, result.observation, last_reward, history, task_prompt=task_prompt)
            result = env.step(PricingAction(message=message))

            reward = result.reward or 0.0
            done = result.done
            error = result.info.get("last_action_error") if hasattr(result.info, 'get') else None

            rewards.append(reward)
            steps_taken = step
            last_reward = reward

            log_step(step=step, action=message, reward=reward, done=done, error=error)

            history.append(f"Step {step}: {message!r} -> reward {reward:+.2f}")

            if done:
                break

        score = sum(rewards) / MAX_TOTAL_REWARD if MAX_TOTAL_REWARD > 0 else 0.0
        score = min(max(score, 0.0), 1.0)  # clamp to [0, 1]
        success = score >= SUCCESS_SCORE_THRESHOLD

    finally:
        try:
            env.close()
        except Exception as e:
            print(f"[DEBUG] env.close() error: {e}", flush=True)
        log_end(success=success, steps=steps_taken, score=score, rewards=rewards)

    return {
        "task": task.name,
        "difficulty": task.difficulty,
        "score": score,
        "success": success,
        "steps": steps_taken,
    }


# ---------- main ----------

async def main() -> None:
    client = OpenAI(base_url=API_BASE_URL, api_key=API_KEY)

    results = []
    for task in TASKS:
        result = await run_task(client, task, seed=42)
        results.append(result)


if __name__ == "__main__":
    asyncio.run(main())
