"""GRPO training for dynamic pricing negotiation.

Mirrors the structure of the Wordle GRPO example:
  1. Build the environment (in-process, no HTTP).
  2. Load tokenizer + define system prompt.
  3. Define rollout_once / rollout_func that play full episodes.
  4. Define multi-signal reward functions.
  5. Configure and run GRPOTrainer.

Run on an A100 (or similar). For quick smoke-tests on CPU, set
SMOKE_TEST=1 to skip training and just roll out a few episodes.
"""

from __future__ import annotations

import os
from collections import defaultdict

from transformers import AutoTokenizer
from datasets import Dataset

from pricing_env import (
    PricingEnv,
    PricingAction,
    extract_offer,
    compute_episode_rewards,
    reward_revenue,
    reward_margin,
    reward_conversion,
    reward_efficiency,
    reward_validity,
)


# ---------------------------------------------------------------------------
# 1. Environment
# ---------------------------------------------------------------------------

env = PricingEnv(list_price=100.0, cost=50.0, max_turns=6, seed=42)
sync_env = env.sync()
sync_env.connect()


# ---------------------------------------------------------------------------
# 2. Tokenizer + system prompt
# ---------------------------------------------------------------------------

MODEL_NAME = "Qwen/Qwen3-1.7B"

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


def build_user_prompt(observation) -> str:
    """Format the negotiation state into a prompt the model can read."""
    lines = [
        f"List price: ${observation.list_price:.2f}",
        f"Your cost floor: ${observation.cost:.2f}",
        f"Turn: {observation.turn + 1} / {observation.max_turns}",
        "",
        "Conversation so far:",
    ]
    if not observation.messages:
        lines.append("[BUYER] (waiting for your opening offer)")
    else:
        for m in observation.messages:
            lines.append(f"[{m.category}] {m.content}")
    lines.append("")
    lines.append("Reply with a single offer in the format: [offer: $X.XX] <justification>")
    return "\n".join(lines)


# ---------------------------------------------------------------------------
# 3. Rollout
# ---------------------------------------------------------------------------

def rollout_once(trainer, sync_env, tokenizer, system_prompt, max_turns=6):
    """Play one full negotiation episode and collect GRPO tensors."""
    from trl.experimental.openenv import generate_rollout_completions

    result = sync_env.reset()
    observation = result.observation

    prompt_ids: list[int] = []
    completion_ids: list[int] = []
    logprobs: list[float] = []

    final_price = None
    sold = False
    turns_used = 0
    valid_offer_count = 0
    total_offer_count = 0

    for _ in range(max_turns):
        if result.done:
            break

        user_prompt = build_user_prompt(observation)
        messages = [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt},
        ]
        prompt_text = tokenizer.apply_chat_template(
            messages,
            add_generation_prompt=True,
            tokenize=False,
            enable_thinking=False,
        )

        rollout_outputs = generate_rollout_completions(trainer, [prompt_text])[0]
        prompt_ids.extend(rollout_outputs["prompt_ids"])
        completion_ids.extend(rollout_outputs["completion_ids"])
        logprobs.extend(rollout_outputs["logprobs"])

        completion_text = rollout_outputs.get("text") or tokenizer.decode(
            rollout_outputs["completion_ids"], skip_special_tokens=True
        )

        # Track validity before the env re-parses it.
        total_offer_count += 1
        offer = extract_offer(completion_text)
        if offer is not None and env.cost <= offer <= 2 * env.list_price:
            valid_offer_count += 1

        result = sync_env.step(PricingAction(message=completion_text))
        observation = result.observation
        turns_used += 1

        if result.info.get("final_outcome") == "sold":
            final_price = result.info.get("offer")
            sold = True
            break
        if result.info.get("final_outcome") in ("walked", "timeout"):
            break

    rewards = compute_episode_rewards(
        final_price=final_price,
        sold=sold,
        turns_used=turns_used,
        max_turns=max_turns,
        list_price=env.list_price,
        cost=env.cost,
        valid_offer_count=valid_offer_count,
        total_offer_count=total_offer_count,
    )

    return {
        "prompt_ids": prompt_ids,
        "completion_ids": completion_ids,
        "logprobs": logprobs,
        **rewards,
    }


def rollout_func(prompts, trainer=None):
    """GRPOTrainer entry point. One episode per prompt in the batch."""
    batch = defaultdict(list)

    for _ in prompts:
        ep = rollout_once(
            trainer=trainer,
            sync_env=sync_env,
            tokenizer=tokenizer,
            system_prompt=SYSTEM_PROMPT,
            max_turns=env.max_turns,
        )
        batch["prompt_ids"].append(ep["prompt_ids"])
        batch["completion_ids"].append(ep["completion_ids"])
        batch["logprobs"].append(ep["logprobs"])
        for k in ("revenue_reward", "margin_reward", "conversion_reward",
                  "efficiency_reward", "validity_reward"):
            batch[k].append(ep[k])

    return dict(batch)


# ---------------------------------------------------------------------------
# 4. Main
# ---------------------------------------------------------------------------

def main():
    global tokenizer

    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
    tokenizer.pad_token = tokenizer.eos_token

    # Dataset is just a placeholder — the env, not the prompts, drives episodes.
    dataset = Dataset.from_dict({"prompt": ["Negotiate a sale."] * 1000})

    if os.environ.get("SMOKE_TEST"):
        print("SMOKE_TEST=1 — skipping GRPOTrainer, running deterministic env check.")
        _smoke_test()
        return

    from trl import GRPOConfig, GRPOTrainer

    output_dir = "pricing-grpo-Qwen3-1.7B"
    grpo_config = GRPOConfig(
        num_train_epochs=1,
        learning_rate=5e-6,
        gradient_accumulation_steps=32,
        per_device_train_batch_size=1,
        warmup_steps=20,
        num_generations=4,              # group size for GRPO advantage
        max_completion_length=128,      # enough for "[offer: $X] <justification>"
        max_prompt_length=1024,
        use_vllm=True,
        vllm_mode="colocate",
        vllm_gpu_memory_utilization=0.3,
        output_dir=output_dir,
        report_to="trackio",
        trackio_space_id=output_dir,
        logging_steps=1,
        save_steps=20,
        gradient_checkpointing=True,
        gradient_checkpointing_kwargs={"use_reentrant": False},
        push_to_hub=False,
    )

    trainer = GRPOTrainer(
        model=MODEL_NAME,
        processing_class=tokenizer,
        reward_funcs=[
            reward_revenue,
            reward_margin,
            reward_conversion,
            reward_efficiency,
            reward_validity,
        ],
        train_dataset=dataset,
        args=grpo_config,
        rollout_func=rollout_func,
    )

    trainer.train()
    trainer.save_model(output_dir)
    sync_env.close()


def _smoke_test():
    """Env-only sanity check: play a few random episodes with fixed offers."""
    import random
    rng = random.Random(0)
    for ep in range(3):
        result = sync_env.reset()
        print(f"\n--- Episode {ep + 1} ---")
        print(f"(hidden WTP: ${sync_env.buyer.wtp:.2f}, persona: {sync_env.buyer.persona})")
        for msg in result.observation.messages:
            print(f"  [{msg.category}] {msg.content}")

        # Dumb strategy: start at list, concede 15% each turn.
        offer = env.list_price
        while not result.done:
            action_text = f"[offer: ${offer:.2f}] Best I can do."
            result = sync_env.step(PricingAction(message=action_text))
            print(f"  [AGENT] offer ${offer:.2f}")
            for msg in result.observation.messages[-1:]:
                print(f"  [{msg.category}] {msg.content}")
            offer *= 0.85

        print(f"  outcome: {result.info.get('final_outcome')} reward={result.reward}")

    sync_env.close()


if __name__ == "__main__":
    # The tokenizer is loaded inside main(); smoke test doesn't need it.
    tokenizer = None
    main()
