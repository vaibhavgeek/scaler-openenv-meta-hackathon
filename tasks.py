"""Task definitions and graders for the pricing negotiation environment.

Three tasks with increasing difficulty:
  - easy:   Eager buyer (high WTP, low sensitivity, short patience)
  - medium: Neutral buyer (moderate WTP, moderate sensitivity)
  - hard:   Bargain hunter (low WTP, high sensitivity, patient)

Each grader scores an episode on a 0.0-1.0 scale combining:
  - Did the deal close? (conversion)
  - How much profit was captured? (revenue quality)
  - How efficiently was it closed? (turn efficiency)
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import List, Dict, Any, Optional

try:
    from .env import PricingEnv, PricingAction, Result
    from .buyer import BuyerState
except ImportError:
    from env import PricingEnv, PricingAction, Result
    from buyer import BuyerState


@dataclass
class TaskDef:
    name: str
    difficulty: str  # "easy" | "medium" | "hard"
    description: str
    list_price: float
    cost: float
    max_turns: int
    # Buyer overrides
    persona: str
    wtp_range: tuple[float, float]
    sensitivity_range: tuple[float, float]
    patience_range: tuple[int, int]
    # Grading weights
    conversion_weight: float = 0.4
    revenue_weight: float = 0.4
    efficiency_weight: float = 0.2


TASKS: List[TaskDef] = [
    TaskDef(
        name="eager_buyer",
        difficulty="easy",
        description="Negotiate with an eager buyer who has high willingness-to-pay "
                    "and low price sensitivity. Closing the deal should be straightforward; "
                    "the challenge is maximizing the sale price.",
        list_price=100.0,
        cost=40.0,
        max_turns=8,
        persona="eager",
        wtp_range=(0.90, 1.10),
        sensitivity_range=(3.0, 5.0),
        patience_range=(2, 4),
        conversion_weight=0.3,
        revenue_weight=0.5,
        efficiency_weight=0.2,
    ),
    TaskDef(
        name="neutral_buyer",
        difficulty="medium",
        description="Negotiate with a neutral buyer who has moderate willingness-to-pay. "
                    "Requires balancing price concessions with deal closure. "
                    "The buyer will walk if pushed too hard.",
        list_price=100.0,
        cost=50.0,
        max_turns=6,
        persona="neutral",
        wtp_range=(0.75, 1.00),
        sensitivity_range=(5.0, 7.0),
        patience_range=(3, 5),
        conversion_weight=0.4,
        revenue_weight=0.4,
        efficiency_weight=0.2,
    ),
    TaskDef(
        name="bargain_hunter",
        difficulty="hard",
        description="Negotiate with a bargain hunter who has low willingness-to-pay "
                    "and high price sensitivity. Margins are tight and time is short. "
                    "Closing above cost is the goal.",
        list_price=100.0,
        cost=60.0,
        max_turns=5,
        persona="bargain_hunter",
        wtp_range=(0.55, 0.80),
        sensitivity_range=(7.0, 10.0),
        patience_range=(4, 6),
        conversion_weight=0.5,
        revenue_weight=0.3,
        efficiency_weight=0.2,
    ),
]

TASK_MAP: Dict[str, TaskDef] = {t.name: t for t in TASKS}


def create_env_for_task(task: TaskDef, seed: int = 42) -> PricingEnv:
    """Create a PricingEnv configured for a specific task."""
    import random
    env = PricingEnv(
        list_price=task.list_price,
        cost=task.cost,
        max_turns=task.max_turns,
        seed=seed,
    )
    return env


def _force_buyer_persona(env: PricingEnv, task: TaskDef) -> None:
    """Override the randomly sampled buyer to match the task's persona constraints."""
    if env.buyer is None:
        return
    rng = env.rng
    wtp_mult = rng.uniform(*task.wtp_range)
    env.buyer.wtp = task.list_price * wtp_mult
    env.buyer.sensitivity = rng.uniform(*task.sensitivity_range)
    env.buyer.patience = rng.randint(*task.patience_range)
    env.buyer.persona = task.persona


def reset_for_task(env: PricingEnv, task: TaskDef) -> Result:
    """Reset the environment and force the buyer to match the task definition."""
    result = env.reset()
    _force_buyer_persona(env, task)
    return result


def grade_episode(
    task: TaskDef,
    sold: bool,
    final_price: Optional[float],
    turns_used: int,
    offers: Optional[List[Optional[float]]] = None,
) -> float:
    """Grade a completed episode for the given task. Returns score in [0.0, 1.0].

    Scoring (if sold):
      conversion_component = 1.0
      revenue_component    = profit / max_profit
      efficiency_component = (max_turns - turns + 1) / max_turns

    Scoring (if not sold, partial credit):
      negotiation_quality based on: offers above cost, in reasonable range,
      concession pattern. Capped at 25% of conversion_weight.

    Final score = weighted sum, clamped to [0, 1].
    """
    # Conversion
    conversion = 1.0 if sold else 0.0

    # Revenue quality
    if sold and final_price is not None:
        profit = max(0.0, final_price - task.cost)
        max_profit = max(task.list_price - task.cost, 1e-6)
        revenue = min(1.0, profit / max_profit)
    else:
        revenue = 0.0

    # Efficiency
    if sold and turns_used > 0:
        efficiency = max(0.0, (task.max_turns - turns_used + 1) / task.max_turns)
    else:
        efficiency = 0.0

    if sold:
        score = (
            task.conversion_weight * conversion
            + task.revenue_weight * revenue
            + task.efficiency_weight * efficiency
        )
    else:
        # Partial credit for failed episodes based on negotiation quality
        negotiation_quality = 0.0
        if offers:
            valid_offers = [o for o in offers if o is not None]
            if valid_offers:
                # Credit for offers above cost (valid pricing)
                above_cost = sum(1 for o in valid_offers if o >= task.cost) / len(valid_offers)
                # Credit for offers in reasonable range
                in_range = sum(
                    1 for o in valid_offers
                    if task.cost <= o <= task.list_price * 1.3
                ) / len(valid_offers)
                # Credit for showing concession (prices should decrease over time)
                concession = 0.0
                if len(valid_offers) >= 2:
                    decreasing = sum(
                        1 for i in range(len(valid_offers) - 1)
                        if valid_offers[i] >= valid_offers[i + 1]
                    )
                    concession = decreasing / (len(valid_offers) - 1)
                negotiation_quality = (above_cost + in_range + concession) / 3.0
        score = task.conversion_weight * 0.25 * negotiation_quality

    return min(max(score, 0.0), 1.0)


def run_task_episode(
    env: PricingEnv,
    task: TaskDef,
    get_action,
    seed: int = 42,
) -> Dict[str, Any]:
    """Run a full episode for a task and return graded results.

    Args:
        env: PricingEnv instance
        task: TaskDef for the episode
        get_action: callable(observation, info) -> str  (the agent's response)
        seed: random seed

    Returns:
        dict with keys: score, sold, final_price, turns_used, rewards
    """
    result = reset_for_task(env, task)
    turns_used = 0
    sold = False
    final_price = None
    rewards: List[float] = []
    offers: List[Optional[float]] = []

    while not result.done:
        action_text = get_action(result.observation, result.info)
        result = env.step(PricingAction(message=action_text))
        turns_used += 1
        rewards.append(result.reward)
        offers.append(result.info.get("offer"))

        if result.info.get("final_outcome") == "sold":
            sold = True
            final_price = result.info.get("offer")

    score = grade_episode(task, sold, final_price, turns_used, offers=offers)
    return {
        "task": task.name,
        "difficulty": task.difficulty,
        "score": score,
        "sold": sold,
        "final_price": final_price,
        "turns_used": turns_used,
        "rewards": rewards,
    }
