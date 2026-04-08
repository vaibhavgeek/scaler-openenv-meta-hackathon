"""Reward signals for pricing negotiation GRPO training.

Four reward components, each normalized roughly to [0, 1] so they can be
combined by GRPO without one dominating the others:

  - revenue   : captured revenue above cost, normalized by max possible margin
  - margin    : pure profit margin; prevents learning to cave to cost
  - conversion: did the deal close at all (0 or 1)
  - efficiency: fewer turns = higher score; encourages decisive negotiation
  - validity  : were all offers parseable and within [cost, 2*list_price]

Each function reads pre-computed per-episode values from kwargs, exactly like
the Wordle reward functions read `correct_reward`, `green_reward`, etc.
"""

from __future__ import annotations


def reward_revenue(completions, **kwargs):
    """Revenue captured above cost, normalized to [0, 1]."""
    values = kwargs.get("revenue_reward")
    if not values:
        return [0.0] * len(completions)
    return [float(v) for v in values]


def reward_margin(completions, **kwargs):
    """Profit margin ((price - cost) / cost), clipped to [0, 1]."""
    values = kwargs.get("margin_reward")
    if not values:
        return [0.0] * len(completions)
    return [float(v) for v in values]


def reward_conversion(completions, **kwargs):
    """1.0 if the deal closed, 0.0 otherwise."""
    values = kwargs.get("conversion_reward")
    if not values:
        return [0.0] * len(completions)
    return [float(v) for v in values]


def reward_efficiency(completions, **kwargs):
    """Fewer turns used = higher reward. Only counts if deal closed."""
    values = kwargs.get("efficiency_reward")
    if not values:
        return [0.0] * len(completions)
    return [float(v) for v in values]


def reward_validity(completions, **kwargs):
    """Fraction of turns where the model emitted a parseable, in-range offer."""
    values = kwargs.get("validity_reward")
    if not values:
        return [0.0] * len(completions)
    return [float(v) for v in values]


# ---------- per-episode computation ----------

def compute_episode_rewards(
    final_price: float | None,
    sold: bool,
    turns_used: int,
    max_turns: int,
    list_price: float,
    cost: float,
    valid_offer_count: int,
    total_offer_count: int,
) -> dict[str, float]:
    """Compute all five reward signals for one finished episode."""

    # Revenue: captured profit normalized by the max achievable profit (list - cost).
    if sold and final_price is not None:
        profit = max(0.0, final_price - cost)
        max_profit = max(list_price - cost, 1e-6)
        revenue = min(1.0, profit / max_profit)
    else:
        revenue = 0.0

    # Margin: (price - cost) / cost, clipped. Independent of list price so the
    # agent learns unit economics, not "anchor to list."
    if sold and final_price is not None:
        margin = max(0.0, min(1.0, (final_price - cost) / max(cost, 1e-6)))
    else:
        margin = 0.0

    # Conversion: binary.
    conversion = 1.0 if sold else 0.0

    # Efficiency: only meaningful if the deal closed. Rewards closing early.
    if sold and turns_used > 0:
        efficiency = max(0.0, (max_turns - turns_used + 1) / max_turns)
    else:
        efficiency = 0.0

    # Validity: parseable offers / total offers attempted.
    if total_offer_count > 0:
        validity = valid_offer_count / total_offer_count
    else:
        validity = 0.0

    return {
        "revenue_reward": revenue,
        "margin_reward": margin,
        "conversion_reward": conversion,
        "efficiency_reward": efficiency,
        "validity_reward": validity,
    }
