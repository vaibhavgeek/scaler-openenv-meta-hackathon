"""Rule-based buyer simulator for pricing negotiation.

The buyer has a hidden willingness-to-pay (WTP), patience (max turns before
walking), and price sensitivity. Acceptance follows a logistic curve:
    P(accept | price) = sigmoid(sensitivity * (WTP - price) / WTP)

Counter-offers are drawn as a fraction of the current offer, pulled toward WTP.
"""

from __future__ import annotations

import math
import random
from typing import Literal, Optional, List, Dict

from pydantic import BaseModel, Field


BuyerAction = Literal["accept", "reject_walk", "counter"]


class BuyerState(BaseModel):
    wtp: float                                    # hidden willingness to pay
    patience: int                                 # turns remaining before walking
    sensitivity: float = 6.0                      # logistic steepness; higher = pickier
    persona: str = "neutral"                      # flavors the generated text
    last_counter: Optional[float] = None
    history: List[Dict] = Field(default_factory=list)

    class Config:
        arbitrary_types_allowed = True


class BuyerResponse(BaseModel):
    action: BuyerAction
    counter_price: Optional[float] = None
    message: str                                  # natural-language reply for the agent to read


def sample_buyer(rng: random.Random, list_price: float) -> BuyerState:
    """Sample a buyer from a realistic distribution.

    WTP is centered slightly below list price so most deals are possible
    but not all. Patience and sensitivity vary across personas.
    """
    persona = rng.choice(["bargain_hunter", "neutral", "eager", "tire_kicker"])

    if persona == "bargain_hunter":
        wtp_mult = rng.uniform(0.55, 0.80)
        patience = rng.randint(4, 6)
        sensitivity = rng.uniform(7.0, 10.0)
    elif persona == "eager":
        wtp_mult = rng.uniform(0.90, 1.10)
        patience = rng.randint(2, 4)
        sensitivity = rng.uniform(3.0, 5.0)
    elif persona == "tire_kicker":
        wtp_mult = rng.uniform(0.40, 0.65)
        patience = rng.randint(3, 5)
        sensitivity = rng.uniform(8.0, 12.0)
    else:  # neutral
        wtp_mult = rng.uniform(0.75, 1.00)
        patience = rng.randint(3, 5)
        sensitivity = rng.uniform(5.0, 7.0)

    return BuyerState(
        wtp=list_price * wtp_mult,
        patience=patience,
        sensitivity=sensitivity,
        persona=persona,
    )


def _sigmoid(x: float) -> float:
    if x < -30:
        return 0.0
    if x > 30:
        return 1.0
    return 1.0 / (1.0 + math.exp(-x))


def buyer_step(
    buyer: BuyerState,
    offer: float,
    rng: random.Random,
) -> BuyerResponse:
    """Run one buyer decision given the agent's offered price."""
    buyer.patience -= 1

    # Acceptance probability is logistic in normalized surplus.
    surplus = (buyer.wtp - offer) / max(buyer.wtp, 1e-6)
    p_accept = _sigmoid(buyer.sensitivity * surplus)

    # Out-of-patience buyers either accept (if tempted) or walk.
    if buyer.patience <= 0:
        if rng.random() < p_accept:
            return BuyerResponse(action="accept", counter_price=None, message=_accept_text(buyer, offer))
        return BuyerResponse(action="reject_walk", counter_price=None, message=_walk_text(buyer))

    # Clear accept.
    if rng.random() < p_accept:
        return BuyerResponse(action="accept", counter_price=None, message=_accept_text(buyer, offer))

    # Clear walk if offer is absurdly above WTP.
    if offer > buyer.wtp * 1.5:
        return BuyerResponse(action="reject_walk", counter_price=None, message=_walk_text(buyer))

    # Otherwise, counter-offer. Pull toward WTP with some noise.
    target = buyer.wtp * rng.uniform(0.85, 0.98)
    # Don't counter above the agent's last offer (that would be silly).
    counter = min(target, offer * rng.uniform(0.75, 0.92))
    counter = max(counter, 1.0)
    buyer.last_counter = counter
    return BuyerResponse(action="counter", counter_price=counter, message=_counter_text(buyer, offer, counter))


# ---------- flavor text (optional but keeps the conversation readable) ----------

def _accept_text(buyer: BuyerState, offer: float) -> str:
    msgs = {
        "bargain_hunter": f"Alright, ${offer:.2f} works. Deal.",
        "eager": f"Perfect, I'll take it at ${offer:.2f}!",
        "tire_kicker": f"Fine, ${offer:.2f}. You've got a deal.",
        "neutral": f"Okay, ${offer:.2f} sounds fair. Let's do it.",
    }
    return msgs.get(buyer.persona, msgs["neutral"])


def _walk_text(buyer: BuyerState) -> str:
    msgs = {
        "bargain_hunter": "That's way over my budget. I'm out.",
        "eager": "Hmm, that's not quite what I was hoping for. I'll pass.",
        "tire_kicker": "Not interested at that price. Goodbye.",
        "neutral": "Sorry, that doesn't work for me. I'll look elsewhere.",
    }
    return msgs.get(buyer.persona, msgs["neutral"])


def _counter_text(buyer: BuyerState, offer: float, counter: float) -> str:
    msgs = {
        "bargain_hunter": f"${offer:.2f} is too steep. How about ${counter:.2f}?",
        "eager": f"I like it, but ${offer:.2f} is a stretch. Could you do ${counter:.2f}?",
        "tire_kicker": f"Not at ${offer:.2f}. I could maybe do ${counter:.2f}.",
        "neutral": f"${offer:.2f} is a bit high. Would you accept ${counter:.2f}?",
    }
    return msgs.get(buyer.persona, msgs["neutral"])
