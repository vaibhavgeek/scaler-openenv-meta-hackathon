"""Pricing negotiation environment.

Implements the OpenEnv interface: reset(), step(), state() return Result objects
with typed Pydantic models for Observation, Action, and Reward.
"""

from __future__ import annotations

import random
import re
from typing import Any, Optional, List, Dict

from pydantic import BaseModel, Field

try:
    from .buyer import BuyerState, BuyerResponse, sample_buyer, buyer_step
except ImportError:
    from buyer import BuyerState, BuyerResponse, sample_buyer, buyer_step


# ---------- Pydantic models (OpenEnv spec) ----------

class Message(BaseModel):
    category: str  # "SYSTEM" | "BUYER" | "AGENT"
    content: str


class Observation(BaseModel):
    prompt: str = ""
    messages: List[Message] = Field(default_factory=list)
    list_price: float = 0.0
    cost: float = 0.0
    turn: int = 0
    max_turns: int = 0


class PricingAction(BaseModel):
    message: str  # raw model output; env parses [offer: X]


class Reward(BaseModel):
    value: float = 0.0
    revenue: float = 0.0
    margin: float = 0.0
    conversion: float = 0.0
    efficiency: float = 0.0
    validity: float = 0.0


class Result(BaseModel):
    observation: Observation
    reward: float = 0.0
    done: bool = False
    info: Dict[str, Any] = Field(default_factory=dict)


class EnvState(BaseModel):
    list_price: float = 0.0
    cost: float = 0.0
    turn: int = 0
    max_turns: int = 0
    done: bool = False
    sold_price: Optional[float] = None
    messages: List[Message] = Field(default_factory=list)
    buyer_persona: Optional[str] = None


# ---------- action parsing ----------

_OFFER_RE = re.compile(r"\[\s*offer\s*:\s*\$?\s*([0-9]+(?:\.[0-9]+)?)\s*\]", re.I)


def extract_offer(text: str) -> float | None:
    """Pull [offer: X] out of the model's response. Returns None if malformed."""
    if not text:
        return None
    match = _OFFER_RE.search(text)
    if not match:
        return None
    try:
        return float(match.group(1))
    except ValueError:
        return None


# ---------- environment ----------

class PricingEnv:
    """Single-buyer negotiation environment.

    Each episode:
      - A new buyer is sampled with a hidden WTP.
      - The agent proposes prices across up to `max_turns` turns.
      - The episode ends on buyer accept, buyer walk, or turn exhaustion.
    """

    def __init__(
        self,
        list_price: float = 100.0,
        cost: float = 50.0,
        max_turns: int = 6,
        seed: int | None = None,
    ):
        self.list_price = list_price
        self.cost = cost
        self.max_turns = max_turns
        self.rng = random.Random(seed)

        self.buyer: BuyerState | None = None
        self.messages: list[Message] = []
        self.turn: int = 0
        self.done: bool = False
        self.sold_price: float | None = None

    # OpenEnv-compatible context-manager / sync hooks (no-ops for in-process env).
    def sync(self):
        return self

    def connect(self):
        pass

    def close(self):
        pass

    def __enter__(self):
        return self

    def __exit__(self, *exc):
        self.close()

    # ---- core API ----

    def reset(self) -> Result:
        self.buyer = sample_buyer(self.rng, self.list_price)
        self.turn = 0
        self.done = False
        self.sold_price = None

        opener = self._opening_message()
        self.messages = [Message(category="BUYER", content=opener)]

        obs = Observation(
            prompt=self._system_context(),
            messages=list(self.messages),
            list_price=self.list_price,
            cost=self.cost,
            turn=0,
            max_turns=self.max_turns,
        )
        return Result(observation=obs, reward=0.0, done=False, info={})

    def step(self, action: PricingAction) -> Result:
        if self.done or self.buyer is None:
            raise RuntimeError("Call reset() before step() or after episode ends.")

        self.turn += 1
        offer = extract_offer(action.message)

        # Malformed action: penalize but keep the episode alive.
        if offer is None:
            self.messages.append(Message(category="AGENT", content=action.message))
            self.messages.append(
                Message(category="BUYER", content="I didn't catch a price there. Could you be specific?")
            )
            info = {"valid_offer": False, "offer": None, "buyer_action": "malformed"}
            if self.turn >= self.max_turns:
                self.done = True
                info["final_outcome"] = "timeout"
            return Result(
                observation=self._make_observation(),
                reward=-0.1,
                done=self.done,
                info=info,
            )

        # Record agent's offer.
        self.messages.append(Message(category="AGENT", content=f"[offer: ${offer:.2f}] {action.message}"))

        # Penalize below-cost offers immediately.
        if offer < self.cost:
            info: Dict[str, Any] = {
                "valid_offer": True,
                "offer": offer,
                "buyer_action": "below_cost",
                "buyer_wtp": self.buyer.wtp,
                "counter_price": None,
            }
            self.messages.append(
                Message(category="BUYER", content="That seems surprisingly low... are you sure?")
            )
            if self.turn >= self.max_turns:
                self.done = True
                info["final_outcome"] = "timeout"
            return Result(
                observation=self._make_observation(),
                reward=-0.2,
                done=self.done,
                info=info,
            )

        # Buyer responds.
        response: BuyerResponse = buyer_step(self.buyer, offer, self.rng)
        self.messages.append(Message(category="BUYER", content=response.message))

        info: Dict[str, Any] = {
            "valid_offer": True,
            "offer": offer,
            "buyer_action": response.action,
            "buyer_wtp": self.buyer.wtp,
            "counter_price": response.counter_price,
        }

        if response.action == "accept":
            self.done = True
            self.sold_price = offer
            info["final_outcome"] = "sold"
            # Normalized reward: profit ratio in [0, 1]
            margin = max(self.list_price - self.cost, 1e-6)
            normalized_reward = max(0.0, (offer - self.cost) / margin)
            return Result(observation=self._make_observation(), reward=normalized_reward, done=True, info=info)

        if response.action == "reject_walk":
            self.done = True
            info["final_outcome"] = "walked"
            return Result(observation=self._make_observation(), reward=0.0, done=True, info=info)

        # Counter-offer: episode continues.
        if self.turn >= self.max_turns:
            self.done = True
            info["final_outcome"] = "timeout"
            return Result(observation=self._make_observation(), reward=0.0, done=True, info=info)

        # Valid offer that keeps negotiation alive — small positive signal.
        return Result(observation=self._make_observation(), reward=0.05, done=False, info=info)

    def state(self) -> EnvState:
        """Return the current environment state (OpenEnv spec)."""
        return EnvState(
            list_price=self.list_price,
            cost=self.cost,
            turn=self.turn,
            max_turns=self.max_turns,
            done=self.done,
            sold_price=self.sold_price,
            messages=list(self.messages),
            buyer_persona=self.buyer.persona if self.buyer else None,
        )

    # ---- helpers ----

    def _system_context(self) -> str:
        return (
            f"You are a sales agent negotiating with a single buyer.\n"
            f"Product list price: ${self.list_price:.2f}\n"
            f"Your cost floor: ${self.cost:.2f} (never sell below this)\n"
            f"You have {self.max_turns} turns to close the deal."
        )

    def _opening_message(self) -> str:
        openers = [
            "Hi, I'm interested in this item. What's the best price you can offer?",
            "Hello — is there any flexibility on the price?",
            "Hey there. I'd like to buy this, but the list price is a bit high. Can we talk?",
            "Interested in this. What can you do on price?",
        ]
        return self.rng.choice(openers)

    def _make_observation(self) -> Observation:
        return Observation(
            prompt=self._system_context(),
            messages=list(self.messages),
            list_price=self.list_price,
            cost=self.cost,
            turn=self.turn,
            max_turns=self.max_turns,
        )
