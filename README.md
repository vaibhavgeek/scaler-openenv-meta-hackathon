---
title: Pricing Negotiation OpenEnv
emoji: 💰
colorFrom: blue
colorTo: green
sdk: docker
app_port: 7860
tags:
  - openenv
---

# Pricing Negotiation OpenEnv

A real-world pricing negotiation environment where an AI agent learns to negotiate
product prices with simulated buyers of varying personas and willingness-to-pay.

## Why This Matters

Voice agents and AI sales agents are no longer a futuristic concept — they're here today. Every Shopify store owner, from a one-person candle brand to a growing DTC fashion label, leaves money on the table because they sell at fixed prices. Meanwhile, premium buyers are willing to pay more, and bargain hunters will walk unless they feel they got a deal.

We're building an AI negotiation agent that lets even the smallest Shopify store capture more revenue by enabling real-time price haggling — managed entirely by agents. Rich, lazy, or impulsive buyers who don't bother negotiating pay list price. Deal-seekers get a fair discount and still convert. The store owner earns more on every transaction without lifting a finger.

**Our intent is to launch this as a Shopify App Store app** — a plug-and-play AI sales agent that any merchant can install in minutes.

## How It Works — Architecture in 3 Steps

1. **Buyer walks in, agent reads the room.** When a customer initiates a price conversation (via chat widget, voice, or storefront prompt), the environment resets with the product's list price and cost floor. A buyer persona is detected or sampled — eager, neutral, bargain hunter — each with a hidden willingness-to-pay and patience level. The agent receives task-specific strategic guidance tailored to the buyer type.

2. **Multi-turn negotiation loop.** The AI agent proposes a price (`[offer: $X.XX]`), the buyer responds (accept, counter-offer, or walk away). Each turn, the agent observes the full conversation history, the buyer's tone, and remaining turns. It decides how much to concede — or whether to hold firm. The environment provides dense per-step reward signals: positive for valid offers that keep negotiation alive (+0.05), penalties for malformed offers (-0.1) and below-cost offers (-0.2), and a normalized profit reward on sale closure.

3. **Deal closes, rewards flow.** If the buyer accepts, the sale is recorded and scored as a normalized profit ratio in [0, 1]. If the buyer walks, the agent still receives partial credit for good negotiation quality — offers above cost, reasonable pricing, and proper concession patterns. Over thousands of episodes, the agent learns which personas to push, which to discount, and when to close fast.

## How to Further Improve This Model

- **Train with GRPO on real transaction data** — Replace synthetic buyer personas with behavioral distributions mined from actual Shopify store transactions (cart abandonment rates, price sensitivity by product category, time-of-day patterns) to make the agent's strategy reflect real customer behavior.
- **Add an LLM-powered buyer** — Swap the rule-based buyer simulator with a second language model whose system prompt encodes WTP + persona. This creates a self-play loop where both sides improve, producing agents that handle free-form negotiation language, not just parsed offers.
- **Multi-product and inventory-aware pricing** — Extend the observation space with inventory levels, demand multipliers, and cross-sell opportunities so the agent learns dynamic pricing (e.g., discount slow movers aggressively, hold firm on best-sellers running low).
- **Deploy with voice integration** — Connect the agent to a voice-to-text pipeline (Whisper + TTS) so customers can literally haggle out loud on the storefront, creating a differentiated shopping experience that no competitor offers.

## Environment Description

The agent acts as a sales negotiator selling a product with a known list price and
cost floor. Each episode, a buyer with a hidden willingness-to-pay (WTP) engages in
a multi-turn price negotiation. The agent must balance maximizing revenue with the
risk of the buyer walking away.

This models a genuine business task: sales pricing optimization. Unlike toy
environments, the buyer has realistic behavioral patterns (logistic acceptance
curves, persona-driven counter-offers, patience limits).

## Action Space

| Field     | Type   | Description                                                    |
|-----------|--------|----------------------------------------------------------------|
| `message` | string | Agent's response containing `[offer: $X.XX]` + justification  |

The environment parses `[offer: $X.XX]` from the agent's message. Malformed
offers receive a -0.1 reward penalty. Offers below cost receive -0.2.

## Observation Space

| Field        | Type          | Description                                    |
|--------------|---------------|------------------------------------------------|
| `prompt`     | string        | System context with list price, cost, turn info|
| `messages`   | list[Message] | Full conversation history (BUYER/AGENT turns)  |
| `list_price` | float         | Product list price                             |
| `cost`       | float         | Agent's cost floor (minimum sale price)        |
| `turn`       | int           | Current turn number                            |
| `max_turns`  | int           | Maximum turns before episode ends              |

## Tasks

Three tasks with genuinely different difficulty through varied margins, turn limits, and buyer behavior:

### 1. `eager_buyer` (Easy)
- **Buyer**: High WTP (90-110% of list), low sensitivity, impatient
- **Environment**: Cost floor $40, 8 turns (wide margin, forgiving)
- **Challenge**: Maximizing sale price (closing is easy)
- **Scoring**: 30% conversion + 50% revenue + 20% efficiency
- **Agent strategy**: Anchor firmly at list price, concede slowly

### 2. `neutral_buyer` (Medium)
- **Buyer**: Moderate WTP (75-100% of list), moderate sensitivity
- **Environment**: Cost floor $50, 6 turns (standard margin)
- **Challenge**: Balancing concessions with closure probability
- **Scoring**: 40% conversion + 40% revenue + 20% efficiency
- **Agent strategy**: Start near list, be prepared to concede 15-25%

### 3. `bargain_hunter` (Hard)
- **Buyer**: Low WTP (55-80% of list), high sensitivity, patient
- **Environment**: Cost floor $60, 5 turns (tight margin, time pressure)
- **Challenge**: Closing above cost at all; margins are razor-thin
- **Scoring**: 50% conversion + 30% revenue + 20% efficiency
- **Agent strategy**: Protect margin, closing above cost matters more than maximizing price

All graders produce scores in [0.0, 1.0]. Failed episodes receive partial credit
based on negotiation quality (offers above cost, reasonable range, concession pattern).

## Reward Function

**Per-step reward shaping** (dense signal, not just end-of-episode):

| Event | Reward | Signal |
|-------|--------|--------|
| Valid offer, negotiation continues | +0.05 | Encourages productive dialogue |
| Malformed offer (no `[offer: $X.XX]`) | -0.1 | Penalizes unparseable output |
| Offer below cost floor | -0.2 | Penalizes strategic errors |
| Sale closes | `(price - cost) / (list_price - cost)` | Normalized profit in [0, 1] |
| Buyer walks / timeout | 0.0 | No reward for failed close |

**Episode grading** combines three weighted signals:
- **Conversion**: 1.0 if sold, partial credit for good negotiation if not
- **Revenue quality**: profit / max_possible_profit
- **Efficiency**: (max_turns - turns_used + 1) / max_turns (if sold)

The GRPO training pipeline uses 5 finer-grained signals:
revenue, margin, conversion, efficiency, validity.

## Buyer Personas

| Persona         | WTP Range    | Sensitivity | Patience |
|-----------------|-------------|-------------|----------|
| eager           | 90-110%     | 3-5         | 2-4 turns|
| neutral         | 75-100%     | 5-7         | 3-5 turns|
| bargain_hunter  | 55-80%      | 7-10        | 4-6 turns|
| tire_kicker     | 40-65%      | 8-12        | 3-5 turns|

Acceptance probability: `sigmoid(sensitivity * (WTP - price) / WTP)`

Counter-offers are placed in the gap between the buyer's target price and the
agent's last offer, ensuring realistic negotiation dynamics.

## Setup & Usage

### Quick Start (Local)

```bash
# Install dependencies
pip install -r requirements.txt

# Run environment sanity check
python smoke_test.py

# Run baseline inference
export API_BASE_URL="https://generativelanguage.googleapis.com/v1beta/openai/"
export MODEL_NAME="gemini-2.0-flash"
export HF_TOKEN="your-api-key"
python inference.py
```

### Docker

```bash
docker build -t pricing-negotiation .
docker run -p 7860:7860 pricing-negotiation
```

### API Endpoints

```
GET  /          -> health check
GET  /tasks     -> list available tasks
POST /reset     -> reset environment (random buyer)
POST /reset/{task_name} -> reset with specific task
POST /step      -> {"message": "[offer: $80.00] ..."}
GET  /state     -> current environment state
```

### HF Spaces

Deploy as a Docker Space tagged with `openenv`. The Dockerfile exposes
port 7860 and runs the FastAPI server.

## Baseline Scores

Baselines with `gemini-2.0-flash` (normalized scores via `sum(rewards)/MAX_TOTAL_REWARD`):

| Task            | Difficulty | Cost | Max Turns | Score |
|-----------------|-----------|------|-----------|-------|
| eager_buyer     | easy      | $40  | 8         | 0.11  |
| neutral_buyer   | medium    | $50  | 6         | 0.11  |
| bargain_hunter  | hard      | $60  | 5         | 0.08  |

## Project Structure

```
pricing_env/
├── __init__.py        # exports
├── buyer.py           # rule-based buyer with hidden WTP
├── env.py             # OpenEnv-compatible PricingEnv with reward shaping
├── rewards.py         # 5 reward signals for GRPO training
├── tasks.py           # 3 task definitions + graders with partial credit
├── app.py             # FastAPI server for HF Spaces
├── server/
│   └── app.py         # server entry point (project.scripts)
├── inference.py       # baseline inference script (OpenAI client)
├── openenv.yaml       # OpenEnv metadata
├── pyproject.toml     # package config with openenv-core dependency
├── uv.lock            # locked dependencies
├── Dockerfile         # container deployment
├── requirements.txt   # Python dependencies
├── smoke_test.py      # env sanity check
├── train_grpo.py      # GRPO training script
└── README.md
```
