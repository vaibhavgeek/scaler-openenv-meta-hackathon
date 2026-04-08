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
offers are penalized but don't end the episode.

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

Three tasks with increasing difficulty:

### 1. `eager_buyer` (Easy)
- **Buyer**: High WTP (90-110% of list), low sensitivity, impatient
- **Challenge**: Maximizing sale price (closing is easy)
- **Scoring**: 30% conversion + 50% revenue + 20% efficiency

### 2. `neutral_buyer` (Medium)
- **Buyer**: Moderate WTP (75-100% of list), moderate sensitivity
- **Challenge**: Balancing concessions with closure probability
- **Scoring**: 40% conversion + 40% revenue + 20% efficiency

### 3. `bargain_hunter` (Hard)
- **Buyer**: Low WTP (55-80% of list), high sensitivity, patient
- **Challenge**: Closing above cost at all; requires strategic discounting
- **Scoring**: 50% conversion + 30% revenue + 20% efficiency

All graders produce scores in [0.0, 1.0].

## Reward Function

Per-step reward is the sale price on deal close, 0.0 otherwise.
Episode grading combines three signals (all [0, 1]):

- **Conversion**: 1.0 if sold, 0.0 otherwise
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

## Setup & Usage

### Quick Start (Local)

```bash
# Install dependencies
pip install -r requirements.txt

# Run environment sanity check
python smoke_test.py

# Run baseline inference
export API_BASE_URL="https://api.openai.com/v1"
export MODEL_NAME="gpt-4o-mini"
export OPENAI_API_KEY="your-key"
python inference.py
```

### Docker

```bash
docker build -t pricing-negotiation .
docker run -p 7860:7860 pricing-negotiation
```

### API Endpoints

```
GET  /          → health check
GET  /tasks     → list available tasks
POST /reset     → reset environment (random buyer)
POST /reset/{task_name} → reset with specific task
POST /step      → {"message": "[offer: $80.00] ..."}
GET  /state     → current environment state
```

### HF Spaces

Deploy as a Docker Space tagged with `openenv`. The Dockerfile exposes
port 7860 and runs the FastAPI server.

## Baseline Scores

Scores vary by model and seed. Approximate baselines with `gpt-4o-mini`:

| Task            | Difficulty | Score  |
|-----------------|-----------|--------|
| eager_buyer     | easy      | ~0.75  |
| neutral_buyer   | medium    | ~0.55  |
| bargain_hunter  | hard      | ~0.35  |

## Project Structure

```
pricing_env/
├── __init__.py        # exports
├── buyer.py           # rule-based buyer with hidden WTP
├── env.py             # OpenEnv-compatible PricingEnv
├── rewards.py         # 5 reward signals for GRPO training
├── tasks.py           # 3 task definitions + graders
├── app.py             # FastAPI server for HF Spaces
├── inference.py       # baseline inference script
├── openenv.yaml       # OpenEnv metadata
├── Dockerfile         # container deployment
├── requirements.txt   # Python dependencies
├── smoke_test.py      # env sanity check
├── train_grpo.py      # GRPO training script
└── README.md
```
