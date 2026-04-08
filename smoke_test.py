"""Standalone env sanity check — no ML deps required."""
import random
try:
    from pricing_env import PricingEnv, PricingAction
except ImportError:
    from env import PricingEnv, PricingAction

env = PricingEnv(list_price=100.0, cost=50.0, max_turns=6, seed=42)
outcomes = {"sold": 0, "walked": 0, "timeout": 0}
total_revenue = 0.0

for ep in range(20):
    result = env.reset()
    print(f"\n--- Ep {ep+1} | persona={env.buyer.persona} | hidden WTP=${env.buyer.wtp:.2f} ---")
    for m in result.observation.messages:
        print(f"  [{m.category}] {m.content}")

    offer = env.list_price
    while not result.done:
        text = f"[offer: ${offer:.2f}] That's my best price."
        result = env.step(PricingAction(message=text))
        print(f"  [AGENT] ${offer:.2f} -> {result.info.get('buyer_action')}")
        if result.observation.messages and not result.done:
            last = result.observation.messages[-1]
            print(f"  [{last.category}] {last.content}")
        offer *= 0.85
        if offer < env.cost:
            offer = env.cost + 1

    outcome = result.info.get("final_outcome", "?")
    outcomes[outcome] = outcomes.get(outcome, 0) + 1
    if outcome == "sold":
        total_revenue += result.reward
    print(f"  => {outcome}, reward={result.reward:.2f}")

print(f"\n=== Summary over 20 episodes ===")
print(f"Outcomes: {outcomes}")
print(f"Total revenue: ${total_revenue:.2f}")
print(f"Avg revenue/episode: ${total_revenue/20:.2f}")
