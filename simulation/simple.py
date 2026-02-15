"""
Simple simulation module.

Provides:
- get_parameters()
- run(cards, params)

Used for quick 1v1 random duel sanity check.
"""

from __future__ import annotations

import random
from typing import Dict, Any, Sequence

from engine.engine import Game
from engine.model import Player, Card


# -----------------------------
# Public API (for UI system)
# -----------------------------

def get_parameters() -> Dict[str, Dict[str, Any]]:
    return {
        "seed": {
            "type": "int_or_none",
            "default": None,
            "description": "Random seed (None = fully random)",
        }
    }


def run(cards: Sequence[Card], params: Dict[str, Any]):
    if len(cards) < 6:
        raise ValueError("Need at least 6 cards for simple 1v1 duel.")

    seed = params.get("seed", None)

    rng = random.Random()
    if seed is not None:
        rng.seed(seed)

    sampled = rng.sample(list(cards), 6)
    p1_cards = sampled[:3]
    p2_cards = sampled[3:]

    p1 = Player("Player 1", p1_cards)
    p2 = Player("Player 2", p2_cards)

    result = Game([p1, p2]).run()

    print("\n=== SIMPLE DUEL ===")
    print("P1:", [c.name for c in p1_cards])
    print("P2:", [c.name for c in p2_cards])
    print("Scores:", result.scores)
    print("Outcomes:", result.outcomes)

    return result
