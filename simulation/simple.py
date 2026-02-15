"""Simple utilities to load card JSON sets and run a quick duel.

Designed to mirror the old notebook workflow:
- pick one or more JSON files from ./data
- parse to engine Card objects
- (optional) dedupe by card.name
- run one random 1v1 match for a sanity check
"""

from __future__ import annotations

import json
import os
import random
from dataclasses import dataclass
from pathlib import Path
from typing import List, Optional, Sequence, Tuple

from engine.engine import Game, parse_card
from engine.model import Card, Player, Result


def list_sets(data_dir: str | os.PathLike = "data") -> List[str]:
    """List available *.json files in the data directory."""
    p = Path(data_dir)
    if not p.exists():
        raise FileNotFoundError(f"Data dir not found: {p.resolve()}")
    return sorted([f.name for f in p.iterdir() if f.is_file() and f.suffix.lower() == ".json"])


def load_cards(
    selected_files: Sequence[str],
    *,
    data_dir: str | os.PathLike = "data",
    dedupe_by_name: bool = True,
) -> Tuple[List[str], List[Card]]:
    """Load and parse cards from one or more JSON files.

    Returns (paths_loaded, cards).
    """
    if not selected_files:
        raise ValueError("selected_files is empty")

    base = Path(data_dir)
    paths: List[str] = []
    cards: List[Card] = []

    for fname in selected_files:
        path = (base / fname).resolve()
        if not path.exists():
            raise FileNotFoundError(f"Set not found: {path}")
        with open(path, "r", encoding="utf-8") as f:
            raw_cards = json.load(f)
        paths.append(str(path))
        cards.extend(parse_card(c) for c in raw_cards)

    if dedupe_by_name:
        seen = set()
        deduped: List[Card] = []
        for c in cards:
            if c.name in seen:
                continue
            seen.add(c.name)
            deduped.append(c)
        cards = deduped

    return paths, cards


def formation_key_from_cards(cards: Sequence[Card]) -> Tuple[str, ...]:
    """Order-insensitive formation key (by name)."""
    return tuple(sorted([c.name for c in cards]))


def draw_random_formation(
    pool: Sequence[Card],
    *,
    size: int = 3,
    rng: Optional[random.Random] = None,
) -> List[Card]:
    """Draw a random formation (no duplicates) from a card pool."""
    rng = rng or random
    if len(pool) < size:
        raise ValueError(f"Not enough cards in pool: need {size}, have {len(pool)}")
    return rng.sample(list(pool), k=size)


@dataclass
class DuelSummary:
    p1_cards: List[str]
    p2_cards: List[str]
    scores: List[int]
    outcomes: List[Result]


def run_random_duel(
    cards: Sequence[Card],
    *,
    seed: Optional[int] = None,
    cards_per_player: int = 3,
) -> DuelSummary:
    """Pick two random formations and run a 1v1 duel."""
    rng = random.Random(seed) if seed is not None else random
    idx = list(range(len(cards)))
    rng.shuffle(idx)
    if len(idx) < cards_per_player * 2:
        raise ValueError("Not enough cards for 2 players")

    p1_cards = [cards[i] for i in idx[:cards_per_player]]
    p2_cards = [cards[i] for i in idx[cards_per_player : cards_per_player * 2]]

    p1 = Player("Player 1", p1_cards)
    p2 = Player("Player 2", p2_cards)
    g = Game([p1, p2])
    res = g.run()

    return DuelSummary(
        p1_cards=[c.name for c in p1_cards],
        p2_cards=[c.name for c in p2_cards],
        scores=res.scores,
        outcomes=res.outcomes,
    )


def format_card_line(card: Card) -> str:
    roles = " ".join(r.value for r in card.roles) if card.roles else ""
    base = f"{card.name} | {roles} ({card.raw_power})" if roles else f"{card.name} | ({card.raw_power})"
    texts = [((s.text or "").strip()) for s in card.skills]
    texts = [t for t in texts if t]
    if texts:
        return base + " | " + " | ".join(texts)
    return base
