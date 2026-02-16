"""
top_formation_partial.py

Run X independent meta worlds inside ONE runner.
- 1 checkpoint per session
- X CSV reports per session (one per world)
- No reshuffle / reseed on resume
"""

from __future__ import annotations

import pickle
import random
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple

from engine.model import Card
from simulation.top_formation import (
    formation_key_from_names,
    pair_key,
    outcome_to_int,
    Stats,
    Runner as BaseRunner,
)
from engine.engine import Game
from engine.model import Player, Result


FormationKey = Tuple[str, str, str]


# ==========================================================
# Parameters
# ==========================================================

def get_parameters():
    return {
        "keep_top": {"type": "int", "default": 200},
        "add_per_session": {"type": "int", "default": 50},
        "report_every": {"type": "int", "default": 100},
        "partitions": {"type": "int", "default": 3},
        "sessions": {"type": "int", "default": 500},
        "seed": {"type": "int_or_none", "default": None},
        "resume_path": {"type": "str_or_none", "default": None},
        "save_to_drive": {"type": "str_or_none", "default": "auto"},
    }


# ==========================================================
# World State
# ==========================================================

@dataclass
class WorldState:
    meta: List[FormationKey]
    ban_window: List[FormationKey]
    meta_cache: Dict


@dataclass
class Checkpoint:
    session: int
    worlds: List[WorldState]
    rng_state: object
    run_id: str
    run_dir: str


# ==========================================================
# Multi Runner
# ==========================================================

class MultiRunner:

    def __init__(self, cards: Sequence[Card], params):

        self.cards_by_name = {c.name: c for c in cards}
        self.card_names = list(self.cards_by_name.keys())

        self.keep_top = int(params["keep_top"])
        self.add_per_session = int(params["add_per_session"])
        self.report_every = int(params["report_every"])
        self.partitions = int(params["partitions"])
        self.seed = params.get("seed")
        self.save_to_drive = params.get("save_to_drive", "auto")

        self.rng = random.Random()
        if self.seed is not None:
            self.rng.seed(int(self.seed))

        base = BaseRunner._resolve_base_dir(self.save_to_drive)
        self.run_id = time.strftime("%Y%m%d-%H%M%S")
        self.run_dir = (base / "top_formation_partial" / self.run_id).resolve()
        self.ckpt_dir = self.run_dir / "checkpoints"
        self.report_dir = self.run_dir / "reports"

        self.ckpt_dir.mkdir(parents=True, exist_ok=True)
        self.report_dir.mkdir(parents=True, exist_ok=True)

        self.session = 0
        self.worlds: List[WorldState] = []

    # ==========================================================

    @classmethod
    def resume(cls, path, cards):
        with open(path, "rb") as f:
            ckpt: Checkpoint = pickle.load(f)

        runner = cls.__new__(cls)
        runner.cards_by_name = {c.name: c for c in cards}
        runner.card_names = list(runner.cards_by_name.keys())

        runner.session = ckpt.session
        runner.worlds = ckpt.worlds
        runner.rng = random.Random()
        runner.rng.setstate(ckpt.rng_state)

        runner.run_id = ckpt.run_id
        runner.run_dir = Path(ckpt.run_dir)
        runner.ckpt_dir = runner.run_dir / "checkpoints"
        runner.report_dir = runner.run_dir / "reports"

        return runner

    # ==========================================================

    def initialize(self):

        # Split cards once
        cards = list(self.cards_by_name.values())
        self.rng.shuffle(cards)

        groups = [[] for _ in range(self.partitions)]
        for i, c in enumerate(cards):
            groups[i % self.partitions].append(c)

        for g in groups:
            world = self._init_world(g)
            self.worlds.append(world)

        self._save()
        self._report()

    # ==========================================================

    def _init_world(self, cards_subset):

        names = [c.name for c in cards_subset]

        def generate(n):
            out = []
            while len(out) < n:
                k = formation_key_from_names(*self.rng.sample(names, 3))
                if k not in out:
                    out.append(k)
            return out

        window = generate(self.keep_top + self.add_per_session)

        cache = {}
        for i in range(len(window)):
            for j in range(i + 1, len(window)):
                a, b = window[i], window[j]
                v = self._simulate_pair(a, b)
                cache[pair_key(a, b)] = v if a < b else -v

        ranked = self._rank(window, cache)
        meta = [k for k, _ in ranked[: self.keep_top]]

        meta_cache = {
            pk: cache[pk]
            for pk in cache
            if pk[0] in meta and pk[1] in meta
        }

        return WorldState(meta=meta, ban_window=window, meta_cache=meta_cache)

    # ==========================================================

    def step(self, n):

        for _ in range(n):
            self.session += 1

            for world in self.worlds:
                self._step_world(world)

            if self.session % self.report_every == 0:
                self._save()
                self._report()

    # ==========================================================

    def _step_world(self, world: WorldState):

        ban = set(world.ban_window)

        challengers = []
        while len(challengers) < self.add_per_session:
            k = formation_key_from_names(*self.rng.sample(self.card_names, 3))
            if k not in ban and k not in challengers:
                challengers.append(k)

        window = world.meta + challengers
        session_cache = {}

        for i in range(len(window)):
            for j in range(i + 1, len(window)):
                a, b = window[i], window[j]
                pk = pair_key(a, b)
                if pk not in session_cache:
                    v = self._simulate_pair(a, b)
                    session_cache[pk] = v if a < b else -v

        ranked = self._rank(window, session_cache)
        new_meta = [k for k, _ in ranked[: self.keep_top]]

        world.meta = new_meta
        world.ban_window = window
        world.meta_cache = {
            pk: session_cache[pk]
            for pk in session_cache
            if pk[0] in new_meta and pk[1] in new_meta
        }

    # ==========================================================

    def _simulate_pair(self, a, b):
        p1 = Player("A", [self.cards_by_name[n] for n in a])
        p2 = Player("B", [self.cards_by_name[n] for n in b])
        res = Game([p1, p2]).run()
        return outcome_to_int(res.outcomes[0])

    def _rank(self, window, cache):

        stats = {k: Stats() for k in window}

        for i in range(len(window)):
            for j in range(i + 1, len(window)):
                a, b = window[i], window[j]
                pk = pair_key(a, b)
                v = cache[pk]
                out_a = v if a < b else -v
                out_b = -out_a

                if out_a > 0:
                    stats[a].W += 1
                elif out_a < 0:
                    stats[a].L += 1
                else:
                    stats[a].D += 1

                if out_b > 0:
                    stats[b].W += 1
                elif out_b < 0:
                    stats[b].L += 1
                else:
                    stats[b].D += 1

        return sorted(
            [(k, stats[k]) for k in window],
            key=lambda x: (x[1].score(), x[1].W, -x[1].L),
            reverse=True,
        )

    # ==========================================================

    def _save(self):

        path = self.ckpt_dir / f"ckpt_s{self.session:05d}.pkl"

        ckpt = Checkpoint(
            session=self.session,
            worlds=self.worlds,
            rng_state=self.rng.getstate(),
            run_id=self.run_id,
            run_dir=str(self.run_dir),
        )

        with open(path, "wb") as f:
            pickle.dump(ckpt, f)

        (self.run_dir / "latest.txt").write_text(str(path))

    # ==========================================================

    def _report(self):

        for idx, world in enumerate(self.worlds, start=1):

            path = self.report_dir / f"report_world{idx}_s{self.session:05d}.csv"

            with open(path, "w", newline="") as f:
                import csv
                w = csv.writer(f)
                w.writerow(["world", "session", "rank", "card1", "card2", "card3"])

                for r, k in enumerate(world.meta[: self.keep_top], start=1):
                    w.writerow([idx, self.session, r, k[0], k[1], k[2]])
