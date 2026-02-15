"""
Top Formation Evolutionary Simulation

Balancing-first evolutionary league.

Provides:
- get_parameters()
- run(cards, params)

Supports:
- random seed (None = fully random)
- checkpoint save/resume
"""

from __future__ import annotations

import os
import csv
import time
import pickle
import random
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Tuple, Sequence, Any

from engine.engine import Game
from engine.model import Player, Card, Result


# =============================
# Public API
# =============================

def get_parameters() -> Dict[str, Dict[str, Any]]:
    return {
        "keep_top": {"type": "int", "default": 200},
        "add_per_session": {"type": "int", "default": 50},
        "report_every": {"type": "int", "default": 100},
        "sessions": {"type": "int", "default": 500},
        "seed": {"type": "int_or_none", "default": None},
        "resume_path": {"type": "str_or_none", "default": None},
    }


def run(cards: Sequence[Card], params: Dict[str, Any]):
    resume_path = params.get("resume_path")

    if resume_path:
        runner = Runner.resume(resume_path, cards)
        print("Resumed from:", resume_path)
    else:
        runner = Runner(cards, params)
        runner.initialize()

    runner.step(params["sessions"])

    print("Run directory:", runner.run_dir)
    return runner


# =============================
# Core Engine
# =============================

def formation_key(names):
    return tuple(sorted(names))


def pair_key(a, b):
    return (a, b) if a < b else (b, a)


@dataclass
class Formation:
    id: int
    key: Tuple[str, str, str]


@dataclass
class Stats:
    W: int = 0
    D: int = 0
    L: int = 0

    def rates(self):
        n = self.W + self.D + self.L
        if n == 0:
            return 0, 0, 0
        return self.W/n, self.D/n, self.L/n

    def score(self):
        w, _, l = self.rates()
        return w - l


class Runner:

    def __init__(self, cards: Sequence[Card], params: Dict[str, Any]):

        self.cards = {c.name: c for c in cards}
        self.card_names = list(self.cards.keys())

        self.keep_top = params["keep_top"]
        self.add_per_session = params["add_per_session"]
        self.report_every = params["report_every"]

        self.rng = random.Random()
        if params.get("seed") is not None:
            self.rng.seed(params["seed"])

        self.run_id = time.strftime("%Y%m%d-%H%M%S")
        
        # If running on Colab + Drive mounted, save to Drive for persistence.
        drive_base = Path("/content/drive/MyDrive/renathea_runs")
        if drive_base.exists():
            base = drive_base
        else:
            base = Path("runs")  # fallback local
        
        self.run_dir = base / "top_formation" / self.run_id
        
        self.ckpt_dir = self.run_dir / "checkpoints"
        self.report_dir = self.run_dir / "reports"

        self.ckpt_dir.mkdir(parents=True, exist_ok=True)
        self.report_dir.mkdir(parents=True, exist_ok=True)

        self.formations: Dict[int, Formation] = {}
        self.pool: List[int] = []
        self.seen = set()
        self.cache: Dict[Tuple[int, int], int] = {}
        self.next_id = 1
        self.session = 0

    # -------------------------

    @classmethod
    def resume(cls, path, cards):
        with open(path, "rb") as f:
            obj = pickle.load(f)
        obj.cards = {c.name: c for c in cards}
        obj.card_names = list(obj.cards.keys())
        return obj

    # -------------------------

    def initialize(self):
        init_size = self.keep_top + self.add_per_session
        ids = self._generate(init_size)
        self.pool = ids
        self._ensure_matches(ids)
        self._trim()
        self._save("init")
        self._report("init")

    # -------------------------

    def step(self, n):
        for _ in range(n):
            self.session += 1
            new_ids = self._generate(self.add_per_session)
            self.pool.extend(new_ids)
            self._ensure_matches(new_ids)
            self._trim()

            if self.session % self.report_every == 0:
                self._save()
                self._report()

    # -------------------------

    def _generate(self, n):
        new = []
        attempts = 0
        while len(new) < n:
            attempts += 1
            if attempts > n*200:
                raise RuntimeError("Too many duplicate attempts.")

            names = self.rng.sample(self.card_names, 3)
            key = formation_key(names)
            if key in self.seen:
                continue

            fid = self.next_id
            self.next_id += 1
            self.seen.add(key)
            self.formations[fid] = Formation(fid, key)
            new.append(fid)

        return new

    # -------------------------

    def _ensure_matches(self, new_ids):
        for a in new_ids:
            for b in self.pool:
                if a == b:
                    continue
                pk = pair_key(a, b)
                if pk in self.cache:
                    continue
                self._simulate(a, b)

    # -------------------------

    def _simulate(self, a, b):
        fa = self.formations[a]
        fb = self.formations[b]

        p1 = Player("A", [self.cards[n] for n in fa.key])
        p2 = Player("B", [self.cards[n] for n in fb.key])
        res = Game([p1, p2]).run()

        outcome = res.outcomes[0]
        v = 1 if outcome == Result.WIN else (-1 if outcome == Result.LOSE else 0)

        pk = pair_key(a, b)
        if a < b:
            self.cache[pk] = v
        else:
            self.cache[pk] = -v

    # -------------------------

    def _trim(self):
        stats = {fid: Stats() for fid in self.pool}

        ids = list(self.pool)
        for i in range(len(ids)):
            for j in range(i+1, len(ids)):
                a, b = ids[i], ids[j]
                pk = pair_key(a, b)
                if pk not in self.cache:
                    continue
                v = self.cache[pk]
                if a < b:
                    va = v
                else:
                    va = -v

                if va == 1:
                    stats[a].W += 1
                    stats[b].L += 1
                elif va == -1:
                    stats[a].L += 1
                    stats[b].W += 1
                else:
                    stats[a].D += 1
                    stats[b].D += 1

        ranked = sorted(
            self.pool,
            key=lambda fid: stats[fid].score(),
            reverse=True
        )

        self.pool = ranked[:self.keep_top]

    # -------------------------

    def _report(self, tag=None):
        fname = f"report_s{self.session:05d}.csv"
        if tag:
            fname = f"report_{tag}.csv"

        path = self.report_dir / fname

        with open(path, "w", newline="", encoding="utf-8") as f:
            w = csv.writer(f)
            w.writerow(["rank", "card1", "card2", "card3"])

            for i, fid in enumerate(self.pool[:50], 1):
                w.writerow([i, *self.formations[fid].key])

    # -------------------------

    def _save(self, tag=None):
        fname = f"ckpt_s{self.session:05d}.pkl"
        if tag:
            fname = f"ckpt_{tag}.pkl"

        path = self.ckpt_dir / fname

        with open(path, "wb") as f:
            pickle.dump(self, f)

        with open(self.run_dir / "latest.txt", "w") as f:
            f.write(str(path))
