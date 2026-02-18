"""
top_formation_partial.py

Run X independent meta worlds inside ONE runner.
- 1 checkpoint per session (actually per report_every milestone, same as top_formation)
- X CSV reports per milestone (one per world)
- No reshuffle / reseed on resume
- Has module-level run(cards, params) so notebook can execute it.
"""

from __future__ import annotations

import csv
import pickle
import random
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple

from engine.engine import Game
from engine.model import Card, Player, Result

from simulation.top_formation import (
    formation_key_from_names,
    pair_key,
    outcome_to_int,
    Stats,
    Runner as BaseRunner,
)

FormationKey = Tuple[str, str, str]


# ==========================================================
# Public API
# ==========================================================

def get_parameters() -> Dict[str, Dict[str, object]]:
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


def run(cards: Sequence[Card], params: Dict[str, object]):
    resume_path = params.get("resume_path")
    if isinstance(resume_path, str) and resume_path.strip() == "":
        resume_path = None

    if resume_path:
        runner = MultiRunner.resume(resume_path, cards)
        print("Resumed from:", resume_path)
    else:
        runner = MultiRunner(cards, params)
        runner.initialize()

    runner.step(int(params["sessions"]))  # additional sessions
    print("Run directory:", runner.run_dir)
    return runner


# ==========================================================
# World State
# ==========================================================

@dataclass
class WorldState:
    meta: List[FormationKey]
    ban_window: List[FormationKey]
    meta_cache: Dict[Tuple[FormationKey, FormationKey], int]


@dataclass
class Checkpoint:
    session: int
    keep_top: int
    add_per_session: int
    report_every: int
    partitions: int

    worlds: List[WorldState]
    rng_state: object

    run_id: str
    run_dir: str

    seed: Optional[int]
    save_to_drive: str


# ==========================================================
# Multi Runner
# ==========================================================

class MultiRunner:
    def __init__(self, cards: Sequence[Card], params: Dict[str, object]):
        self.cards_by_name = {c.name: c for c in cards}
        self.card_names = list(self.cards_by_name.keys())
        if len(self.card_names) < 3:
            raise ValueError("Need at least 3 cards in pool.")

        self.keep_top = int(params["keep_top"])
        self.add_per_session = int(params["add_per_session"])
        self.report_every = int(params["report_every"])
        self.partitions = int(params["partitions"])
        self.seed = params.get("seed", None)
        self.save_to_drive = str(params.get("save_to_drive", "auto"))

        if self.partitions <= 0:
            raise ValueError("partitions must be > 0")
        if self.keep_top <= 0 or self.add_per_session <= 0:
            raise ValueError("keep_top and add_per_session must be > 0")
        if self.report_every <= 0:
            raise ValueError("report_every must be > 0")

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

        self.session: int = 0
        self.worlds: List[WorldState] = []

    # -------------------------
    # Resume
    # -------------------------

    @classmethod
    def resume(cls, path: str | Path, cards: Sequence[Card]) -> "MultiRunner":
        path = Path(path)
        with open(path, "rb") as f:
            ckpt: Checkpoint = pickle.load(f)

        runner = cls.__new__(cls)

        runner.cards_by_name = {c.name: c for c in cards}
        runner.card_names = list(runner.cards_by_name.keys())

        runner.keep_top = ckpt.keep_top
        runner.add_per_session = ckpt.add_per_session
        runner.report_every = ckpt.report_every
        runner.partitions = ckpt.partitions
        runner.seed = ckpt.seed
        runner.save_to_drive = ckpt.save_to_drive

        runner.rng = random.Random()
        runner.rng.setstate(ckpt.rng_state)

        runner.run_id = ckpt.run_id
        runner.run_dir = Path(ckpt.run_dir)
        runner.ckpt_dir = runner.run_dir / "checkpoints"
        runner.report_dir = runner.run_dir / "reports"
        runner.ckpt_dir.mkdir(parents=True, exist_ok=True)
        runner.report_dir.mkdir(parents=True, exist_ok=True)

        runner.session = ckpt.session
        runner.worlds = ckpt.worlds

        return runner

    # -------------------------
    # Initialize
    # -------------------------

    def initialize(self):
        # Split cards once (disjoint)
        cards = list(self.cards_by_name.values())
        self.rng.shuffle(cards)

        groups: List[List[Card]] = [[] for _ in range(self.partitions)]
        for i, c in enumerate(cards):
            groups[i % self.partitions].append(c)

        # ensure each partition can form a formation
        for i, g in enumerate(groups, start=1):
            if len(g) < 3:
                raise ValueError(f"Partition {i} has only {len(g)} cards (<3). Reduce partitions.")

        self.worlds = []
        for g in groups:
            self.worlds.append(self._init_world(g))

        self.session = 0
        self._save(tag="init")
        self._report(tag="init")

    def _init_world(self, cards_subset: List[Card]) -> WorldState:
        names = [c.name for c in cards_subset]

        def generate(n: int) -> List[FormationKey]:
            out: List[FormationKey] = []
            local_seen: set[FormationKey] = set()
            attempts = 0
            max_attempts = n * 400
            while len(out) < n:
                attempts += 1
                if attempts > max_attempts:
                    raise RuntimeError("Failed generating initial window. Partition too small.")
                k = formation_key_from_names(*self.rng.sample(names, 3))
                if k in local_seen:
                    continue
                local_seen.add(k)
                out.append(k)
            return out

        window = generate(self.keep_top + self.add_per_session)

        # full round robin for init window
        cache: Dict[Tuple[FormationKey, FormationKey], int] = {}
        for i in range(len(window)):
            a = window[i]
            for j in range(i + 1, len(window)):
                b = window[j]
                pk = pair_key(a, b)
                if pk in cache:
                    continue
                out_a = self._simulate_pair(a, b)
                # store from min perspective
                cache[pk] = out_a if a == pk[0] else -out_a

        ranked = self._rank(window, cache)
        meta = [k for k, _ in ranked[: self.keep_top]]

        # keep meta-vs-meta only
        meta_cache: Dict[Tuple[FormationKey, FormationKey], int] = {}
        for i in range(len(meta)):
            for j in range(i + 1, len(meta)):
                pk = pair_key(meta[i], meta[j])
                meta_cache[pk] = cache[pk]

        return WorldState(meta=meta, ban_window=list(window), meta_cache=meta_cache)

    # -------------------------
    # Step
    # -------------------------

    def step(self, n: int):
        if not self.worlds:
            raise RuntimeError("Runner not initialized. Call initialize() or resume().")

        for _ in range(int(n)):
            self.session += 1
            for world in self.worlds:
                self._step_world(world)

            if self.session % self.report_every == 0:
                self._save()
                self._report()

    def _step_world(self, world: WorldState):
        ban = set(world.ban_window)

        challengers: List[FormationKey] = []
        local_seen: set[FormationKey] = set()
        attempts = 0
        max_attempts = self.add_per_session * 400

        while len(challengers) < self.add_per_session:
            attempts += 1
            if attempts > max_attempts:
                raise RuntimeError("Failed generating challengers. Pool too small or ban too strict.")
            k = formation_key_from_names(*self.rng.sample(self.card_names, 3))
            if k in ban or k in local_seen:
                continue
            local_seen.add(k)
            challengers.append(k)

        window = list(world.meta) + challengers
        session_cache: Dict[Tuple[FormationKey, FormationKey], int] = {}

        # simulate challengers-only efficiently:
        # challengers vs meta
        for c in challengers:
            for m in world.meta:
                pk = pair_key(c, m)
                if pk in session_cache:
                    continue
                out_c = self._simulate_pair(c, m)
                session_cache[pk] = out_c if c == pk[0] else -out_c

        # challengers vs challengers
        for i in range(len(challengers)):
            a = challengers[i]
            for j in range(i + 1, len(challengers)):
                b = challengers[j]
                pk = pair_key(a, b)
                if pk in session_cache:
                    continue
                out_a = self._simulate_pair(a, b)
                session_cache[pk] = out_a if a == pk[0] else -out_a

        # rank using meta_cache + session_cache
        ranked = self._rank(window, session_cache, meta_cache=world.meta_cache)
        new_meta = [k for k, _ in ranked[: self.keep_top]]

        # update ban window (meta + challengers = 250)
        world.ban_window = list(window)

        # rebuild meta_cache bounded for new meta
        world.meta_cache = self._build_meta_cache(new_meta, world.meta_cache, session_cache)
        world.meta = new_meta

    # -------------------------
    # Duel / Rank
    # -------------------------

    def _simulate_pair(self, a: FormationKey, b: FormationKey) -> int:
        p1 = Player("A", [self.cards_by_name[n] for n in a])
        p2 = Player("B", [self.cards_by_name[n] for n in b])
        res = Game([p1, p2]).run()
        return outcome_to_int(res.outcomes[0])

    def _rank(
        self,
        window: List[FormationKey],
        window_cache: Dict[Tuple[FormationKey, FormationKey], int],
        meta_cache: Optional[Dict[Tuple[FormationKey, FormationKey], int]] = None,
    ):
        st: Dict[FormationKey, Stats] = {k: Stats() for k in window}

        for i in range(len(window)):
            a = window[i]
            for j in range(i + 1, len(window)):
                b = window[j]
                pk = pair_key(a, b)

                v = None
                if meta_cache is not None and pk in meta_cache:
                    v = meta_cache[pk]
                else:
                    v = window_cache.get(pk)

                if v is None:
                    out_a = self._simulate_pair(a, b)
                    v = out_a if a == pk[0] else -out_a
                    window_cache[pk] = v

                # convert to a/b perspective
                out_a = v if a == pk[0] else -v
                out_b = -out_a

                self._accumulate(st[a], out_a)
                self._accumulate(st[b], out_b)

        ranked = sorted(
            [(k, st[k]) for k in window],
            key=lambda x: (x[1].score(), x[1].W, -x[1].L),
            reverse=True,
        )
        return ranked

    @staticmethod
    def _accumulate(s: Stats, v: int):
        if v > 0:
            s.W += 1
        elif v < 0:
            s.L += 1
        else:
            s.D += 1

    def _build_meta_cache(
        self,
        new_meta: List[FormationKey],
        old_meta_cache: Dict[Tuple[FormationKey, FormationKey], int],
        session_cache: Dict[Tuple[FormationKey, FormationKey], int],
    ) -> Dict[Tuple[FormationKey, FormationKey], int]:
        out: Dict[Tuple[FormationKey, FormationKey], int] = {}
        for i in range(len(new_meta)):
            a = new_meta[i]
            for j in range(i + 1, len(new_meta)):
                b = new_meta[j]
                pk = pair_key(a, b)

                if pk in old_meta_cache:
                    out[pk] = old_meta_cache[pk]
                elif pk in session_cache:
                    out[pk] = session_cache[pk]
                else:
                    out_a = self._simulate_pair(a, b)
                    out[pk] = out_a if a == pk[0] else -out_a
        return out

    # -------------------------
    # Save / Report
    # -------------------------

    def _save(self, tag: Optional[str] = None):
        fname = f"ckpt_s{self.session:05d}.pkl" if tag is None else f"ckpt_{tag}_s{self.session:05d}.pkl"
        path = self.ckpt_dir / fname

        ckpt = Checkpoint(
            session=self.session,
            keep_top=self.keep_top,
            add_per_session=self.add_per_session,
            report_every=self.report_every,
            partitions=self.partitions,
            worlds=self.worlds,
            rng_state=self.rng.getstate(),
            run_id=self.run_id,
            run_dir=str(self.run_dir),
            seed=None if self.seed is None else int(self.seed),
            save_to_drive=self.save_to_drive,
        )

        with open(path, "wb") as f:
            pickle.dump(ckpt, f, protocol=pickle.HIGHEST_PROTOCOL)

        (self.run_dir / "latest.txt").write_text(str(path), encoding="utf-8")

    def _report(self, tag: Optional[str] = None):
        # Write one CSV per world (full meta keep_top rows)
        for idx, world in enumerate(self.worlds, start=1):
            fname = f"report_world{idx}_s{self.session:05d}.csv" if tag is None else f"report_{tag}_world{idx}_s{self.session:05d}.csv"
            path = self.report_dir / fname

            # Compute W/D/L within meta using world.meta_cache
            meta = list(world.meta)
            st: Dict[FormationKey, Stats] = {k: Stats() for k in meta}

            for i in range(len(meta)):
                a = meta[i]
                for j in range(i + 1, len(meta)):
                    b = meta[j]
                    pk = pair_key(a, b)
                    v = world.meta_cache.get(pk)
                    if v is None:
                        out_a = self._simulate_pair(a, b)
                        v = out_a if a == pk[0] else -out_a
                        world.meta_cache[pk] = v

                    out_a = v if a == pk[0] else -v
                    out_b = -out_a
                    self._accumulate(st[a], out_a)
                    self._accumulate(st[b], out_b)

            ranked = sorted(
                meta,
                key=lambda k: (st[k].score(), st[k].W, -st[k].L),
                reverse=True,
            )

            with open(path, "w", newline="", encoding="utf-8") as f:
                w = csv.writer(f)
                w.writerow([
                    "run_id", "world", "session",
                    "keep_top", "add_per_session", "report_every", "seed",
                    "rank",
                    "card1", "card2", "card3",
                    "W", "D", "L",
                    "pctW", "pctD", "pctL",
                    "pctW_minus_pctL",
                ])

                for r, key in enumerate(ranked[: self.keep_top], start=1):
                    s = st[key]
                    pctW, pctD, pctL = s.rates()
                    w.writerow([
                        self.run_id, idx, self.session,
                        self.keep_top, self.add_per_session, self.report_every, self.seed,
                        r,
                        key[0], key[1], key[2],
                        s.W, s.D, s.L,
                        round(pctW, 6), round(pctD, 6), round(pctL, 6),
                        round(s.score(), 6),
                    ])
