"""
Top Formation Evolutionary Simulation (Stable + Meta-Relative + Full Report)

Changes vs previous:
- Report outputs ALL meta (keep_top), not fixed 50
- CSV includes run metadata (run_id, keep_top, add_per_session, etc.)
- Zero-sum preserved within meta
- Resume-safe
"""

from __future__ import annotations

import csv
import pickle
import random
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple

from engine.engine import Game
from engine.model import Card, Player, Result


# ==========================================================
# Public API
# ==========================================================

def get_parameters() -> Dict[str, Dict[str, Any]]:
    return {
        "keep_top": {"type": "int", "default": 200},
        "add_per_session": {"type": "int", "default": 50},
        "report_every": {"type": "int", "default": 100},
        "sessions": {"type": "int", "default": 500},
        "seed": {"type": "int_or_none", "default": None},
        "resume_path": {"type": "str_or_none", "default": None},
        "save_to_drive": {"type": "str_or_none", "default": "auto"},
    }


def run(cards: Sequence[Card], params: Dict[str, Any]):
    resume_path = params.get("resume_path")
    if isinstance(resume_path, str) and resume_path.strip() == "":
        resume_path = None

    if resume_path:
        runner = Runner.resume(resume_path, cards)
        print("Resumed from:", resume_path)
    else:
        runner = Runner(cards, params)
        runner.initialize()

    runner.step(int(params["sessions"]))
    print("Run directory:", runner.run_dir)
    return runner


# ==========================================================
# Internals
# ==========================================================

FormationKey = Tuple[str, str, str]


def formation_key_from_names(n1: str, n2: str, n3: str) -> FormationKey:
    return tuple(sorted((n1, n2, n3)))  # type: ignore


def outcome_to_int(out: Result) -> int:
    if out == Result.WIN:
        return 1
    if out == Result.DRAW:
        return 0
    return -1


def pair_key(a: FormationKey, b: FormationKey):
    return (a, b) if a < b else (b, a)


@dataclass
class Stats:
    W: int = 0
    D: int = 0
    L: int = 0

    def rates(self):
        n = self.W + self.D + self.L
        if n == 0:
            return (0.0, 0.0, 0.0)
        return (self.W / n, self.D / n, self.L / n)

    def score(self):
        w, _, l = self.rates()
        return w - l


@dataclass
class Checkpoint:
    session: int
    keep_top: int
    add_per_session: int
    report_every: int
    meta: List[FormationKey]
    ban_window: List[FormationKey]
    meta_cache: Dict
    rng_state: object
    run_id: str
    run_dir: str
    seed: Optional[int]
    save_to_drive: str


class Runner:

    # ==========================================================
    # Init / Resume
    # ==========================================================

    def __init__(self, cards: Sequence[Card], params: Dict[str, Any]):

        self.cards_by_name = {c.name: c for c in cards}
        self.card_names = list(self.cards_by_name.keys())

        self.keep_top = int(params["keep_top"])
        self.add_per_session = int(params["add_per_session"])
        self.report_every = int(params["report_every"])
        self.seed = params.get("seed")
        self.save_to_drive = str(params.get("save_to_drive", "auto"))

        self.rng = random.Random()
        if self.seed is not None:
            self.rng.seed(int(self.seed))

        self.run_id = time.strftime("%Y%m%d-%H%M%S")
        base = self._resolve_base_dir(self.save_to_drive)

        self.run_dir = (base / "top_formation" / self.run_id).resolve()
        self.ckpt_dir = self.run_dir / "checkpoints"
        self.report_dir = self.run_dir / "reports"
        self.ckpt_dir.mkdir(parents=True, exist_ok=True)
        self.report_dir.mkdir(parents=True, exist_ok=True)

        self.session = 0
        self.meta: List[FormationKey] = []
        self.ban_window: List[FormationKey] = []
        self.meta_cache: Dict = {}

    @staticmethod
    def _resolve_base_dir(save_to_drive: str) -> Path:

        local = Path("runs")
        mydrive = Path("/content/drive/MyDrive")
        drive_base = mydrive / "renathea_runs"

        if save_to_drive == "no":
            return local

        if mydrive.exists():
            drive_base.mkdir(parents=True, exist_ok=True)
            return drive_base

        return local

    @classmethod
    def resume(cls, checkpoint_path, cards):
        with open(checkpoint_path, "rb") as f:
            ckpt: Checkpoint = pickle.load(f)

        runner = cls.__new__(cls)
        runner.cards_by_name = {c.name: c for c in cards}
        runner.card_names = list(runner.cards_by_name.keys())

        runner.keep_top = ckpt.keep_top
        runner.add_per_session = ckpt.add_per_session
        runner.report_every = ckpt.report_every
        runner.seed = ckpt.seed
        runner.save_to_drive = ckpt.save_to_drive

        runner.rng = random.Random()
        runner.rng.setstate(ckpt.rng_state)

        runner.run_id = ckpt.run_id
        runner.run_dir = Path(ckpt.run_dir)
        runner.ckpt_dir = runner.run_dir / "checkpoints"
        runner.report_dir = runner.run_dir / "reports"

        runner.session = ckpt.session
        runner.meta = list(ckpt.meta)
        runner.ban_window = list(ckpt.ban_window)
        runner.meta_cache = dict(ckpt.meta_cache)

        return runner

    # ==========================================================
    # Simulation
    # ==========================================================

    def initialize(self):
        window_size = self.keep_top + self.add_per_session
        window = self._generate(window_size, set())

        window_cache = {}
        self._simulate_full(window, window_cache)

        ranked = self._rank(window, window_cache, None)
        self.meta = [k for k, _ in ranked[: self.keep_top]]

        self.ban_window = list(window)
        self.meta_cache = self._build_meta_cache(self.meta, None, window_cache)

        self._save(tag="init")
        self._report(tag="init")

    def step(self, n_sessions: int):

        for _ in range(n_sessions):
            self.session += 1

            challengers = self._generate(self.add_per_session, set(self.ban_window))
            window = self.meta + challengers

            session_cache = {}
            self._simulate_challengers(self.meta, challengers, session_cache)

            ranked = self._rank(window, session_cache, self.meta_cache)
            new_meta = [k for k, _ in ranked[: self.keep_top]]

            self.ban_window = list(window)
            self.meta_cache = self._build_meta_cache(new_meta, self.meta_cache, session_cache)
            self.meta = new_meta

            if self.session % self.report_every == 0:
                self._save()
                self._report()

    # ==========================================================
    # Core Logic
    # ==========================================================

    def _generate(self, n, ban):
        out = []
        while len(out) < n:
            k = formation_key_from_names(*self.rng.sample(self.card_names, 3))
            if k in ban or k in out:
                continue
            out.append(k)
        return out

    def _simulate_pair(self, a, b):
        p1 = Player("A", [self.cards_by_name[n] for n in a])
        p2 = Player("B", [self.cards_by_name[n] for n in b])
        res = Game([p1, p2]).run()
        return outcome_to_int(res.outcomes[0])

    def _simulate_full(self, keys, cache):
        for i in range(len(keys)):
            for j in range(i + 1, len(keys)):
                a, b = keys[i], keys[j]
                v = self._simulate_pair(a, b)
                cache[pair_key(a, b)] = v if a < b else -v

    def _simulate_challengers(self, meta, challengers, cache):

        for c in challengers:
            for m in meta:
                if pair_key(c, m) not in cache:
                    v = self._simulate_pair(c, m)
                    cache[pair_key(c, m)] = v if c < m else -v

        for i in range(len(challengers)):
            for j in range(i + 1, len(challengers)):
                a, b = challengers[i], challengers[j]
                if pair_key(a, b) not in cache:
                    v = self._simulate_pair(a, b)
                    cache[pair_key(a, b)] = v if a < b else -v

    def _rank(self, window, session_cache, meta_cache):

        stats = {k: Stats() for k in window}

        for i in range(len(window)):
            for j in range(i + 1, len(window)):
                a, b = window[i], window[j]
                pk = pair_key(a, b)

                if meta_cache and pk in meta_cache:
                    v = meta_cache[pk]
                else:
                    v = session_cache.get(pk)
                    if v is None:
                        v = self._simulate_pair(a, b)

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

        ranked = sorted(
            [(k, stats[k]) for k in window],
            key=lambda x: (x[1].score(), x[1].W, -x[1].L),
            reverse=True,
        )

        return ranked

    def _build_meta_cache(self, meta, old_cache, session_cache):
        out = {}
        for i in range(len(meta)):
            for j in range(i + 1, len(meta)):
                pk = pair_key(meta[i], meta[j])
                if old_cache and pk in old_cache:
                    out[pk] = old_cache[pk]
                elif pk in session_cache:
                    out[pk] = session_cache[pk]
                else:
                    v = self._simulate_pair(meta[i], meta[j])
                    out[pk] = v if meta[i] < meta[j] else -v
        return out

    # ==========================================================
    # Save / Report
    # ==========================================================

    def _save(self, tag=None):

        fname = f"ckpt_s{self.session:05d}.pkl"
        path = self.ckpt_dir / fname

        ckpt = Checkpoint(
            session=self.session,
            keep_top=self.keep_top,
            add_per_session=self.add_per_session,
            report_every=self.report_every,
            meta=self.meta,
            ban_window=self.ban_window,
            meta_cache=self.meta_cache,
            rng_state=self.rng.getstate(),
            run_id=self.run_id,
            run_dir=str(self.run_dir),
            seed=self.seed,
            save_to_drive=self.save_to_drive,
        )

        with open(path, "wb") as f:
            pickle.dump(ckpt, f)

        (self.run_dir / "latest.txt").write_text(str(path))

    def _report(self, tag=None):

        meta = list(self.meta)
        stats = {k: Stats() for k in meta}

        for i in range(len(meta)):
            for j in range(i + 1, len(meta)):
                pk = pair_key(meta[i], meta[j])
                v = self.meta_cache[pk]
                out_a = v if meta[i] < meta[j] else -v
                out_b = -out_a

                if out_a > 0:
                    stats[meta[i]].W += 1
                elif out_a < 0:
                    stats[meta[i]].L += 1
                else:
                    stats[meta[i]].D += 1

                if out_b > 0:
                    stats[meta[j]].W += 1
                elif out_b < 0:
                    stats[meta[j]].L += 1
                else:
                    stats[meta[j]].D += 1

        ranked = sorted(
            meta,
            key=lambda k: (stats[k].score(), stats[k].W, -stats[k].L),
            reverse=True,
        )

        fname = f"report_s{self.session:05d}.csv"
        path = self.report_dir / fname

        with open(path, "w", newline="") as f:
            w = csv.writer(f)

            w.writerow([
                "run_id",
                "session",
                "keep_top",
                "add_per_session",
                "rank",
                "card1",
                "card2",
                "card3",
                "W",
                "D",
                "L",
                "pctW",
                "pctD",
                "pctL",
                "pctW_minus_pctL",
            ])

            for r, k in enumerate(ranked[: self.keep_top], start=1):
                s = stats[k]
                pctW, pctD, pctL = s.rates()
                w.writerow([
                    self.run_id,
                    self.session,
                    self.keep_top,
                    self.add_per_session,
                    r,
                    k[0],
                    k[1],
                    k[2],
                    s.W,
                    s.D,
                    s.L,
                    round(pctW, 6),
                    round(pctD, 6),
                    round(pctL, 6),
                    round(s.score(), 6),
                ])
