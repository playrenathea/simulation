"""Top formation evolutionary search (balancing-first).

Concept:
- Maintain a league (pool) of formations (3 cards).
- Init with (keep_top + add_per_session) formations.
- Each session: add add_per_session new formations.
- Only simulate matches needed: (new vs all existing pool incl. new), skipping self.
- Rank formations by (%W - %L) within the current pool, then keep top keep_top.
- Every report_every sessions: write a report CSV and save a checkpoint that can resume.

Notes:
- Formation identity is order-insensitive and derived from Card.name.
- Duplicates (including order permutations) are rejected globally per run.
"""

from __future__ import annotations

import csv
import json
import os
import pickle
import random
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple

from engine.engine import Game, parse_card
from engine.model import Card, Player, Result


# -----------------------------
# Keys / Encoding
# -----------------------------

FormationKey = Tuple[str, str, str]  # sorted 3 names


def formation_key(names: Sequence[str]) -> FormationKey:
    if len(names) != 3:
        raise ValueError("formation_key expects exactly 3 names")
    return tuple(sorted(names))  # type: ignore[return-value]


def outcome_to_int(outcome: Result) -> int:
    # perspective: +1 win, 0 draw, -1 lose
    if outcome == Result.WIN:
        return 1
    if outcome == Result.DRAW:
        return 0
    return -1


def int_to_outcome(v: int) -> Result:
    if v > 0:
        return Result.WIN
    if v < 0:
        return Result.LOSE
    return Result.DRAW


def pair_key(a_id: int, b_id: int) -> Tuple[int, int]:
    return (a_id, b_id) if a_id < b_id else (b_id, a_id)


# -----------------------------
# Loading
# -----------------------------

def list_sets(data_dir: str | os.PathLike = "data") -> List[str]:
    p = Path(data_dir)
    return sorted([f.name for f in p.iterdir() if f.is_file() and f.suffix.lower() == ".json"])


def load_cards(
    selected_files: Sequence[str],
    *,
    data_dir: str | os.PathLike = "data",
    dedupe_by_name: bool = True,
) -> Tuple[List[str], List[Card]]:
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


# -----------------------------
# Core data structures
# -----------------------------


@dataclass(frozen=True)
class Formation:
    """Formation is identified by its key (3 card names sorted).

    The actual Card objects are resolved via name->Card map.
    """

    id: int
    key: FormationKey

    @property
    def names(self) -> Tuple[str, str, str]:
        return self.key


@dataclass
class FormationStats:
    wins: int = 0
    draws: int = 0
    losses: int = 0

    def as_rates(self) -> Tuple[float, float, float]:
        n = self.wins + self.draws + self.losses
        if n <= 0:
            return (0.0, 0.0, 0.0)
        return (self.wins / n, self.draws / n, self.losses / n)

    def wl_rate_diff(self) -> float:
        w, _d, l = self.as_rates()
        return w - l


@dataclass
class RunConfig:
    run_id: str
    selected_sets: List[str]
    dedupe_by_name: bool
    keep_top: int
    add_per_session: int
    report_every: int
    seed: Optional[int]
    formation_key_mode: str = "order_insensitive_sorted_names"
    ranking_metric: str = "%W-%L"
    exclude_self_match: bool = True


@dataclass
class Checkpoint:
    config: RunConfig
    session_idx: int
    rng_state: object
    next_formation_id: int

    # current pool
    pool_ids: List[int]

    # global duplicate guard
    seen_keys: set

    # registry
    formations: Dict[int, Formation]

    # match cache: (min_id, max_id) -> outcome int (from min_id perspective)
    match_cache: Dict[Tuple[int, int], int]


# -----------------------------
# League runner
# -----------------------------


class TopFormationRunner:
    def __init__(
        self,
        cards: Sequence[Card],
        *,
        run_dir: str | os.PathLike = "runs/top_formation",
        keep_top: int = 200,
        add_per_session: int = 50,
        report_every: int = 100,
        seed: Optional[int] = None,
        selected_sets: Optional[List[str]] = None,
        dedupe_by_name: bool = True,
    ):
        if keep_top <= 0:
            raise ValueError("keep_top must be > 0")
        if add_per_session <= 0:
            raise ValueError("add_per_session must be > 0")
        if report_every <= 0:
            raise ValueError("report_every must be > 0")
        if len(cards) < 3:
            raise ValueError("Need at least 3 cards")

        self.cards_by_name: Dict[str, Card] = {c.name: c for c in cards}
        self.card_names: List[str] = list(self.cards_by_name.keys())

        self.rng = random.Random(seed) if seed is not None else random.Random()
        if seed is None:
            self.rng.seed(time.time_ns())

        run_id = time.strftime("%Y%m%d-%H%M%S")
        self.config = RunConfig(
            run_id=run_id,
            selected_sets=selected_sets or [],
            dedupe_by_name=dedupe_by_name,
            keep_top=keep_top,
            add_per_session=add_per_session,
            report_every=report_every,
            seed=seed,
        )

        self.run_dir = Path(run_dir) / run_id
        self.ckpt_dir = self.run_dir / "checkpoints"
        self.report_dir = self.run_dir / "reports"
        self.ckpt_dir.mkdir(parents=True, exist_ok=True)
        self.report_dir.mkdir(parents=True, exist_ok=True)

        self.formations: Dict[int, Formation] = {}
        self.pool_ids: List[int] = []
        self.seen_keys: set = set()
        self.match_cache: Dict[Tuple[int, int], int] = {}
        self.next_formation_id: int = 1
        self.session_idx: int = 0

        self._save_config()

    @classmethod
    def resume_from_checkpoint(cls, ckpt_path: str | os.PathLike, cards: Sequence[Card]):
        """Resume a run from a checkpoint file."""
        ckpt_path = Path(ckpt_path)
        with open(ckpt_path, "rb") as f:
            ckpt: Checkpoint = pickle.load(f)

        # Bypass __init__ to avoid creating a new run folder.
        runner = cls.__new__(cls)
        runner.cards_by_name = {c.name: c for c in cards}
        runner.card_names = list(runner.cards_by_name.keys())

        runner.config = ckpt.config
        runner.run_dir = ckpt_path.parent.parent
        runner.ckpt_dir = runner.run_dir / "checkpoints"
        runner.report_dir = runner.run_dir / "reports"
        runner.ckpt_dir.mkdir(parents=True, exist_ok=True)
        runner.report_dir.mkdir(parents=True, exist_ok=True)

        runner.rng = random.Random()
        runner.rng.setstate(ckpt.rng_state)
        runner.session_idx = ckpt.session_idx
        runner.next_formation_id = ckpt.next_formation_id
        runner.pool_ids = list(ckpt.pool_ids)
        runner.seen_keys = set(ckpt.seen_keys)
        runner.formations = dict(ckpt.formations)
        runner.match_cache = dict(ckpt.match_cache)

        return runner

    def initialize(self) -> None:
        """Create initial pool and compute initial cache."""
        init_n = self.config.keep_top + self.config.add_per_session
        new_ids = self._generate_unique_formations(init_n)
        self.pool_ids = new_ids

        # Build initial cache for full pool (round-robin) once.
        self._ensure_matches_for_new_ids(new_ids)

        # Trim to keep_top
        ranked = self.rank_pool(self.pool_ids)
        self.pool_ids = [fid for fid, _stats in ranked[: self.config.keep_top]]
        self.session_idx = 0

        # Save initial checkpoint + report
        self.save_checkpoint(tag="init")
        self.write_report(tag="init")

    def step(self, n_sessions: int = 1) -> None:
        """Run N sessions. Checkpoints/reports are written every report_every sessions."""
        if not self.pool_ids:
            raise RuntimeError("Runner not initialized. Call initialize() or resume_from_checkpoint().")

        for _ in range(n_sessions):
            self.session_idx += 1

            # Add batch
            new_ids = self._generate_unique_formations(self.config.add_per_session)
            self.pool_ids.extend(new_ids)

            # Only compute matches that involve newly added formations.
            self._ensure_matches_for_new_ids(new_ids)

            # Rank & trim
            ranked = self.rank_pool(self.pool_ids)
            self.pool_ids = [fid for fid, _stats in ranked[: self.config.keep_top]]

            # Savepoint
            if self.session_idx % self.config.report_every == 0:
                self.save_checkpoint()
                self.write_report()

    def rank_pool(self, pool_ids: Sequence[int]) -> List[Tuple[int, FormationStats]]:
        """Compute W/D/L within the pool and return sorted list (best first)."""
        stats: Dict[int, FormationStats] = {fid: FormationStats() for fid in pool_ids}

        ids = list(pool_ids)
        for i in range(len(ids)):
            a = ids[i]
            for j in range(i + 1, len(ids)):
                b = ids[j]
                pk = pair_key(a, b)
                if pk not in self.match_cache:
                    self._simulate_and_store(a, b)
                v = self.match_cache[pk]
                # v is from min_id perspective
                min_id, _max_id = pk
                if a == min_id:
                    out_a = int_to_outcome(v)
                    out_b = int_to_outcome(-v)
                else:
                    out_a = int_to_outcome(-v)
                    out_b = int_to_outcome(v)

                self._accumulate(stats[a], out_a)
                self._accumulate(stats[b], out_b)

        ranked = sorted(
            [(fid, stats[fid]) for fid in ids],
            key=lambda x: (x[1].wl_rate_diff(), x[1].wins, -x[1].losses),
            reverse=True,
        )
        return ranked

    def save_checkpoint(self, tag: Optional[str] = None) -> Path:
        """Save a checkpoint that can be resumed."""
        fname = self._ckpt_name(tag)
        path = self.ckpt_dir / fname
        ckpt = Checkpoint(
            config=self.config,
            session_idx=self.session_idx,
            rng_state=self.rng.getstate(),
            next_formation_id=self.next_formation_id,
            pool_ids=list(self.pool_ids),
            seen_keys=set(self.seen_keys),
            formations=dict(self.formations),
            match_cache=dict(self.match_cache),
        )
        with open(path, "wb") as f:
            pickle.dump(ckpt, f, protocol=pickle.HIGHEST_PROTOCOL)

        # also update latest pointer
        with open(self.run_dir / "latest.txt", "w", encoding="utf-8") as f:
            f.write(str(path))
        return path

    def write_report(self, tag: Optional[str] = None, top_n: int = 50) -> Path:
        """Write a report CSV for the current pool."""
        ranked = self.rank_pool(self.pool_ids)
        fname = self._report_name(tag)
        path = self.report_dir / fname

        with open(path, "w", newline="", encoding="utf-8") as f:
            w = csv.writer(f)
            w.writerow(
                [
                    "session",
                    "rank",
                    "card_1",
                    "card_2",
                    "card_3",
                    "W",
                    "D",
                    "L",
                    "pctW",
                    "pctD",
                    "pctL",
                    "pctW_minus_pctL",
                ]
            )
            for r, (fid, st) in enumerate(ranked[:top_n], start=1):
                names = self.formations[fid].names
                pctW, pctD, pctL = st.as_rates()
                w.writerow(
                    [
                        self.session_idx,
                        r,
                        names[0],
                        names[1],
                        names[2],
                        st.wins,
                        st.draws,
                        st.losses,
                        round(pctW, 6),
                        round(pctD, 6),
                        round(pctL, 6),
                        round(st.wl_rate_diff(), 6),
                    ]
                )

        return path

    # -----------------
    # Internal helpers
    # -----------------

    def _save_config(self) -> None:
        path = self.run_dir / "config.json"
        with open(path, "w", encoding="utf-8") as f:
            json.dump(self.config.__dict__, f, ensure_ascii=False, indent=2)

    def _ckpt_name(self, tag: Optional[str]) -> str:
        if tag:
            return f"ckpt_{tag}_s{self.session_idx:05d}.pkl"
        return f"ckpt_s{self.session_idx:05d}.pkl"

    def _report_name(self, tag: Optional[str]) -> str:
        if tag:
            return f"report_{tag}_s{self.session_idx:05d}.csv"
        return f"report_s{self.session_idx:05d}.csv"

    def _generate_unique_formations(self, n: int) -> List[int]:
        """Generate N new unique formations (by key), update registries, return ids."""
        new_ids: List[int] = []
        attempts = 0
        max_attempts = n * 200

        while len(new_ids) < n:
            attempts += 1
            if attempts > max_attempts:
                raise RuntimeError(
                    f"Failed to generate {n} unique formations. Pool too small or too many duplicates. "
                    f"Generated {len(new_ids)} after {attempts} attempts."
                )

            names = self.rng.sample(self.card_names, k=3)
            key = formation_key(names)
            if key in self.seen_keys:
                continue

            fid = self.next_formation_id
            self.next_formation_id += 1
            self.seen_keys.add(key)
            self.formations[fid] = Formation(id=fid, key=key)
            new_ids.append(fid)

        return new_ids

    def _ensure_matches_for_new_ids(self, new_ids: Sequence[int]) -> None:
        """Compute match results for each new formation vs every formation in current pool.

        Efficiency rule: only do (new vs all (old + new)),
        skipping self-match and skipping already-cached pairs.
        """
        pool = list(self.pool_ids)
        for a in new_ids:
            for b in pool:
                if a == b:
                    continue  # self-match excluded (always draw)
                pk = pair_key(a, b)
                if pk in self.match_cache:
                    continue
                self._simulate_and_store(a, b)

    def _simulate_and_store(self, a_id: int, b_id: int) -> None:
        """Simulate 1v1 between formation a and b, store in cache."""
        a = self.formations[a_id]
        b = self.formations[b_id]

        p1 = Player("A", [self.cards_by_name[n] for n in a.names])
        p2 = Player("B", [self.cards_by_name[n] for n in b.names])
        res = Game([p1, p2]).run()
        out_a = res.outcomes[0]
        v = outcome_to_int(out_a)

        pk = pair_key(a_id, b_id)
        min_id, _max_id = pk
        # store from min_id perspective
        if a_id == min_id:
            self.match_cache[pk] = v
        else:
            self.match_cache[pk] = -v

    def _accumulate(self, st: FormationStats, out: Result) -> None:
        if out == Result.WIN:
            st.wins += 1
        elif out == Result.DRAW:
            st.draws += 1
        else:
            st.losses += 1
