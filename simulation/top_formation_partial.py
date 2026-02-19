"""
top_formation_partial.py

Partitioned Top-Formation Evolution (disjoint worlds)

Key points:
- Split pool once into X partitions (no cross mixing)
- 1 session = step ALL worlds once
- Print only at report milestones (like top_formation.py)
- Save 1 checkpoint per milestone
- Save X CSV reports per milestone (one per world)
- Resume is BACKWARD COMPATIBLE with old checkpoints that lack param fields

Report format (per world, per milestone):
- rank, cards
- W/D/L + pctW/pctD/pctL + pctW_minus_pctL
Computed from meta-vs-meta outcomes (meta_cache) only, like top_formation.py.
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
from engine.model import Card, Player
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

def get_parameters():
    return {
        # Core
        "keep_top": {"type": "int", "default": 100},
        "add_per_session": {"type": "int", "default": 30},
        "partitions": {"type": "int", "default": 3},

        # Run control
        "sessions": {"type": "int", "default": 500},
        "seed": {"type": "int_or_none", "default": None},

        # Reporting / saving
        "report_every": {"type": "int", "default": 50},
        "report_top_n": {"type": "int_or_none", "default": None},  # None => keep_top
        "resume_path": {"type": "str_or_none", "default": None},
        "save_to_drive": {"type": "str_or_none", "default": "auto"},
    }


def run(cards: Sequence[Card], params: Dict[str, Any]):
    resume_path = params.get("resume_path")
    if isinstance(resume_path, str) and resume_path.strip() == "":
        resume_path = None

    sessions_to_run = int(params["sessions"])

    if resume_path:
        runner = MultiRunner.resume(resume_path, cards, params)
        print("------------------------------------------------------------")
        print("Mode: RESUME")
        print("Resume from:", resume_path)
    else:
        runner = MultiRunner(cards, params)
        runner.initialize()
        print("------------------------------------------------------------")
        print("Mode: NEW RUN")

    target_session = runner.session + sessions_to_run

    print("Run dir:", runner.run_dir)
    print("Start session:", runner.session)
    print("Target session:", target_session)
    print("------------------------------------------------------------")

    runner.step(sessions_to_run, target_session=target_session)

    print("------------------------------------------------------------")
    print("DONE. Run directory:", runner.run_dir)
    return runner


# ==========================================================
# Data Classes
# ==========================================================

@dataclass
class WorldState:
    world_id: int
    card_names: List[str]
    meta: List[FormationKey]
    ban_window: List[FormationKey]
    meta_cache: Dict[Tuple[FormationKey, FormationKey], int]  # meta-vs-meta only (unordered, min-key perspective)


@dataclass
class Checkpoint:
    session: int
    keep_top: int
    add_per_session: int
    report_every: int
    report_top_n: Optional[int]
    partitions: int
    seed: Optional[int]
    worlds: List[WorldState]
    rng_state: object
    run_id: str
    run_dir: str


# ==========================================================
# Runner
# ==========================================================

class MultiRunner:
    def __init__(self, cards: Sequence[Card], params: Dict[str, Any]):
        self.cards_by_name = {c.name: c for c in cards}
        self.all_card_names = list(self.cards_by_name.keys())

        self.keep_top = int(params["keep_top"])
        self.add_per_session = int(params["add_per_session"])
        self.report_every = int(params["report_every"])
        self.partitions = int(params["partitions"])
        self.seed = params.get("seed", None)
        self.save_to_drive = str(params.get("save_to_drive", "auto"))

        self.report_top_n = params.get("report_top_n", None)
        if self.report_top_n is None:
            self.report_top_n = None
        else:
            self.report_top_n = int(self.report_top_n)

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
    # Resume (Backward Compatible)
    # ==========================================================

    @classmethod
    def resume(cls, path: str | Path, cards: Sequence[Card], params: Dict[str, Any]) -> "MultiRunner":
        path = Path(path)
        with open(path, "rb") as f:
            ckpt = pickle.load(f)

        runner = cls.__new__(cls)

        # cards
        runner.cards_by_name = {c.name: c for c in cards}
        runner.all_card_names = list(runner.cards_by_name.keys())

        # --- restore session/worlds ---
        runner.session = getattr(ckpt, "session", 0)

        if not hasattr(ckpt, "worlds"):
            raise ValueError(
                "Checkpoint ini bukan top_formation_partial (tidak ada field 'worlds'). "
                "Pilih checkpoint dari folder top_formation_partial."
            )
        runner.worlds = getattr(ckpt, "worlds")

        # --- restore RNG ---
        runner.rng = random.Random()
        if hasattr(ckpt, "rng_state"):
            runner.rng.setstate(ckpt.rng_state)
        else:
            seed = params.get("seed", None)
            if seed is not None:
                runner.rng.seed(int(seed))

        # --- restore run dir ---
        runner.run_id = getattr(ckpt, "run_id", time.strftime("%Y%m%d-%H%M%S"))
        runner.run_dir = Path(getattr(ckpt, "run_dir", str(Path("runs") / "top_formation_partial" / runner.run_id))).resolve()

        runner.ckpt_dir = runner.run_dir / "checkpoints"
        runner.report_dir = runner.run_dir / "reports"
        runner.ckpt_dir.mkdir(parents=True, exist_ok=True)
        runner.report_dir.mkdir(parents=True, exist_ok=True)

        # --- restore save mode ---
        runner.save_to_drive = str(params.get("save_to_drive", "auto"))

        # --- restore parameters (backward compat) ---
        keep_top = getattr(ckpt, "keep_top", None)
        add_per_session = getattr(ckpt, "add_per_session", None)
        report_every = getattr(ckpt, "report_every", None)
        report_top_n = getattr(ckpt, "report_top_n", None)
        partitions = getattr(ckpt, "partitions", None)
        seed = getattr(ckpt, "seed", None)

        if keep_top is None:
            try:
                keep_top = len(runner.worlds[0].meta)
            except Exception:
                keep_top = int(params["keep_top"])

        if add_per_session is None:
            try:
                bw = len(runner.worlds[0].ban_window)
                add_per_session = int(bw - int(keep_top))
                if add_per_session <= 0:
                    add_per_session = int(params["add_per_session"])
            except Exception:
                add_per_session = int(params["add_per_session"])

        if report_every is None:
            report_every = int(params["report_every"])

        if partitions is None:
            partitions = len(runner.worlds)

        # report_top_n: if missing, use params, else keep checkpoint
        p_rtn = params.get("report_top_n", None)
        if report_top_n is None:
            report_top_n = p_rtn
        if report_top_n is not None:
            report_top_n = int(report_top_n)

        runner.keep_top = int(keep_top)
        runner.add_per_session = int(add_per_session)
        runner.report_every = int(report_every)
        runner.partitions = int(partitions)
        runner.seed = seed
        runner.report_top_n = report_top_n

        return runner

    # ==========================================================
    # Initialize
    # ==========================================================

    def initialize(self):
        names = list(self.all_card_names)
        self.rng.shuffle(names)

        groups: List[List[str]] = [[] for _ in range(self.partitions)]
        for i, nm in enumerate(names):
            groups[i % self.partitions].append(nm)

        self.worlds = []
        for wid, g in enumerate(groups, start=1):
            self.worlds.append(self._init_world(wid, g))

        self._save()
        self._report()

    def _init_world(self, wid: int, card_names: List[str]) -> WorldState:
        window = self._generate(card_names, self.keep_top + self.add_per_session, set())

        # Full cache inside initial window (unordered, min perspective), then trim to meta_cache only
        window_cache: Dict[Tuple[FormationKey, FormationKey], int] = {}

        for i in range(len(window)):
            a = window[i]
            for j in range(i + 1, len(window)):
                b = window[j]
                out_a = self._simulate_pair(a, b)
                pk = pair_key(a, b)
                window_cache[pk] = out_a if a == pk[0] else -out_a

        ranked = self._rank(window, window_cache)
        meta = [k for k, _ in ranked[: self.keep_top]]

        meta_cache: Dict[Tuple[FormationKey, FormationKey], int] = {}
        for i in range(len(meta)):
            for j in range(i + 1, len(meta)):
                pk = pair_key(meta[i], meta[j])
                meta_cache[pk] = window_cache[pk]

        return WorldState(
            world_id=wid,
            card_names=list(card_names),
            meta=meta,
            ban_window=list(window),
            meta_cache=meta_cache,
        )

    # ==========================================================
    # Step
    # ==========================================================

    def step(self, n: int, target_session: Optional[int] = None):
        for _ in range(int(n)):
            self.session += 1

            for world in self.worlds:
                self._step_world(world)

            if self.session % self.report_every == 0:
                ckpt_path = self._save()
                report_paths = self._report()

                total_str = f"/{target_session}" if target_session else ""
                print(f"Session {self.session}{total_str} ✅", flush=True)
                print(f"Latest report: {report_paths[0]}", flush=True)
                print(f"Latest checkpoint: {ckpt_path}", flush=True)
                print("------------------------------------------------------------", flush=True)

    def _step_world(self, world: WorldState):
        challengers = self._generate(
            world.card_names,
            self.add_per_session,
            set(world.ban_window),
        )

        window = world.meta + challengers

        # Full RR on window for correctness (same behavior as your current partial)
        # Stored unordered, min-key perspective
        session_cache: Dict[Tuple[FormationKey, FormationKey], int] = {}
        for i in range(len(window)):
            a = window[i]
            for j in range(i + 1, len(window)):
                b = window[j]
                pk = pair_key(a, b)
                if pk in session_cache:
                    continue
                out_a = self._simulate_pair(a, b)
                session_cache[pk] = out_a if a == pk[0] else -out_a

        ranked = self._rank(window, session_cache)
        new_meta = [k for k, _ in ranked[: self.keep_top]]

        world.meta = new_meta
        world.ban_window = list(window)

        # Rebuild bounded meta_cache for new meta pairs only:
        # - prefer old meta_cache if pair existed
        # - else use session_cache
        # - else simulate (rare)
        new_meta_cache: Dict[Tuple[FormationKey, FormationKey], int] = {}
        for i in range(len(new_meta)):
            a = new_meta[i]
            for j in range(i + 1, len(new_meta)):
                b = new_meta[j]
                pk = pair_key(a, b)
                if pk in world.meta_cache:
                    new_meta_cache[pk] = world.meta_cache[pk]
                elif pk in session_cache:
                    new_meta_cache[pk] = session_cache[pk]
                else:
                    out_a = self._simulate_pair(a, b)
                    new_meta_cache[pk] = out_a if a == pk[0] else -out_a

        world.meta_cache = new_meta_cache

    # ==========================================================
    # Helpers
    # ==========================================================

    def _generate(self, card_names: List[str], n: int, ban: set[FormationKey]) -> List[FormationKey]:
        out: List[FormationKey] = []
        local_seen: set[FormationKey] = set()

        max_attempts = n * 1000
        attempts = 0

        while len(out) < n:
            attempts += 1
            if attempts > max_attempts:
                raise RuntimeError(
                    f"Failed generating {n} formations. "
                    f"Partition too small or bans too strict. generated={len(out)}"
                )

            k = formation_key_from_names(*self.rng.sample(card_names, 3))
            if k in ban or k in local_seen:
                continue
            local_seen.add(k)
            out.append(k)

        return out

    def _simulate_pair(self, a: FormationKey, b: FormationKey) -> int:
        p1 = Player("A", [self.cards_by_name[n] for n in a])
        p2 = Player("B", [self.cards_by_name[n] for n in b])
        res = Game([p1, p2]).run()
        return outcome_to_int(res.outcomes[0])

    def _rank(self, window: List[FormationKey], cache: Dict[Tuple[FormationKey, FormationKey], int]):
        stats = {k: Stats() for k in window}

        for i in range(len(window)):
            a = window[i]
            for j in range(i + 1, len(window)):
                b = window[j]
                pk = pair_key(a, b)
                v = cache[pk]
                out_a = v if a == pk[0] else -v
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

    # ==========================================================
    # Save / Report
    # ==========================================================

    def _save(self) -> str:
        path = self.ckpt_dir / f"ckpt_s{self.session:05d}.pkl"

        ckpt = Checkpoint(
            session=self.session,
            keep_top=self.keep_top,
            add_per_session=self.add_per_session,
            report_every=self.report_every,
            report_top_n=self.report_top_n,
            partitions=self.partitions,
            seed=self.seed if (self.seed is None or isinstance(self.seed, int)) else int(self.seed),
            worlds=self.worlds,
            rng_state=self.rng.getstate(),
            run_id=self.run_id,
            run_dir=str(self.run_dir),
        )

        with open(path, "wb") as f:
            pickle.dump(ckpt, f, protocol=pickle.HIGHEST_PROTOCOL)

        (self.run_dir / "latest.txt").write_text(str(path), encoding="utf-8")
        return str(path)

    def _report(self) -> List[str]:
        """
        Report like top_formation.py:
        Compute W/D/L and pct from meta-vs-meta only, using world.meta_cache.
        """
        out_paths: List[str] = []

        top_n = self.keep_top if self.report_top_n is None else min(self.keep_top, int(self.report_top_n))

        for world in self.worlds:
            meta = list(world.meta)
            st: Dict[FormationKey, Stats] = {k: Stats() for k in meta}

            # compute within-meta stats using meta_cache (unordered, min perspective)
            for i in range(len(meta)):
                a = meta[i]
                for j in range(i + 1, len(meta)):
                    b = meta[j]
                    pk = pair_key(a, b)
                    v = world.meta_cache.get(pk)
                    if v is None:
                        # safety fallback: simulate once and store
                        out_a = self._simulate_pair(a, b)
                        v = out_a if a == pk[0] else -out_a
                        world.meta_cache[pk] = v

                    out_a = v if a == pk[0] else -v
                    out_b = -out_a

                    if out_a > 0:
                        st[a].W += 1
                    elif out_a < 0:
                        st[a].L += 1
                    else:
                        st[a].D += 1

                    if out_b > 0:
                        st[b].W += 1
                    elif out_b < 0:
                        st[b].L += 1
                    else:
                        st[b].D += 1

            ranked = sorted(
                meta,
                key=lambda k: (st[k].score(), st[k].W, -st[k].L),
                reverse=True,
            )

            path = self.report_dir / f"report_world{world.world_id}_s{self.session:05d}.csv"
            with open(path, "w", newline="", encoding="utf-8") as f:
                w = csv.writer(f)
                w.writerow([
                    "world", "session", "rank",
                    "card1", "card2", "card3",
                    "W", "D", "L",
                    "pctW", "pctD", "pctL",
                    "pctW_minus_pctL"
                ])

                for r, key in enumerate(ranked[:top_n], start=1):
                    s = st[key]
                    pctW, pctD, pctL = s.rates()
                    w.writerow([
                        world.world_id, self.session, r,
                        key[0], key[1], key[2],
                        s.W, s.D, s.L,
                        round(pctW, 6), round(pctD, 6), round(pctL, 6),
                        round(s.score(), 6)
                    ])

            out_paths.append(str(path))

        return out_paths
