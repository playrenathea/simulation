"""
Top Formation Evolutionary Simulation (Powerful + Efficient + Long-run Safe)

Key goals (per user spec):
- Meta size fixed: keep_top (default 200)
- Each session adds add_per_session (default 50) challengers
- Evaluate window = meta + challengers (250)
- Save only:
  - meta formations (200)
  - ban window (250) to prevent immediate repeats
  - match cache for meta vs meta only (C(200,2)=19,900; excludes self)
  - RNG state + session
- Next challengers must NOT be in last window of 250 (ban_keys), but may reappear later.
- Per session simulate only:
  - challengers vs meta (50*200)
  - challengers vs challengers (C(50,2))
  - exclude self, no double counting

Public API:
- get_parameters()
- run(cards, params)
- Runner class with .initialize(), .step(n), .resume(path, cards)
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


# =============================
# Public API
# =============================

def get_parameters() -> Dict[str, Dict[str, Any]]:
    return {
        "keep_top": {"type": "int", "default": 200},
        "add_per_session": {"type": "int", "default": 50},
        "report_every": {"type": "int", "default": 100},
        "sessions": {"type": "int", "default": 500},
        "seed": {"type": "int_or_none", "default": None},          # None = fully random
        "resume_path": {"type": "str_or_none", "default": None},   # checkpoint path
        "save_to_drive": {"type": "str_or_none", "default": "auto"} # "auto" | "yes" | "no"
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


# =============================
# Internals
# =============================

FormationKey = Tuple[str, str, str]  # 3 names, sorted (order-insensitive)


def formation_key_from_names(n1: str, n2: str, n3: str) -> FormationKey:
    return tuple(sorted((n1, n2, n3)))  # type: ignore[return-value]


def outcome_to_int(out: Result) -> int:
    # Perspective: +1 win, 0 draw, -1 lose
    if out == Result.WIN:
        return 1
    if out == Result.DRAW:
        return 0
    return -1


def pair_key(a: FormationKey, b: FormationKey) -> Tuple[FormationKey, FormationKey]:
    return (a, b) if a < b else (b, a)


def get_from_cache(cache: Dict[Tuple[FormationKey, FormationKey], int], a: FormationKey, b: FormationKey) -> int:
    """Return outcome from a's perspective (+1/0/-1) using unordered cache stored from min-key perspective."""
    if a == b:
        return 0
    pk = pair_key(a, b)
    v = cache.get(pk)
    if v is None:
        raise KeyError("Pair not in cache")
    # v is from min perspective
    if a == pk[0]:
        return v
    return -v


@dataclass
class Stats:
    W: int = 0
    D: int = 0
    L: int = 0

    def rates(self) -> Tuple[float, float, float]:
        n = self.W + self.D + self.L
        if n <= 0:
            return (0.0, 0.0, 0.0)
        return (self.W / n, self.D / n, self.L / n)

    def score(self) -> float:
        w, _d, l = self.rates()
        return w - l  # ranking metric


@dataclass
class Checkpoint:
    # minimal durable state
    session: int
    keep_top: int
    add_per_session: int
    report_every: int

    # meta & ban
    meta: List[FormationKey]       # size keep_top
    ban_window: List[FormationKey] # size keep_top + add_per_session (250)

    # only cache for meta vs meta
    meta_cache: Dict[Tuple[FormationKey, FormationKey], int]

    # rng
    rng_state: object

    # output location info
    run_id: str
    run_dir: str

    # optional context
    seed: Optional[int]
    save_to_drive: str


class Runner:
    def __init__(self, cards: Sequence[Card], params: Dict[str, Any]):
        # Card map
        self.cards_by_name: Dict[str, Card] = {c.name: c for c in cards}
        self.card_names: List[str] = list(self.cards_by_name.keys())
        if len(self.card_names) < 3:
            raise ValueError("Need at least 3 cards.")

        # Params
        self.keep_top = int(params["keep_top"])
        self.add_per_session = int(params["add_per_session"])
        self.report_every = int(params["report_every"])
        self.seed = params.get("seed", None)
        self.save_to_drive = str(params.get("save_to_drive", "auto"))

        if self.keep_top <= 0 or self.add_per_session <= 0:
            raise ValueError("keep_top and add_per_session must be > 0")
        if self.report_every <= 0:
            raise ValueError("report_every must be > 0")

        # RNG
        self.rng = random.Random()
        if self.seed is not None:
            self.rng.seed(int(self.seed))

        # Output directories
        self.run_id = time.strftime("%Y%m%d-%H%M%S")
        base = self._resolve_base_dir(self.save_to_drive)
        self.run_dir = (base / "top_formation" / self.run_id).resolve()
        self.ckpt_dir = self.run_dir / "checkpoints"
        self.report_dir = self.run_dir / "reports"
        self.ckpt_dir.mkdir(parents=True, exist_ok=True)
        self.report_dir.mkdir(parents=True, exist_ok=True)

        # State
        self.session: int = 0
        self.meta: List[FormationKey] = []
        self.ban_window: List[FormationKey] = []

        # Only cache meta vs meta (unordered)
        self.meta_cache: Dict[Tuple[FormationKey, FormationKey], int] = {}

    @staticmethod
    def _resolve_base_dir(save_to_drive: str) -> Path:
        """
        save_to_drive:
          - "auto": use Drive if mounted, else local runs/
          - "yes": force Drive if mounted
          - "no": always local runs/
        """
    
        local = Path("runs")
        mydrive = Path("/content/drive/MyDrive")
        drive_base = mydrive / "renathea_runs"
    
        # Always local if explicitly no
        if save_to_drive == "no":
            return local
    
        # If Drive is mounted, use it and create folder if needed
        if mydrive.exists():
            drive_base.mkdir(parents=True, exist_ok=True)
            return drive_base
    
        # Fallback to local
        return local

    @classmethod
    def resume(cls, checkpoint_path: str | Path, cards: Sequence[Card]) -> "Runner":
        checkpoint_path = Path(checkpoint_path)
        with open(checkpoint_path, "rb") as f:
            ckpt: Checkpoint = pickle.load(f)

        # Create runner shell without running __init__ fully
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
        runner.ckpt_dir.mkdir(parents=True, exist_ok=True)
        runner.report_dir.mkdir(parents=True, exist_ok=True)

        runner.session = ckpt.session
        runner.meta = list(ckpt.meta)
        runner.ban_window = list(ckpt.ban_window)
        runner.meta_cache = dict(ckpt.meta_cache)

        return runner

    # =============================
    # Simulation
    # =============================

    def initialize(self):
        """
        Initial window = keep_top + add_per_session (250).
        We compute full round-robin among 250 once (C(250,2)=31,125).
        Then trim to meta 200 and build meta_cache (meta vs meta only).
        """
        window_size = self.keep_top + self.add_per_session
        window = self._generate_unique_formations(window_size, ban=set(), ensure_unique=True)

        # Full cache for initial window (unordered, min perspective)
        window_cache: Dict[Tuple[FormationKey, FormationKey], int] = {}
        self._simulate_full_round_robin(window, window_cache)

        # Rank and trim to meta
        ranked = self._rank_window(window, window_cache, meta_cache=None)
        self.meta = [k for k, _st in ranked[: self.keep_top]]

        # Ban window = the evaluated window
        self.ban_window = list(window)

        # Build meta_cache subset from window_cache (meta vs meta only)
        self.meta_cache = self._build_meta_cache_from_sources(self.meta, old_meta_cache=None, window_cache=window_cache)

        self.session = 0
        self._save(tag="init")
        self._report(tag="init")

    def step(self, n_sessions: int):
        """
        Each session:
          - Generate challengers (50) not in last ban_window (250)
          - Compute only matches involving challengers:
              challengers vs meta, challengers vs challengers
          - Rank full window (meta+challengers) using:
              meta_cache for meta-vs-meta
              session_cache for any pair touching challengers
          - Trim to meta
          - Rebuild meta_cache for new meta only (19,900 max)
          - Update ban_window to current evaluated window (250)
          - Save/report every report_every
        """
        if not self.meta:
            raise RuntimeError("Runner not initialized. Call initialize() or resume().")

        ban = set(self.ban_window)

        for _ in range(int(n_sessions)):
            self.session += 1

            # 1) Generate challengers (not in last 250)
            challengers = self._generate_unique_formations(
                self.add_per_session,
                ban=ban,
                ensure_unique=True,
            )

            # 2) Window to evaluate
            window = list(self.meta) + list(challengers)  # 250

            # 3) Session cache: only pairs involving challengers (unordered)
            session_cache: Dict[Tuple[FormationKey, FormationKey], int] = {}
            self._simulate_challengers_only(self.meta, challengers, session_cache)

            # 4) Rank window (250) using meta_cache + session_cache
            ranked = self._rank_window(window, session_cache, meta_cache=self.meta_cache)
            new_meta = [k for k, _st in ranked[: self.keep_top]]

            # 5) Update ban window to current evaluated window
            self.ban_window = list(window)
            ban = set(self.ban_window)

            # 6) Rebuild meta_cache only for new_meta (meta vs meta only)
            # Sources:
            # - old meta_cache contains outcomes for old-meta pairs that survive together
            # - session_cache contains outcomes for any pair involving challengers (some survive)
            self.meta_cache = self._build_meta_cache_from_sources(new_meta, old_meta_cache=self.meta_cache, window_cache=session_cache)

            # 7) Update meta
            self.meta = new_meta

            # 8) Save/report at milestones
            if self.session % self.report_every == 0:
                self._save()
                self._report()

    # =============================
    # Generation
    # =============================

    def _generate_unique_formations(self, n: int, ban: set[FormationKey], ensure_unique: bool = True) -> List[FormationKey]:
        """
        Generate n formations. Reject if in ban.
        ensure_unique=True also rejects duplicates within the generated batch.
        """
        out: List[FormationKey] = []
        local_seen: set[FormationKey] = set()
        attempts = 0
        max_attempts = n * 400

        while len(out) < n:
            attempts += 1
            if attempts > max_attempts:
                raise RuntimeError(
                    f"Failed generating {n} unique formations. "
                    f"Pool too small or bans too strict. Generated={len(out)} attempts={attempts}"
                )

            n1, n2, n3 = self.rng.sample(self.card_names, 3)
            key = formation_key_from_names(n1, n2, n3)

            if key in ban:
                continue
            if ensure_unique and (key in local_seen):
                continue

            local_seen.add(key)
            out.append(key)

        return out

    # =============================
    # Simulation / Caching
    # =============================

    def _simulate_pair(self, a: FormationKey, b: FormationKey) -> int:
        """Simulate 1v1 and return outcome int from a's perspective."""
        p1 = Player("A", [self.cards_by_name[n] for n in a])
        p2 = Player("B", [self.cards_by_name[n] for n in b])
        res = Game([p1, p2]).run()
        return outcome_to_int(res.outcomes[0])

    def _put_cache(self, cache: Dict[Tuple[FormationKey, FormationKey], int], a: FormationKey, b: FormationKey, out_a: int):
        """Store unordered cache from min-key perspective."""
        if a == b:
            return
        pk = pair_key(a, b)
        if a == pk[0]:
            cache[pk] = out_a
        else:
            cache[pk] = -out_a

    def _simulate_full_round_robin(self, keys: List[FormationKey], cache: Dict[Tuple[FormationKey, FormationKey], int]):
        """Compute all unordered pairs among keys once. Excludes self."""
        for i in range(len(keys)):
            a = keys[i]
            for j in range(i + 1, len(keys)):
                b = keys[j]
                pk = pair_key(a, b)
                if pk in cache:
                    continue
                out_a = self._simulate_pair(a, b)
                self._put_cache(cache, a, b, out_a)

    def _simulate_challengers_only(
        self,
        meta: List[FormationKey],
        challengers: List[FormationKey],
        cache: Dict[Tuple[FormationKey, FormationKey], int],
    ):
        """
        Simulate only:
          - challengers vs meta (50*200)
          - challengers vs challengers (C(50,2))
        Excludes self and avoids double counting via unordered caching.
        """
        # challengers vs meta
        for c in challengers:
            for m in meta:
                pk = pair_key(c, m)
                if pk in cache:
                    continue
                out_c = self._simulate_pair(c, m)
                self._put_cache(cache, c, m, out_c)

        # challengers vs challengers (unordered)
        for i in range(len(challengers)):
            a = challengers[i]
            for j in range(i + 1, len(challengers)):
                b = challengers[j]
                pk = pair_key(a, b)
                if pk in cache:
                    continue
                out_a = self._simulate_pair(a, b)
                self._put_cache(cache, a, b, out_a)

    # =============================
    # Ranking + Meta cache rebuild
    # =============================

    def _rank_window(
        self,
        window: List[FormationKey],
        window_cache: Dict[Tuple[FormationKey, FormationKey], int],
        meta_cache: Optional[Dict[Tuple[FormationKey, FormationKey], int]],
    ) -> List[Tuple[FormationKey, Stats]]:
        """
        Rank full window using pairwise results.
        Source priority:
          1) meta_cache if both are meta and pair exists
          2) window_cache for any pair involving challengers
        We compute Stats by iterating all unordered pairs in window (C(250,2)=31,125),
        but only doing lookups (not simulations).
        """
        st: Dict[FormationKey, Stats] = {k: Stats() for k in window}

        for i in range(len(window)):
            a = window[i]
            for j in range(i + 1, len(window)):
                b = window[j]

                # fetch outcome from best cache
                v: Optional[int] = None
                pk = pair_key(a, b)

                if meta_cache is not None and pk in meta_cache:
                    # outcome from min perspective stored in meta_cache
                    v = meta_cache[pk]
                else:
                    v = window_cache.get(pk)

                if v is None:
                    # Shouldn't happen. As a safety net, simulate once and store in window_cache.
                    out_a = self._simulate_pair(a, b)
                    self._put_cache(window_cache, a, b, out_a)
                    v = window_cache[pk]

                # Convert v (min perspective) to a's perspective
                if a == pk[0]:
                    out_a = v
                    out_b = -v
                else:
                    out_a = -v
                    out_b = v

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

    def _build_meta_cache_from_sources(
        self,
        new_meta: List[FormationKey],
        old_meta_cache: Optional[Dict[Tuple[FormationKey, FormationKey], int]],
        window_cache: Dict[Tuple[FormationKey, FormationKey], int],
    ) -> Dict[Tuple[FormationKey, FormationKey], int]:
        """
        Build cache for meta-vs-meta only (unordered), using available sources:
          - old_meta_cache for pairs that existed before
          - window_cache for pairs involving challengers this session (or init window)
        This ensures saved cache stays bounded (~19,900).
        """
        out: Dict[Tuple[FormationKey, FormationKey], int] = {}
        for i in range(len(new_meta)):
            a = new_meta[i]
            for j in range(i + 1, len(new_meta)):
                b = new_meta[j]
                pk = pair_key(a, b)

                if old_meta_cache is not None and pk in old_meta_cache:
                    out[pk] = old_meta_cache[pk]
                    continue
                if pk in window_cache:
                    out[pk] = window_cache[pk]
                    continue

                # Safety fallback (rare): simulate and store
                out_a = self._simulate_pair(a, b)
                self._put_cache(out, a, b, out_a)

        return out

    # =============================
    # Save / Report
    # =============================

    def _save(self, tag: Optional[str] = None):
        fname = f"ckpt_s{self.session:05d}.pkl" if tag is None else f"ckpt_{tag}_s{self.session:05d}.pkl"
        path = self.ckpt_dir / fname

        ckpt = Checkpoint(
            session=self.session,
            keep_top=self.keep_top,
            add_per_session=self.add_per_session,
            report_every=self.report_every,
            meta=list(self.meta),
            ban_window=list(self.ban_window),
            meta_cache=dict(self.meta_cache),
            rng_state=self.rng.getstate(),
            run_id=self.run_id,
            run_dir=str(self.run_dir),
            seed=self.seed if (self.seed is None or isinstance(self.seed, int)) else int(self.seed),
            save_to_drive=self.save_to_drive,
        )

        with open(path, "wb") as f:
            pickle.dump(ckpt, f, protocol=pickle.HIGHEST_PROTOCOL)

        # latest pointer
        latest = self.run_dir / "latest.txt"
        latest.write_text(str(path), encoding="utf-8")

    def _report(self, tag: Optional[str] = None, top_n: int = 50):
        # For reporting meta only, we compute Stats within meta using meta_cache.
        # This is (C(200,2)=19,900 lookups), small.
        meta = list(self.meta)
        st: Dict[FormationKey, Stats] = {k: Stats() for k in meta}

        for i in range(len(meta)):
            a = meta[i]
            for j in range(i + 1, len(meta)):
                b = meta[j]
                pk = pair_key(a, b)
                v = self.meta_cache.get(pk)
                if v is None:
                    # safety simulate
                    out_a = self._simulate_pair(a, b)
                    self._put_cache(self.meta_cache, a, b, out_a)
                    v = self.meta_cache[pk]

                if a == pk[0]:
                    out_a = v
                    out_b = -v
                else:
                    out_a = -v
                    out_b = v

                self._accumulate(st[a], out_a)
                self._accumulate(st[b], out_b)

        ranked = sorted(
            meta,
            key=lambda k: (st[k].score(), st[k].W, -st[k].L),
            reverse=True,
        )

        fname = f"report_s{self.session:05d}.csv" if tag is None else f"report_{tag}_s{self.session:05d}.csv"
        path = self.report_dir / fname

        with open(path, "w", newline="", encoding="utf-8") as f:
            w = csv.writer(f)
            w.writerow([
                "session", "rank",
                "card1", "card2", "card3",
                "W", "D", "L",
                "pctW", "pctD", "pctL",
                "pctW_minus_pctL"
            ])

            for r, key in enumerate(ranked[:top_n], start=1):
                s = st[key]
                pctW, pctD, pctL = s.rates()
                w.writerow([
                    self.session, r,
                    key[0], key[1], key[2],
                    s.W, s.D, s.L,
                    round(pctW, 6), round(pctD, 6), round(pctL, 6),
                    round(s.score(), 6)
                ])
