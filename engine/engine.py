# engine/engine.py
from __future__ import annotations

from dataclasses import dataclass, field
from typing import List, Optional, Tuple, Union, Any, Dict, TypedDict

from .model import (
    Player, Card, Skill, SkillType, SkillTiming,
    Zone, Status, Law, Result
)
from .dsl import CardFilter, Count, Value, Condition


# ----------------------------
# Runtime state wrapper
# ----------------------------

@dataclass
class CardEntry:
    # Same pointer approach as notebook (feature parity)
    card: Tuple[int, int]                 # (player_index, card_index)
    zone: Optional[Zone] = None
    side: int = 0                         # owner index (you=int)
    this: int = 0                         # index in state.entries

    # SINGLE status (Renathea rule)
    # Protected dominates: once Protected, cannot become Silenced.
    status: Optional[Status] = None

    visibility: List[int] = field(default_factory=list)

    # Forward-compatible (not used in Vanilla; does not change behavior)
    virtual_skills: List[Skill] = field(default_factory=list)

    def get_card(self, state: "GameState") -> Card:
        player_index, card_index = self.card
        return state.players[player_index].cards[card_index]

    def get_skills(self, state: "GameState") -> List[Skill]:
        return state.ruleset.entry_skills(state, self)

    def get_roles(self, state: "GameState"):
        return state.ruleset.entry_roles(state, self)

    def get_raw_power(self, state: "GameState") -> int:
        return state.ruleset.entry_raw_power(state, self)

    def get_powerup(self, state: "GameState") -> int:
        return state.ruleset.entry_powerup(state, self)

    def get_total_power(self, state: "GameState") -> int:
        return self.get_raw_power(state) + self.get_powerup(state)


# ----------------------------
# Ruleset (global rules live here)
# ----------------------------

class Ruleset:
    """
    Feature parity goal:
    - Implement the exact global-law behaviors from the notebook, but centralized.
    - CardEntry delegates to Ruleset.
    """

    # ---- Core accessors ----

    def entry_roles(self, state: "GameState", entry: CardEntry):
        if state.active_law == Law.NO_ROLES:
            return []
        return entry.get_card(state).roles

    def entry_raw_power(self, state: "GameState", entry: CardEntry) -> int:
        if state.active_law == Law.RAW_POWER_0:
            return 0

        raw_power = entry.get_card(state).raw_power

        if state.active_law == Law.PROTECTED_2 and entry.status == Status.PROTECTED:
            raw_power += 2

        if state.active_law == Law.SILENCED_3 and entry.status == Status.SILENCED:
            raw_power += 3

        return raw_power

    def entry_skills(self, state: "GameState", entry: CardEntry) -> List[Skill]:
        # base + virtual (virtual unused in Vanilla)
        skills = list(entry.get_card(state).skills) + list(entry.virtual_skills)

        if state.active_law == Law.NO_SILENCE:
            skills = [s for s in skills if s.skill_type != SkillType.SILENCE]

        if state.active_law == Law.NO_POWERUP:
            skills = [s for s in skills if s.skill_type != SkillType.POWERUP]

        return skills

    def entry_powerup(self, state: "GameState", entry: CardEntry) -> int:
        if state.active_law == Law.NO_POWERUP:
            return 0

        # Silenced blocks Powerup (unless future Passive overrides later)
        if entry.status == Status.SILENCED:
            return 0

        # Protected inverse law
        if state.active_law == Law.PROTECTED_INVERSE and entry.status == Status.PROTECTED:
            return 0

        total = 0
        for skill in self.entry_skills(state, entry):
            if skill.skill_type != SkillType.POWERUP:
                continue
            total += self.skill_powerup_value(state, owner=entry.side, this=entry.this, skill=skill)
        return total

    # ---- Skill evaluation ----

    def _condition_met(self, state: "GameState", owner: int, this: int, skill: Skill) -> bool:
        cond = skill.condition
        if cond is None:
            return True
        if isinstance(cond, bool):
            return cond
        return cond.evaluate(state, owner, this)

    def _resolve_value(self, state: "GameState", owner: int, this: int, skill: Skill) -> int:
        v = skill.value
        if v is None:
            return 0
        if isinstance(v, Value):
            return v.evaluate(state, owner, this)
        return int(v)

    def _resolve_count(self, state: "GameState", owner: int, this: int, skill: Skill) -> int:
        c = skill.count
        if c is None:
            return 1
        if isinstance(c, Count):
            return c.evaluate(state, owner, this)
        return int(c)

    def skill_powerup_value(self, state: "GameState", owner: int, this: int, skill: Skill) -> int:
        if not self._condition_met(state, owner, this, skill):
            return 0
        value = self._resolve_value(state, owner, this, skill)
        count = self._resolve_count(state, owner, this, skill)
        return value * count

    def apply_active_skill(self, state: "GameState", owner: int, this: int, skill: Skill) -> None:
        """
        Active skills: SILENCE / PROTECT. Applies status to targets.

        Renathea single-status dominance:
        - If target is Protected, it cannot become Silenced.
        - If target is Silenced, it can become Protected (and then becomes Protected).
        => Protected always wins over Silenced.
        """
        if not self._condition_met(state, owner, this, skill):
            return

        if skill.target is None:
            return

        targets = skill.target.evaluate(state, owner, this)

        if skill.skill_type == SkillType.PROTECT:
            for idx in targets:
                state.entries[idx].status = Status.PROTECTED

        elif skill.skill_type == SkillType.SILENCE:
            for idx in targets:
                e = state.entries[idx]
                if e.status == Status.PROTECTED:
                    continue  # Protected blocks Silence
                e.status = Status.SILENCED


# ----------------------------
# GameState + Resolver + Game
# ----------------------------

@dataclass
class GameState:
    players: List[Player]
    entries: List[CardEntry] = field(default_factory=list)

    active_law: Optional[Law] = None
    ruleset: Ruleset = field(default_factory=Ruleset)
    cache: Dict[str, Any] = field(default_factory=dict)


class Resolver:
    """
    Pipeline (Interpretasi A / simultan):
      1) resolve_continuous()  -> determine active_law (parity behavior)
      2) apply_active_skills() -> apply SILENCE/PROTECT once
      3) scoring happens in Game using ruleset accessors
    """

    def resolve_continuous(self, state: GameState) -> None:
        laws: List[Law] = []
        for entry in state.entries:
            if entry.zone != Zone.ARENA:
                continue
            for skill in state.ruleset.entry_skills(state, entry):
                if skill.timing != SkillTiming.Continuous:
                    continue
                if skill.skill_type != SkillType.LAW:
                    continue
                if skill.law is not None:
                    laws.append(skill.law)

        state.active_law = laws[0] if len(laws) == 1 else None

        if state.active_law == Law.LAW_CANCEL:
            state.active_law = None

    def apply_active_skills(self, state: GameState) -> None:
        for player_idx, _player in enumerate(state.players):
            for entry in state.entries:
                if entry.zone != Zone.ARENA:
                    continue
                if entry.side != player_idx:
                    continue

                for skill in state.ruleset.entry_skills(state, entry):
                    if skill.timing != SkillTiming.Active:
                        continue
                    if skill.skill_type in (SkillType.SILENCE, SkillType.PROTECT):
                        state.ruleset.apply_active_skill(state, owner=player_idx, this=entry.this, skill=skill)


@dataclass
class MatchResult:
    outcomes: List[Result]
    scores: List[int]


class Game:
    """
    Engine core. Runs a single duel (shadowboxing setup helper included).
    """

    def __init__(self, players: List[Player], ruleset: Optional[Ruleset] = None):
        self.state = GameState(players=players, ruleset=ruleset or Ruleset())
        self.resolver = Resolver()

    def setup_all_to_arena(self) -> None:
        entries: List[CardEntry] = []
        for p_idx, p in enumerate(self.state.players):
            for c_idx, _card in enumerate(p.cards):
                entries.append(
                    CardEntry(
                        card=(p_idx, c_idx),
                        zone=Zone.ARENA,
                        side=p_idx,
                    )
                )
        for i, e in enumerate(entries):
            e.this = i
        self.state.entries = entries

    def get_total_power(self, player_idx: int) -> int:
        total = 0
        for entry in self.state.entries:
            if entry.zone == Zone.ARENA and entry.side == player_idx:
                total += entry.get_total_power(self.state)

        if self.state.active_law == Law.POWER_MAX_15:
            total = min(total, 15)
        if self.state.active_law == Law.POWER_MIN_15:
            total = max(total, 15)

        return total

    def run(self, auto_setup_shadowboxing: bool = True) -> MatchResult:
        if auto_setup_shadowboxing and not self.state.entries:
            self.setup_all_to_arena()

        self.resolver.resolve_continuous(self.state)
        self.resolver.apply_active_skills(self.state)

        scores = [self.get_total_power(i) for i in range(len(self.state.players))]

        outcomes: List[Result] = []
        if len(scores) == 2:
            if scores[0] > scores[1]:
                outcomes = [Result.WIN, Result.LOSE]
            elif scores[0] < scores[1]:
                outcomes = [Result.LOSE, Result.WIN]
            else:
                outcomes = [Result.DRAW, Result.DRAW]
        else:
            mx = max(scores) if scores else 0
            for s in scores:
                outcomes.append(Result.WIN if s == mx else Result.LOSE)

        return MatchResult(outcomes=outcomes, scores=scores)


# ----------------------------
# JSON parsing helpers (skills-only)
# ----------------------------

class RawCardFilter(TypedDict, total=False):
    side: str
    zones: List[str]
    subject: str
    roles: List[str]
    power: List[Any]           # [comparison, int|value]
    skill_types: List[str]
    status: str
    exception: "RawCardFilter"
    role_variance: str


class RawCount(TypedDict, total=False):
    mode: str
    card_filter: RawCardFilter
    skill_type: str
    role_variance: str


class RawValue(TypedDict, total=False):
    aggregation: str
    card_filter: RawCardFilter
    multiplier: int


class RawCondition(TypedDict, total=False):
    mode: str
    left: Any
    right: Any
    comparison: str


class RawSkill(TypedDict, total=False):
    text: str
    type: str
    value: Any
    target: RawCardFilter
    count: Any
    condition: Any
    law: str


class RawCard(TypedDict, total=False):
    name: str
    roles: List[str]
    raw_power: int
    skills: List[RawSkill]


def parse_card_filter(data: Optional[RawCardFilter]) -> Optional[CardFilter]:
    if not data:
        return None

    from .dsl import Subject, RoleVariance, Comparison
    from .model import Side, Zone, Role, Status, SkillType

    power = None
    if "power" in data and data["power"] is not None:
        comp = Comparison(data["power"][0])
        rhs = data["power"][1]
        if isinstance(rhs, dict):
            rhs = parse_value(rhs)  # type: ignore[arg-type]
        power = (comp, rhs)

    return CardFilter(
        side=Side(data["side"]) if data.get("side") else None,
        zones=[Zone(z) for z in data["zones"]] if data.get("zones") else None,
        subject=Subject(data["subject"]) if data.get("subject") else None,
        roles=[Role(r) for r in data["roles"]] if data.get("roles") else None,
        power=power,
        skill_types=[SkillType(t) for t in data["skill_types"]] if data.get("skill_types") else None,
        status=Status(data["status"]) if data.get("status") else None,
        exception=parse_card_filter(data["exception"]) if data.get("exception") else None,
        role_variance=RoleVariance(data["role_variance"]) if data.get("role_variance") else None,
    )


def parse_count(data: Any) -> Optional[Union[int, Count]]:
    if data is None:
        return None
    if isinstance(data, int):
        return data
    if isinstance(data, dict):
        from .dsl import CountMode, RoleVariance
        from .model import SkillType

        # Backward-compatible defaults:
        # - if mode missing -> default Card
        # - if card_filter missing -> default {} (match all)
        # - if skill_type present but mode missing -> default Skill
        mode_str = data.get("mode")
        if mode_str is None:
            mode_str = "Skill" if data.get("skill_type") else "Card"

        cf = data.get("card_filter", {}) or {}

        return Count(
            mode=CountMode(mode_str),
            card_filter=parse_card_filter(cf),
            skill_type=SkillType(data["skill_type"]) if data.get("skill_type") else None,
            role_variance=RoleVariance(data["role_variance"]) if data.get("role_variance") else None,
        )
    return None


def parse_value(data: Any) -> Optional[Union[int, Value]]:
    if data is None:
        return None
    if isinstance(data, int):
        return data
    if isinstance(data, dict):
        from .dsl import Aggregation
        return Value(
            aggregation=Aggregation(data["aggregation"]),
            card_filter=parse_card_filter(data["card_filter"]),
            multiplier=int(data.get("multiplier", 1)),
        )
    return None


def parse_condition(data: Any) -> Optional[Union[bool, Condition]]:
    if data is None:
        return None
    if isinstance(data, bool):
        return data
    if isinstance(data, dict):
        from .dsl import ConditionMode, Comparison
        mode = ConditionMode(data["mode"])

        def parse_side(x):
            if isinstance(x, dict):
                if "aggregation" in x:
                    return parse_value(x)
                return parse_count(x)
            return x

        left = parse_side(data["left"])
        right = parse_side(data.get("right"))
        comp = Comparison(data["comparison"]) if data.get("comparison") else None
        return Condition(mode=mode, left=left, right=right, comparison=comp)
    return None


def parse_skill(data: RawSkill) -> Skill:
    from .model import SkillType, Law

    st = SkillType(data["type"])
    return Skill(
        text=data.get("text", ""),
        skill_type=st,
        value=parse_value(data.get("value")),
        target=parse_card_filter(data.get("target")),
        count=parse_count(data.get("count")),
        condition=parse_condition(data.get("condition", True)),
        law=Law(data["law"]) if data.get("law") else None,
    )


def parse_card(data: RawCard) -> Card:
    from .model import Role

    if "skills" not in data:
        raise ValueError(f"Card {data.get('name')} missing 'skills' field")

    raw_skills = data["skills"] or []
    skills = [parse_skill(s) for s in raw_skills]

    return Card(
        name=data["name"],
        roles=[Role(r) for r in data.get("roles", [])],
        raw_power=int(data.get("raw_power", 0)),
        skills=skills,
    )
