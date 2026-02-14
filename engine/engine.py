# engine/engine.py
from __future__ import annotations

from dataclasses import dataclass, field
from typing import List, Optional, Tuple, Union, Any, Dict, TypedDict

from .model import (
    Player, Card, Skill, SkillType, SkillTiming,
    Zone, Status, Law, Result, Role, Passive
)
from .dsl import (
    CardFilter, Count, Value, Condition,
    BoolCondition, BoolOp, ConditionLike,
    SkillFilter, Comparison, ConditionMode
)


# ----------------------------
# Runtime state wrapper
# ----------------------------

@dataclass
class CardEntry:
    card: Tuple[int, int]                 # (player_index, card_index)
    zone: Optional[Zone] = None
    side: int = 0                         # owner index
    this: int = 0                         # index in state.entries

    status: Optional[Status] = None

    visibility: List[int] = field(default_factory=list)

    # Continuous injections
    virtual_skills: List[Skill] = field(default_factory=list)
    virtual_roles: List[Role] = field(default_factory=list)

    # Passive flags
    ignore_law: bool = False
    powerup_while_silenced: bool = False
    has_deflector: bool = False

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
# Ruleset
# ----------------------------

class Ruleset:
    def is_copied_skill(self, skill: Skill) -> bool:
        return bool(skill.params.get("_copied", False))

    def _law_for_entry(self, state: "GameState", entry: CardEntry) -> Optional[Law]:
        return None if entry.ignore_law else state.active_law

    def entry_roles(self, state: "GameState", entry: CardEntry):
        law = self._law_for_entry(state, entry)
        if law == Law.NO_ROLES:
            return []
        return list(entry.get_card(state).roles) + list(entry.virtual_roles)

    def entry_raw_power(self, state: "GameState", entry: CardEntry) -> int:
        law = self._law_for_entry(state, entry)
        if law == Law.RAW_POWER_0:
            return 0

        raw_power = entry.get_card(state).raw_power

        if law == Law.PROTECTED_2 and entry.status == Status.PROTECTED:
            raw_power += 2

        if law == Law.SILENCED_3 and entry.status == Status.SILENCED:
            raw_power += 3

        return raw_power

    def entry_skills(self, state: "GameState", entry: CardEntry) -> List[Skill]:
        law = self._law_for_entry(state, entry)

        skills = list(entry.get_card(state).skills) + list(entry.virtual_skills)

        if law == Law.NO_SILENCE:
            skills = [s for s in skills if s.skill_type != SkillType.SILENCE]

        if law == Law.NO_POWERUP:
            skills = [s for s in skills if s.skill_type != SkillType.POWERUP]

        return skills

    # -------- Technomancer: remove_tidak condition transformer --------

    def _remove_tidak_transform(self, cond: ConditionLike) -> ConditionLike:
        # bool
        if isinstance(cond, bool):
            return cond

        # BoolCondition
        if isinstance(cond, BoolCondition):
            if cond.op == BoolOp.NOT:
                # remove NOT: {"not": X} -> X (transformed)
                if not cond.items:
                    return True
                inner = cond.items[0]
                return self._remove_tidak_transform(inner)
            # AND/OR: recurse
            return BoolCondition(
                op=cond.op,
                items=[self._remove_tidak_transform(x) for x in cond.items],
            )

        # Condition
        if isinstance(cond, Condition):
            mode = cond.mode
            left = cond.left
            right = cond.right
            comp = cond.comparison

            # Absence -> Presence
            if mode == ConditionMode.ABSENCE:
                mode = ConditionMode.PRESENCE

            # "!=" -> "=="
            if mode == ConditionMode.COMPARISON and comp == Comparison.NOT_EQUAL:
                comp = Comparison.EQUAL

            return Condition(mode=mode, left=left, right=right, comparison=comp)

        return cond

    # ---------------------------------------------------------------

    def _condition_met(self, state, owner, this, skill, entry=None) -> bool:
        cond = skill.condition
        if cond is None:
            return True
    
        # Dogmatist: if_fail -> semua kondisi "jika" dianggap gagal
        # (kita treat: kalau condition itu struktur (Condition/BoolCondition) => fail)
        if state.active_law == Law.IF_FAIL and entry is not None and not entry.ignore_law:
            if isinstance(cond, (Condition, BoolCondition)):
                return False
            if isinstance(cond, bool):
                return cond
            return False
    
        # Technomancer: remove_tidak transform
        if state.active_law == Law.REMOVE_TIDAK and entry is not None and not entry.ignore_law:
            if isinstance(cond, (bool, Condition, BoolCondition)):
                cond = self._remove_tidak_transform(cond)
    
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

    def skill_powerup_value(self, state, owner, this, skill, entry):
        if not self._condition_met(state, owner, this, skill, entry=entry):
            return 0
    
        # Artificer: semua Powerup jadi (+3)
        if state.active_law == Law.POWERUP_3 and not entry.ignore_law:
            value = 3
        else:
            value = self._resolve_value(state, owner, this, skill)
    
        count = self._resolve_count(state, owner, this, skill)
        return value * count

    def entry_powerup(self, state: "GameState", entry: CardEntry) -> int:
        law = self._law_for_entry(state, entry)

        if law == Law.NO_POWERUP:
            return 0

        if entry.status == Status.SILENCED and not entry.powerup_while_silenced:
            return 0

        if law == Law.PROTECTED_INVERSE and entry.status == Status.PROTECTED:
            return 0

        total = 0
        for skill in self.entry_skills(state, entry):
            if skill.skill_type != SkillType.POWERUP:
                continue
            total += self.skill_powerup_value(state, owner=entry.side, this=entry.this, skill=skill, entry=entry)
        return total

    def apply_active_skill(self, state: "GameState", owner: int, this: int, skill: Skill) -> None:
        if skill.target is None:
            return

        src_entry = state.entries[this]
        if not self._condition_met(state, owner, this, skill, entry=src_entry):
            return

        targets = skill.target.evaluate(state, owner, this)

        # Deflector reflect
        if targets:
            reflected = [idx for idx in targets if state.entries[idx].has_deflector]
            if reflected:
                targets = [idx for idx in targets if idx not in set(reflected)]
                if this not in targets:
                    targets.append(this)

        if skill.skill_type == SkillType.PROTECT:
            for idx in targets:
                state.entries[idx].status = Status.PROTECTED

        elif skill.skill_type == SkillType.SILENCE:
            for idx in targets:
                e = state.entries[idx]
                if e.status == Status.PROTECTED:
                    continue
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

    owner_power_bonus: List[int] = field(default_factory=list)


class Resolver:
    def _reset_continuous(self, state: GameState) -> None:
        state.active_law = None
        state.owner_power_bonus = [0 for _ in range(len(state.players))]
        for e in state.entries:
            e.virtual_skills = []
            e.virtual_roles = []
            e.ignore_law = False
            e.powerup_while_silenced = False
            e.has_deflector = False

    def _build_copy_virtuals_once(self, state: GameState) -> bool:
        before = [(e.this, len(e.virtual_skills)) for e in state.entries]
        for e in state.entries:
            e.virtual_skills = []

        for owner_idx, _p in enumerate(state.players):
            for entry in state.entries:
                if entry.zone != Zone.ARENA:
                    continue
                if entry.side != owner_idx:
                    continue

                for sk in entry.get_card(state).skills:
                    if sk.skill_type != SkillType.COPY:
                        continue
                    if sk.timing != SkillTiming.Continuous:
                        continue
                    if sk.skill_filter is None:
                        continue
                    if not state.ruleset._condition_met(state, owner_idx, entry.this, sk, entry=entry):
                        continue

                    sf = sk.skill_filter
                    source_indices = sf.card_filter.evaluate(state, owner_idx, entry.this)

                    collected: List[Skill] = []
                    for i in source_indices:
                        src_entry = state.entries[i]
                        src_skills = list(src_entry.get_card(state).skills)  # base only

                        for s2 in src_skills:
                            if s2.skill_type == SkillType.COPY:
                                continue
                            if state.ruleset.is_copied_skill(s2):
                                continue
                            collected.append(s2)

                    if sf.include:
                        inc = set(sf.include)
                        collected = [s2 for s2 in collected if s2.skill_type in inc]
                    if sf.exclude:
                        exc = set(sf.exclude)
                        collected = [s2 for s2 in collected if s2.skill_type not in exc]

                    injected: List[Skill] = []
                    for s2 in collected:
                        cloned = Skill(
                            text=s2.text,
                            skill_type=s2.skill_type,
                            timing=s2.timing,
                            value=s2.value,
                            target=s2.target,
                            count=s2.count,
                            condition=s2.condition,
                            skill_filter=None,
                            law=s2.law,
                            passive=s2.passive,
                            params=dict(s2.params),
                        )
                        cloned.params["_copied"] = True
                        injected.append(cloned)

                    if injected:
                        entry.virtual_skills.extend(injected)

        after = [(e.this, len(e.virtual_skills)) for e in state.entries]
        return before != after

    def apply_copy_skills(self, state: GameState) -> None:
        for _ in range(5):
            if not self._build_copy_virtuals_once(state):
                break

    def apply_passives(self, state: GameState) -> None:
        def roles_in_your_arena_excluding(owner: int, this_entry: int) -> set[Role]:
            out: set[Role] = set()
            for e in state.entries:
                if e.zone != Zone.ARENA:
                    continue
                if e.side != owner:
                    continue
                if e.this == this_entry:
                    continue
                out.update(state.ruleset.entry_roles(state, e))
            return out

        for entry in state.entries:
            if entry.zone != Zone.ARENA:
                continue

            owner = entry.side
            for sk in state.ruleset.entry_skills(state, entry):
                if sk.skill_type != SkillType.PASSIVE:
                    continue
                if sk.timing != SkillTiming.Continuous:
                    continue
                if sk.passive is None:
                    continue
                if not state.ruleset._condition_met(state, owner, entry.this, sk, entry=entry):
                    continue

                if sk.passive == Passive.FREEFOLK:
                    entry.ignore_law = True

                elif sk.passive == Passive.BERSERKER:
                    entry.powerup_while_silenced = True

                elif sk.passive == Passive.CHANTER:
                    state.owner_power_bonus[owner] += 7

                elif sk.passive == Passive.DEFLECTOR:
                    entry.has_deflector = True

                elif sk.passive == Passive.GENERALIST:
                    present = roles_in_your_arena_excluding(owner, entry.this)
                    base_roles = set(entry.get_card(state).roles)
                    already_virtual = set(entry.virtual_roles)

                    for r in Role:
                        if r in present:
                            continue
                        if r in base_roles:
                            continue
                        if r in already_virtual:
                            continue
                        entry.virtual_roles.append(r)

    def resolve_active_law(self, state: GameState) -> None:
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

    def resolve_continuous(self, state: GameState) -> None:
        self._reset_continuous(state)
        self.apply_copy_skills(state)    # Copy first
        self.apply_passives(state)       # Passive second
        self.resolve_active_law(state)   # Law third

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
    def __init__(self, players: List[Player], ruleset: Optional[Ruleset] = None):
        self.state = GameState(players=players, ruleset=ruleset or Ruleset())
        self.resolver = Resolver()

    def setup_all_to_arena(self) -> None:
        entries: List[CardEntry] = []
        for p_idx, p in enumerate(self.state.players):
            for c_idx, _card in enumerate(p.cards):
                entries.append(CardEntry(card=(p_idx, c_idx), zone=Zone.ARENA, side=p_idx))
        for i, e in enumerate(entries):
            e.this = i
        self.state.entries = entries
        self.state.owner_power_bonus = [0 for _ in range(len(self.state.players))]

    def get_total_power(self, player_idx: int) -> int:
        total = 0
        for entry in self.state.entries:
            if entry.zone == Zone.ARENA and entry.side == player_idx:
                total += entry.get_total_power(self.state)

        if self.state.owner_power_bonus:
            total += self.state.owner_power_bonus[player_idx]

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
# JSON parsing helpers
# ----------------------------

class RawCardFilter(TypedDict, total=False):
    side: str
    zones: List[str]
    subject: str
    roles: List[str]
    power: List[Any]
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


class RawSkillFilter(TypedDict, total=False):
    card_filter: RawCardFilter
    include: List[str]
    exclude: List[str]


class RawSkill(TypedDict, total=False):
    text: str
    type: str
    value: Any
    target: RawCardFilter
    count: Any
    condition: Any
    law: str
    passive: str
    skill_filter: RawSkillFilter


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
            rhs = parse_value(rhs)
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

        mode_str = data.get("mode")
        if mode_str is None:
            mode_str = "Skill" if data.get("skill_type") else "Card"

        cf = parse_card_filter(data.get("card_filter", {}) or {}) or CardFilter()

        return Count(
            mode=CountMode(mode_str),
            card_filter=cf,
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
            card_filter=parse_card_filter(data["card_filter"]) or CardFilter(),
            multiplier=int(data.get("multiplier", 1)),
        )
    return None


def parse_condition(data: Any) -> Optional[ConditionLike]:
    if data is None:
        return None
    if isinstance(data, bool):
        return data
    if isinstance(data, dict):
        if "and" in data:
            items = [parse_condition(x) for x in (data.get("and") or [])]
            items = [x for x in items if x is not None]
            return BoolCondition(op=BoolOp.AND, items=items)
        if "or" in data:
            items = [parse_condition(x) for x in (data.get("or") or [])]
            items = [x for x in items if x is not None]
            return BoolCondition(op=BoolOp.OR, items=items)
        if "not" in data:
            item = parse_condition(data.get("not"))
            return BoolCondition(op=BoolOp.NOT, items=[item] if item is not None else [])

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


def parse_skill_filter(data: Any) -> Optional[SkillFilter]:
    if not data or not isinstance(data, dict):
        return None

    from .model import SkillType

    cf = parse_card_filter(data.get("card_filter")) or CardFilter()
    inc = data.get("include")
    exc = data.get("exclude")

    return SkillFilter(
        card_filter=cf,
        include=[SkillType(x) for x in inc] if inc else None,
        exclude=[SkillType(x) for x in exc] if exc else None,
    )


def parse_skill(data: RawSkill) -> Skill:
    from .model import SkillType, Law, Passive

    st = SkillType(data["type"])
    return Skill(
        text=data.get("text", ""),
        skill_type=st,
        value=parse_value(data.get("value")),
        target=parse_card_filter(data.get("target")),
        count=parse_count(data.get("count")),
        condition=parse_condition(data.get("condition", True)),
        law=Law(data["law"]) if data.get("law") else None,
        passive=Passive(data["passive"]) if data.get("passive") else None,
        skill_filter=parse_skill_filter(data.get("skill_filter")),
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
