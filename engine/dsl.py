# engine/dsl.py
from __future__ import annotations

from dataclasses import dataclass
from enum import StrEnum
from typing import List, Optional, Union, Tuple, Dict

from .model import Side, Zone, Role, Status, SkillType


class Subject(StrEnum):
    THIS = "This"
    OTHER = "Other"


class RoleVariance(StrEnum):
    UNIQUE = "Unique"
    DUPLICATED = "Duplicated"


class CountMode(StrEnum):
    CARD = "Card"
    SKILL = "Skill"
    ROLE = "Role"


class ConditionMode(StrEnum):
    PRESENCE = "Presence"
    ABSENCE = "Absence"
    SINGLE = "Single"
    COMPARISON = "Comparison"


class Comparison(StrEnum):
    GREATER = ">"
    GREATER_EQUAL = ">="
    LESS = "<"
    LESS_EQUAL = "<="
    EQUAL = "=="
    NOT_EQUAL = "!="


class Aggregation(StrEnum):
    TOTAL = "total"
    MAX = "max"
    MIN = "min"


@dataclass
class CardFilter:
    side: Optional[Side] = None
    zones: Optional[List[Zone]] = None
    subject: Optional[Subject] = None
    roles: Optional[List[Role]] = None
    power: Optional[Tuple[Comparison, Union[int, "Value"]]] = None
    skill_types: Optional[List[SkillType]] = None
    status: Optional[Status] = None
    exception: Optional["CardFilter"] = None
    role_variance: Optional[RoleVariance] = None

    def _matches_subject(self, this: int, total: int) -> List[int]:
        if self.subject == Subject.THIS:
            return [this]
        if self.subject == Subject.OTHER:
            return [i for i in range(total) if i != this]
        return list(range(total))

    def _matches_side(self, state, you: int, indices: List[int]) -> List[int]:
        if self.side is None:
            return indices
        if self.side == Side.YOU:
            return [i for i in indices if state.entries[i].side == you]
        if self.side == Side.OPPONENT:
            return [i for i in indices if state.entries[i].side != you]
        return indices

    def _matches_zone(self, state, indices: List[int]) -> List[int]:
        if not self.zones:
            return indices
        zones_set = set(self.zones)
        return [i for i in indices if state.entries[i].zone in zones_set]

    def _matches_roles(self, state, indices: List[int]) -> List[int]:
        if not self.roles:
            return indices
        roles_set = set(self.roles)

        out: List[int] = []
        for i in indices:
            entry = state.entries[i]
            entry_roles = state.ruleset.entry_roles(state, entry)
            if any(r in roles_set for r in entry_roles):
                out.append(i)
        return out

    def _matches_status(self, state, indices: List[int]) -> List[int]:
        if self.status is None:
            return indices
        return [i for i in indices if state.entries[i].status == self.status]

    def _matches_power(self, state, you: int, this: int, indices: List[int]) -> List[int]:
        if self.power is None:
            return indices

        comp, rhs = self.power
        rhs_val = rhs.evaluate(state, you, this) if isinstance(rhs, Value) else rhs

        def ok(lhs: int) -> bool:
            if comp == Comparison.GREATER:
                return lhs > rhs_val
            if comp == Comparison.GREATER_EQUAL:
                return lhs >= rhs_val
            if comp == Comparison.LESS:
                return lhs < rhs_val
            if comp == Comparison.LESS_EQUAL:
                return lhs <= rhs_val
            if comp == Comparison.EQUAL:
                return lhs == rhs_val
            if comp == Comparison.NOT_EQUAL:
                return lhs != rhs_val
            return False

        out: List[int] = []
        for i in indices:
            entry = state.entries[i]
            lhs = state.ruleset.entry_raw_power(state, entry)
            if ok(lhs):
                out.append(i)
        return out

    def _matches_skill_types(self, state, indices: List[int]) -> List[int]:
        if not self.skill_types:
            return indices
        types_set = set(self.skill_types)

        out: List[int] = []
        for i in indices:
            entry = state.entries[i]
            skills = state.ruleset.entry_skills(state, entry)
            if any(s.skill_type in types_set for s in skills):
                out.append(i)
        return out

    def _apply_role_variance(self, state, indices: List[int]) -> List[int]:
        if self.role_variance is None:
            return indices

        role_counts: Dict[Role, int] = {}
        for i in indices:
            for r in state.ruleset.entry_roles(state, state.entries[i]):
                role_counts[r] = role_counts.get(r, 0) + 1

        def keep_entry(i: int) -> bool:
            entry_roles = state.ruleset.entry_roles(state, state.entries[i])
            if self.role_variance == RoleVariance.UNIQUE:
                return any(role_counts.get(r, 0) == 1 for r in entry_roles)
            if self.role_variance == RoleVariance.DUPLICATED:
                return any(role_counts.get(r, 0) >= 2 for r in entry_roles)
            return True

        return [i for i in indices if keep_entry(i)]

    def evaluate(self, state, you: int, this: int) -> List[int]:
        indices = self._matches_subject(this, total=len(state.entries))
        indices = self._matches_side(state, you, indices)
        indices = self._matches_zone(state, indices)
        indices = self._matches_status(state, indices)
        indices = self._matches_roles(state, indices)
        indices = self._matches_skill_types(state, indices)
        indices = self._matches_power(state, you, this, indices)
        indices = self._apply_role_variance(state, indices)

        if self.exception is not None:
            excluded = set(self.exception.evaluate(state, you, this))
            indices = [i for i in indices if i not in excluded]

        return indices


@dataclass
class Count:
    mode: CountMode
    card_filter: CardFilter
    skill_type: Optional[SkillType] = None
    role_variance: Optional[RoleVariance] = None

    def evaluate(self, state, you: int, this: int) -> int:
        indices = self.card_filter.evaluate(state, you, this)

        if self.mode == CountMode.CARD:
            return len(indices)

        if self.mode == CountMode.SKILL:
            total = 0
            for i in indices:
                entry = state.entries[i]
                skills = state.ruleset.entry_skills(state, entry)
                if self.skill_type is None:
                    total += len(skills)
                else:
                    total += sum(1 for s in skills if s.skill_type == self.skill_type)
            return total

        if self.mode == CountMode.ROLE:
            roles: List[Role] = []
            for i in indices:
                roles.extend(state.ruleset.entry_roles(state, state.entries[i]))

            if not roles:
                return 0

            if self.role_variance is None:
                return len(roles)

            from collections import Counter
            c = Counter(roles)
            if self.role_variance == RoleVariance.UNIQUE:
                return sum(1 for _r, n in c.items() if n == 1)
            if self.role_variance == RoleVariance.DUPLICATED:
                return sum(1 for _r, n in c.items() if n >= 2)
            return len(roles)

        return 0


@dataclass
class Value:
    aggregation: Aggregation
    card_filter: CardFilter
    multiplier: int = 1

    def evaluate(self, state, you: int, this: int) -> int:
        indices = self.card_filter.evaluate(state, you, this)
        values = [state.ruleset.entry_raw_power(state, state.entries[i]) for i in indices]

        if not values:
            return 0

        if self.aggregation == Aggregation.TOTAL:
            return sum(values) * self.multiplier
        if self.aggregation == Aggregation.MAX:
            return max(values) * self.multiplier
        if self.aggregation == Aggregation.MIN:
            return min(values) * self.multiplier
        return 0


@dataclass
class Condition:
    mode: ConditionMode
    left: Union[int, Count, Value]
    right: Optional[Union[int, Count, Value]] = None
    comparison: Optional[Comparison] = None

    def _eval_side(self, x, state, you: int, this: int) -> int:
        if isinstance(x, (Count, Value)):
            return x.evaluate(state, you, this)
        return int(x)

    def evaluate(self, state, you: int, this: int) -> bool:
        if self.mode in (ConditionMode.PRESENCE, ConditionMode.ABSENCE, ConditionMode.SINGLE):
            left_val = self._eval_side(self.left, state, you, this)
            if self.mode == ConditionMode.PRESENCE:
                return left_val > 0
            if self.mode == ConditionMode.ABSENCE:
                return left_val == 0
            if self.mode == ConditionMode.SINGLE:
                return left_val == 1

        if self.mode == ConditionMode.COMPARISON:
            if self.right is None or self.comparison is None:
                return False
            l = self._eval_side(self.left, state, you, this)
            r = self._eval_side(self.right, state, you, this)

            if self.comparison == Comparison.GREATER:
                return l > r
            if self.comparison == Comparison.GREATER_EQUAL:
                return l >= r
            if self.comparison == Comparison.LESS:
                return l < r
            if self.comparison == Comparison.LESS_EQUAL:
                return l <= r
            if self.comparison == Comparison.EQUAL:
                return l == r
            if self.comparison == Comparison.NOT_EQUAL:
                return l != r
        return False


# logical conditions
class BoolOp(StrEnum):
    AND = "and"
    OR = "or"
    NOT = "not"


ConditionLike = Union[bool, Condition, "BoolCondition"]


@dataclass
class BoolCondition:
    op: BoolOp
    items: List[ConditionLike]

    def evaluate(self, state, you: int, this: int) -> bool:
        if self.op == BoolOp.AND:
            return all(_eval_condition(x, state, you, this) for x in self.items)
        if self.op == BoolOp.OR:
            return any(_eval_condition(x, state, you, this) for x in self.items)
        if self.op == BoolOp.NOT:
            if not self.items:
                return True
            return not _eval_condition(self.items[0], state, you, this)
        return False


def _eval_condition(x: ConditionLike, state, you: int, this: int) -> bool:
    if x is None:
        return True
    if isinstance(x, bool):
        return x
    return x.evaluate(state, you, this)


@dataclass
class SkillFilter:
    card_filter: CardFilter
    include: Optional[List[SkillType]] = None
    exclude: Optional[List[SkillType]] = None
