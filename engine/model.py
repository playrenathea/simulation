# engine/model.py
from __future__ import annotations

from dataclasses import dataclass, field
from enum import StrEnum
from typing import List, Optional, Union, Dict, Any, TYPE_CHECKING


# ----------------------------
# Enums (grammar / data layer)
# ----------------------------

class Side(StrEnum):
    YOU = "You"
    OPPONENT = "Opponent"


class Zone(StrEnum):
    ARENA = "Arena"
    BACKSTAGE = "Backstage"


class Role(StrEnum):
    TANK = "Tank"
    MELEE = "Melee"
    AGILITY = "Agility"
    RANGED = "Ranged"
    MAGIC = "Magic"
    SUPPORT = "Support"


class Status(StrEnum):
    PROTECTED = "Protected"
    SILENCED = "Silenced"


class SkillType(StrEnum):
    POWERUP = "Powerup"
    SILENCE = "Silence"
    PROTECT = "Protect"
    LAW = "Law"
    PASSIVE = "Passive"
    COPY = "Copy"


class SkillTiming(StrEnum):
    Continuous = "Continuous"  # Law / Passive / Copy
    Active = "Active"          # Silence / Protect
    Power = "Power"            # Powerup


class Result(StrEnum):
    WIN = "win"
    LOSE = "lose"
    DRAW = "draw"


class Law(StrEnum):
    # parity / existing
    NO_ROLES = "no_roles"
    NO_SILENCE = "no_silence"
    NO_POWERUP = "no_powerup"
    PROTECTED_INVERSE = "protected_inverse"
    RAW_POWER_0 = "raw_power_0"
    PROTECTED_2 = "protected_2"
    SILENCED_3 = "silenced_3"
    POWER_MAX_15 = "power_max_15"
    POWER_MIN_15 = "power_min_15"
    LAW_CANCEL = "law_cancel"

    # NEW
    REMOVE_TIDAK = "remove_tidak"


class Passive(StrEnum):
    GENERALIST = "generalist"
    BERSERKER = "berserker"
    FREEFOLK = "freefolk"
    CHANTER = "chanter"
    DEFLECTOR = "deflector"


# ----------------------------
# Core data classes (no rules)
# ----------------------------

if TYPE_CHECKING:
    from .dsl import CardFilter, Count, Value, ConditionLike, SkillFilter


@dataclass
class Skill:
    text: str
    skill_type: SkillType
    timing: Optional[SkillTiming] = None

    value: Optional[Union[int, "Value"]] = None
    target: Optional["CardFilter"] = None
    count: Optional[Union[int, "Count"]] = None
    condition: Optional["ConditionLike"] = True

    # Copy-specific
    skill_filter: Optional["SkillFilter"] = None

    # Law/Passive payload
    law: Optional[Law] = None
    passive: Optional[Passive] = None

    params: Dict[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if self.timing is None:
            if self.skill_type == SkillType.POWERUP:
                self.timing = SkillTiming.Power
            elif self.skill_type in (SkillType.SILENCE, SkillType.PROTECT):
                self.timing = SkillTiming.Active
            else:
                self.timing = SkillTiming.Continuous

        if not isinstance(self.text, str):
            raise TypeError("Skill.text must be a str")
        if not isinstance(self.skill_type, SkillType):
            raise TypeError("Skill.skill_type must be a SkillType enum")


@dataclass
class Card:
    name: str
    roles: List[Role]
    raw_power: int
    skills: List[Skill]

    def __post_init__(self) -> None:
        if not isinstance(self.name, str) or not self.name:
            raise ValueError("Card.name must be a non-empty string")
        if not isinstance(self.raw_power, int):
            raise TypeError("Card.raw_power must be int")
        if self.raw_power < 0:
            raise ValueError("Card.raw_power must be >= 0")
        if any(not isinstance(r, Role) for r in self.roles):
            raise TypeError("Card.roles must be a list[Role]")
        if any(not isinstance(s, Skill) for s in self.skills):
            raise TypeError("Card.skills must be a list[Skill]")


@dataclass
class Player:
    name: str
    cards: List[Card]

    def __str__(self) -> str:
        card_names = ", ".join(c.name for c in self.cards)
        return f"{self.name} ({card_names})"
