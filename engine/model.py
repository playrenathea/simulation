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
    # As requested (maps to old notebook semantics)
    Continuous = "Continuous"  # Law / Passive / Copy
    Active = "Active"          # Silence / Protect
    Power = "Power"            # Powerup


class Result(StrEnum):
    WIN = "win"
    LOSE = "lose"
    DRAW = "draw"


class Law(StrEnum):
    # Keep identical to notebook (feature parity)
    NO_ROLES = "no_roles"                    # affects roles (treated as empty)
    NO_SILENCE = "no_silence"                # remove Silence skills
    NO_POWERUP = "no_powerup"                # remove Powerup skills and Powerup calc
    PROTECTED_INVERSE = "protected_inverse"  # if protected, powerup becomes 0
    RAW_POWER_0 = "raw_power_0"              # raw power becomes 0
    PROTECTED_2 = "protected_2"              # +2 raw power if protected
    SILENCED_3 = "silenced_3"                # +3 raw power if silenced
    POWER_MAX_15 = "power_max_15"            # cap total power per player
    POWER_MIN_15 = "power_min_15"            # floor total power per player
    LAW_CANCEL = "law_cancel"                # cancels law selection


class Passive(StrEnum):
    # Placeholder for future stages (Passive enum requested)
    _PLACEHOLDER = "placeholder"


# ----------------------------
# Core data classes (no rules)
# ----------------------------

if TYPE_CHECKING:
    from .dsl import CardFilter, Count, Value, Condition, LogicCondition


@dataclass
class Skill:
    # Keep fields aligned to old Ability (feature parity)
    text: str
    skill_type: SkillType
    timing: Optional[SkillTiming] = None

    # These mirror old Ability fields; interpreted by engine/dsl
    value: Optional[Union[int, "Value"]] = None
    target: Optional["CardFilter"] = None
    count: Optional[Union[int, "Count"]] = None
    condition: Optional[Union[bool, "Condition", "LogicCondition"]] = True

    # Law/Passive payload (enum)
    law: Optional[Law] = None
    passive: Optional[Passive] = None

    # Future-proof params (Copy etc). Not used in Vanilla.
    params: Dict[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        # "Validasi ringan" + default timing mapping (same semantics as notebook)
        if self.timing is None:
            if self.skill_type == SkillType.POWERUP:
                self.timing = SkillTiming.Power
            elif self.skill_type in (SkillType.SILENCE, SkillType.PROTECT):
                self.timing = SkillTiming.Active
            else:
                self.timing = SkillTiming.Continuous

        # minimal guardrails
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
