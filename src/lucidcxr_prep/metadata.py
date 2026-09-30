"""Structured extraction from the free-text ClinicalReadings files.

Only mechanically unambiguous fields are extracted: sex and age, both stated directly and
without interpretation in the source text. The remaining diagnosis narrative is carried forward
verbatim, not summarized or classified -- deciding what counts as "active" vs "inactive" TB from
free text is a clinical judgment call, not an engineering one (see conversation record), so no
such flag is derived here. Montgomery text in particular mixes both in one note (e.g. "old
inactive disease in RL and new active TB in LL") and hedges ("?active") -- exactly the kind of
case a keyword regex would get wrong silently.
"""

from __future__ import annotations

import re

MONT_RE = re.compile(
    # Sex is not always M/F in this source -- e.g. patient 0080 is coded "O". Accept any single
    # letter and pass it through unchanged rather than guessing what a non-M/F code means.
    r"Patient's Sex:\s*(\S)\s*\n"
    r"Patient's Age:\s*0*(\d+)Y\s*\n?"
    r"(.*)",
    re.IGNORECASE | re.DOTALL,
)
SHEN_RE = re.compile(
    # Verified on the full 662 rows, not assumed -- the source has several inconsistencies:
    # a typo ("femal" for "female"), no space before the age ("female24yrs"), and a trailing
    # comma ("male ,"). "female" must come before "femal" in the alternation or the shorter
    # match would leave a dangling "e" that breaks the rest of the pattern.
    r"\s*(female|femal|male)\s*,?\s*(\d+)\s*(yrs?|years?|months?|days?)?\.?\s*\n?"
    r"(.*)",
    re.IGNORECASE | re.DOTALL,
)

# Classify by first letter, not by stripping a trailing "s" -- that approach silently broke on
# "yrs".rstrip("s") == "yr" (not "y"), which matched no dict key and turned 658 of 662 rows into
# NaN with no error raised (caught only because the printed age range looked infant-only).
_FIRST_LETTER_TO_UNIT = {"y": "years", "m": "months", "d": "days"}
_UNIT_TO_YEAR_DIVISOR = {"years": 1, "months": 12, "days": 365.25}


def _unit_key(raw_unit: str | None) -> str | None:
    if raw_unit is None:
        return None
    return _FIRST_LETTER_TO_UNIT.get(raw_unit[0].lower())


def parse_montgomery(clinical_text: str) -> dict:
    m = MONT_RE.match(clinical_text)
    if not m:
        raise ValueError(f"Unrecognized Montgomery clinical_text format: {clinical_text!r}")
    sex, age, rest = m.groups()
    return {
        "sex": sex.upper(),
        "age_value": int(age),
        "age_unit": "years",
        "age_years": float(age),  # unambiguous in this source -- always 'Y' (verified, 138/138)
        "diagnosis_text": rest.strip(),
    }


def parse_shenzhen(clinical_text: str) -> dict:
    m = SHEN_RE.match(clinical_text)
    if not m:
        raise ValueError(f"Unrecognized Shenzhen clinical_text format: {clinical_text!r}")
    sex, age, raw_unit, rest = m.groups()
    unit_key = _unit_key(raw_unit)
    divisor = _UNIT_TO_YEAR_DIVISOR.get(unit_key)
    return {
        "sex": sex[0].upper(),
        "age_value": int(age),
        # None when the source gave no unit at all (1 of 662 rows) -- age_years is left NaN for
        # that row rather than guessing "years", since a bare number could plausibly be any unit.
        "age_unit": unit_key,
        "age_years": (int(age) / divisor) if divisor else float("nan"),
        "diagnosis_text": rest.strip(),
    }
