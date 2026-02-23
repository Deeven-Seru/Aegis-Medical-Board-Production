"""
Content safety filters for input sanitization and output bias detection.

Provides:
- Input filters: prevent prompt injection and validate clinical text fields.
- Output filters: flag potentially biased or harmful content in agent responses.
"""

import re
from typing import List, Tuple
from .logger import get_logger

logger = get_logger("Filters")

# Patterns that may indicate prompt injection attempts in clinical text fields
_INJECTION_PATTERNS: List[re.Pattern] = [
    re.compile(r"ignore\s+(all\s+)?previous\s+instructions", re.IGNORECASE),
    re.compile(r"disregard\s+(all\s+)?prior\s+(instructions|context)", re.IGNORECASE),
    re.compile(r"you\s+are\s+now\s+a", re.IGNORECASE),
    re.compile(r"system:\s*", re.IGNORECASE),
    re.compile(r"</?(system|user|assistant)\s*>", re.IGNORECASE),
    re.compile(r"(?:^|\n)\s*\[INST\]", re.IGNORECASE),
]

# Terms that may indicate demographic bias leaking into clinical reasoning
_BIAS_INDICATORS: List[re.Pattern] = [
    re.compile(
        r"\b(race|ethnicity|religion|socioeconomic\s+status)\b.{0,80}\b(predispos\w*|inherent\w*|typical\w*|always)\b",
        re.IGNORECASE,
    ),
    re.compile(
        r"\b(men|women|males?|females?)\b.{0,80}\b(always|never|cannot|unable)\b.{0,80}\b(tolerat\w*|recover\w*|present\w*)\b",
        re.IGNORECASE,
    ),
]


def sanitize_input(text: str) -> str:
    """Strip control characters and collapse excessive whitespace."""
    # Remove non-printable control characters (keep newlines and tabs)
    cleaned = re.sub(r"[\x00-\x08\x0b\x0c\x0e-\x1f\x7f]", "", text)
    # Collapse runs of whitespace into single spaces (preserve newlines)
    cleaned = re.sub(r"[^\S\n]+", " ", cleaned)
    return cleaned.strip()


def check_prompt_injection(text: str) -> Tuple[bool, str]:
    """Return (is_safe, reason).  is_safe=True means no injection detected."""
    for pattern in _INJECTION_PATTERNS:
        if pattern.search(text):
            reason = f"Potential prompt injection detected: matched pattern '{pattern.pattern}'"
            logger.warning(reason)
            return False, reason
    return True, ""


def check_output_bias(text: str) -> List[str]:
    """Return a list of bias warnings found in the given output text."""
    warnings: List[str] = []
    for pattern in _BIAS_INDICATORS:
        if pattern.search(text):
            warnings.append(f"Potential bias indicator: matched pattern '{pattern.pattern}'")
    if warnings:
        logger.warning(f"Output bias check raised {len(warnings)} warning(s)")
    return warnings


def filter_patient_case_fields(fields: dict) -> dict:
    """Sanitize free-text fields in a patient case dictionary.

    Returns a new dictionary with sanitized values.  Raises ``ValueError``
    if prompt injection is detected in any field.
    """
    text_keys = [
        "chief_complaint",
        "history_of_present_illness",
        "patient_identifier",
    ]
    sanitized = dict(fields)
    for key in text_keys:
        if key in sanitized and isinstance(sanitized[key], str):
            sanitized[key] = sanitize_input(sanitized[key])
            is_safe, reason = check_prompt_injection(sanitized[key])
            if not is_safe:
                raise ValueError(f"Input rejected for field '{key}': {reason}")

    # Sanitize list-of-string fields
    list_keys = ["past_medical_history", "medications", "imaging_reports"]
    for key in list_keys:
        if key in sanitized and isinstance(sanitized[key], list):
            cleaned_list = []
            for item in sanitized[key]:
                if isinstance(item, str):
                    clean = sanitize_input(item)
                    is_safe, reason = check_prompt_injection(clean)
                    if not is_safe:
                        raise ValueError(f"Input rejected for field '{key}': {reason}")
                    cleaned_list.append(clean)
                else:
                    cleaned_list.append(item)
            sanitized[key] = cleaned_list

    return sanitized
