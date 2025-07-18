# preprocessing.py
import re

def normalize_text(text: str) -> str:
    """
    Normalize input text:
      - remove extra spaces
      - unify punctuation
    """
    # Remove multiple spaces and strip
    normalized = re.sub(r"\s+", " ", text).strip()
    return normalized

def split_into_segments(text: str):
    """
    Split normalized text into sentences or reasoning units.
    Simple version based on period punctuation.
    """
    segments = [seg.strip() for seg in re.split(r"[.؟!]", text) if seg.strip()]
    return segments

