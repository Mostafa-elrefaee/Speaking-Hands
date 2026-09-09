"""Label -> spoken text normalisation. Deliberately independent of the model.

    _word(label)        internal label -> raw word(s) (label_map or de-underscored label)
    finalize(text)      whitespace, capitalization, punctuation -> the utterance
    label_to_text       _word + finalize   (immediate_word mode)
    join_phrase(words)  join + finalize    (buffered_phrase mode)

``label_map`` values are trusted as written ("I love you" keeps its "I");
raw labels are lower-cased unless ``capitalization: none``.
"""
from __future__ import annotations

import re
from typing import List, Optional

from ..config import TTSConfig

_WS = re.compile(r"\s+")


def _clean(text: str) -> str:
    return _WS.sub(" ", str(text)).strip()


def word_for_label(label: str, cfg: TTSConfig) -> Optional[str]:
    if label is None:
        return None
    key = str(label).strip()
    if key in cfg.label_map:
        text = _clean(cfg.label_map[key])
    elif cfg.unknown_label_policy == "skip":
        return None
    else:  # speak_label: the raw label (optionally de-underscored)
        text = _clean(key.replace("_", " ").replace("-", " ") if cfg.split_label_words else key)
        if cfg.capitalization != "none":
            text = text.lower()
    return text or None


def finalize(text: Optional[str], cfg: TTSConfig) -> Optional[str]:
    if text is None:
        return None
    text = _clean(text)
    if not text:
        return None
    if cfg.capitalization == "lower":
        text = text.lower()
    elif cfg.capitalization == "sentence":
        text = text[0].upper() + text[1:]
    if cfg.punctuation and text[-1] not in ".!?":
        text += "."
    return text


def normalize_text(text: str, cfg: TTSConfig) -> Optional[str]:
    return finalize(text, cfg)


def label_to_text(label: str, cfg: TTSConfig) -> Optional[str]:
    return finalize(word_for_label(label, cfg), cfg)


def join_phrase(words: List[str], cfg: TTSConfig) -> Optional[str]:
    words = [w for w in (_clean(w) for w in words if w) if w]
    return finalize(" ".join(words), cfg) if words else None
