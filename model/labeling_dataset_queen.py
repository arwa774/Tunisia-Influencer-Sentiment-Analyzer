"""
Sentiment Classification using Qwen2.5-7B-Instruct (BitsAndBytes 8-bit) via Hugging Face.

  ① Emoji hybrid boost  — emojis in mixed comments can override/boost LLM label
  ② Richer system prompt — Darja-aware few-shot examples + key vocab glossary
  ③ Expanded emoji lexicon — added ❣️ and other common missing emojis
  ④ Batch inference: 20 comments per LLM call  (~10-15x faster)
  ⑤ Checkpoint save: every 100 comments        (crash-safe)
  ⑥ Auto-resume:     skips already-done rows   (restart-safe)
  ⑦ Live printing:   result shown per comment  (quality monitoring)
"""

import argparse
import os
import sys
import unicodedata
import pandas as pd
import json
import re
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig
from tqdm import tqdm

# ── Default settings ──────────────────────────────────────────────────────────
DEFAULT_INPUT  = "/kaggle/input/datasets/wassimsghaier/comments/final_cleaned_youtube_comments.csv"
DEFAULT_OUTPUT = "/kaggle/working/result_hf_4bit_new.csv"
DEFAULT_COLUMN = "comment"
# MODEL_ID       = "Qwen/Qwen2.5-3B-Instruct"
# MODEL_ID       = "CohereForAI/aya-expanse-8b"
MODEL_ID       = "Qwen/Qwen2.5-7B-Instruct"
BATCH_SIZE = 20   # comments per LLM prompt
SAVE_EVERY = 100  # save checkpoint every N processed comments

# ──────────────────────────────────────────────────────────────────────────────
#  IMPROVEMENT ①+③ — EMOJI OVERRIDE THRESHOLD
#  When the LLM says Neutral but emojis in the same comment strongly signal
#  Positive or Negative (confidence ≥ threshold), the emoji wins.
# ──────────────────────────────────────────────────────────────────────────────
EMOJI_OVERRIDE_THRESHOLD = 0.65


# ══════════════════════════════════════════════════════════════════════════════
#  EMOJI SENTIMENT LEXICON
#  ③ NEW entries marked with  ← NEW
# ══════════════════════════════════════════════════════════════════════════════
EMOJI_SENTIMENT: dict[str, tuple[float, float, float]] = {
    # ── Very positive ─────────────────────────────────────────────────────────
    "\U0001F600": (0.95, 0.02, 0.03),  # 😀
    "\U0001F603": (0.95, 0.02, 0.03),  # 😃
    "\U0001F604": (0.95, 0.02, 0.03),  # 😄
    "\U0001F601": (0.92, 0.02, 0.06),  # 😁
    "\U0001F606": (0.90, 0.02, 0.08),  # 😆
    "\U0001F605": (0.70, 0.05, 0.25),  # 😅
    "\U0001F923": (0.92, 0.01, 0.07),  # 🤣
    "\U0001F602": (0.88, 0.02, 0.10),  # 😂
    "\U0001F642": (0.78, 0.05, 0.17),  # 🙂
    "\U0001F60A": (0.90, 0.02, 0.08),  # 😊
    "\U0001F607": (0.88, 0.02, 0.10),  # 😇
    "\U0001F970": (0.95, 0.01, 0.04),  # 🥰
    "\U0001F60D": (0.95, 0.01, 0.04),  # 😍
    "\U0001F929": (0.96, 0.01, 0.03),  # 🤩
    "\U0001F618": (0.92, 0.01, 0.07),  # 😘
    "\U0001F617": (0.80, 0.03, 0.17),  # 😗
    "\U0001F619": (0.82, 0.03, 0.15),  # 😙
    "\U0001F61A": (0.85, 0.02, 0.13),  # 😚
    "\U0001F60B": (0.88, 0.02, 0.10),  # 😋
    "\U0001F61B": (0.75, 0.05, 0.20),  # 😛
    "\U0001F61C": (0.75, 0.05, 0.20),  # 😜
    "\U0001F92A": (0.72, 0.05, 0.23),  # 🤪
    "\U0001F61D": (0.70, 0.08, 0.22),  # 😝
    "\U0001F911": (0.70, 0.05, 0.25),  # 🤑
    "\U0001F917": (0.90, 0.02, 0.08),  # 🤗
    "\U0001F973": (0.95, 0.01, 0.04),  # 🥳
    "\U0001F60E": (0.85, 0.03, 0.12),  # 😎
    "\U0001F913": (0.65, 0.05, 0.30),  # 🤓
    "\U0001F979": (0.75, 0.10, 0.15),  # 🥹
    # ── Love / affection ──────────────────────────────────────────────────────
    "\u2764":     (0.95, 0.01, 0.04),  # ❤
    "\u2763":     (0.93, 0.01, 0.06),  # ❣️  ← NEW  heavy heart exclamation
    "\U0001F9E1": (0.92, 0.01, 0.07),  # 🧡
    "\U0001F49B": (0.92, 0.01, 0.07),  # 💛
    "\U0001F49A": (0.90, 0.02, 0.08),  # 💚
    "\U0001F499": (0.90, 0.02, 0.08),  # 💙
    "\U0001F49C": (0.90, 0.02, 0.08),  # 💜
    "\U0001F5A4": (0.55, 0.25, 0.20),  # 🖤
    "\U0001F90D": (0.80, 0.05, 0.15),  # 🤍
    "\U0001F90E": (0.75, 0.05, 0.20),  # 🤎
    "\U0001F495": (0.93, 0.01, 0.06),  # 💕
    "\U0001F49E": (0.93, 0.01, 0.06),  # 💞
    "\U0001F493": (0.92, 0.01, 0.07),  # 💓
    "\U0001F497": (0.93, 0.01, 0.06),  # 💗
    "\U0001F496": (0.94, 0.01, 0.05),  # 💖
    "\U0001F498": (0.93, 0.01, 0.06),  # 💘
    "\U0001F49D": (0.92, 0.01, 0.07),  # 💝
    "\U0001F49F": (0.90, 0.02, 0.08),  # 💟
    "\U0001F48C": (0.88, 0.02, 0.10),  # 💌
    "\U0001F48B": (0.88, 0.02, 0.10),  # 💋
    "\U0001F63B": (0.93, 0.01, 0.06),  # 😻
    "\U0001F491": (0.93, 0.01, 0.06),  # 💑  ← NEW
    "\U0001F48F": (0.90, 0.02, 0.08),  # 💏  ← NEW
    # ── Praise / celebration ──────────────────────────────────────────────────
    "\U0001F44D": (0.90, 0.02, 0.08),  # 👍
    "\U0001F44C": (0.88, 0.03, 0.09),  # 👌
    "\U0001F64C": (0.92, 0.01, 0.07),  # 🙌
    "\U0001F44F": (0.90, 0.02, 0.08),  # 👏
    "\U0001F91D": (0.80, 0.03, 0.17),  # 🤝
    "\U0001F4AA": (0.85, 0.03, 0.12),  # 💪
    "\U0001F3C6": (0.90, 0.02, 0.08),  # 🏆
    "\U0001F947": (0.92, 0.01, 0.07),  # 🥇
    "\U0001F389": (0.93, 0.01, 0.06),  # 🎉
    "\U0001F38A": (0.93, 0.01, 0.06),  # 🎊
    "\U0001F388": (0.88, 0.02, 0.10),  # 🎈
    "\U0001F942": (0.88, 0.02, 0.10),  # 🥂
    "\U0001F37E": (0.88, 0.02, 0.10),  # 🍾
    "\u2728":     (0.85, 0.02, 0.13),  # ✨
    "\U0001F31F": (0.90, 0.02, 0.08),  # 🌟
    "\u2B50":     (0.88, 0.02, 0.10),  # ⭐
    "\U0001F308": (0.88, 0.02, 0.10),  # 🌈
    "\U0001F525": (0.80, 0.05, 0.15),  # 🔥
    "\U0001F4AF": (0.92, 0.01, 0.07),  # 💯
    "\u2705":     (0.88, 0.03, 0.09),  # ✅
    "\U0001FAF6": (0.92, 0.01, 0.07),  # 🫶
    "\U0001FAC2": (0.88, 0.02, 0.10),  # 🫂
    "\U0001F64B": (0.82, 0.03, 0.15),  # 🙋  ← NEW
    "\U0001F44B": (0.70, 0.05, 0.25),  # 👋  ← NEW  (greeting, mild positive)
    "\u2764\uFE0F":(0.95, 0.01, 0.04), # ❤️  ← NEW  (red heart + variation selector)
    # ── Mild positive / hopeful ───────────────────────────────────────────────
    "\U0001F60C": (0.75, 0.08, 0.17),  # 😌
    "\U0001F64F": (0.75, 0.05, 0.20),  # 🙏
    "\U0001F338": (0.82, 0.03, 0.15),  # 🌸
    "\U0001F33A": (0.82, 0.03, 0.15),  # 🌺
    "\U0001F33B": (0.85, 0.02, 0.13),  # 🌻
    "\U0001F33C": (0.83, 0.03, 0.14),  # 🌼
    "\U0001F339": (0.85, 0.03, 0.12),  # 🌹
    "\U0001F490": (0.85, 0.02, 0.13),  # 💐
    "\U0001F340": (0.83, 0.03, 0.14),  # 🍀
    "\u2600":     (0.85, 0.02, 0.13),  # ☀
    "\U0001F98B": (0.82, 0.02, 0.16),  # 🦋
    "\U0001F54A": (0.80, 0.03, 0.17),  # 🕊
    "\U0001F63A": (0.82, 0.03, 0.15),  # 😺
    "\U0001F31E": (0.88, 0.02, 0.10),  # 🌞  ← NEW
    # ── Neutral / informational ───────────────────────────────────────────────
    "\U0001F610": (0.15, 0.15, 0.70),  # 😐
    "\U0001F611": (0.10, 0.20, 0.70),  # 😑
    "\U0001F636": (0.10, 0.10, 0.80),  # 😶
    "\U0001F914": (0.15, 0.15, 0.70),  # 🤔
    "\U0001F928": (0.10, 0.30, 0.60),  # 🤨
    "\U0001F9D0": (0.20, 0.15, 0.65),  # 🧐
    "\U0001F60F": (0.35, 0.30, 0.35),  # 😏
    "\U0001FAE0": (0.20, 0.30, 0.50),  # 🫠
    "\U0001F643": (0.40, 0.30, 0.30),  # 🙃
    "\U0001F440": (0.20, 0.20, 0.60),  # 👀
    "\U0001F937": (0.15, 0.15, 0.70),  # 🤷
    "\U0001F4A4": (0.15, 0.20, 0.65),  # 💤
    "\U0001FAE3": (0.30, 0.30, 0.40),  # 🫣
    "\U0001FAE4": (0.15, 0.45, 0.40),  # 🫤
    "\U0001FAE5": (0.10, 0.30, 0.60),  # 🫥
    # ── Negative / sadness ────────────────────────────────────────────────────
    "\U0001F622": (0.05, 0.88, 0.07),  # 😢
    "\U0001F62D": (0.05, 0.90, 0.05),  # 😭
    "\U0001F61E": (0.05, 0.88, 0.07),  # 😞
    "\U0001F61F": (0.05, 0.85, 0.10),  # 😟
    "\U0001F614": (0.08, 0.82, 0.10),  # 😔
    "\U0001F615": (0.10, 0.75, 0.15),  # 😕
    "\U0001F641": (0.08, 0.82, 0.10),  # 🙁
    "\u2639":     (0.05, 0.88, 0.07),  # ☹
    "\U0001F623": (0.05, 0.88, 0.07),  # 😣
    "\U0001F616": (0.05, 0.88, 0.07),  # 😖
    "\U0001F62B": (0.05, 0.88, 0.07),  # 😫
    "\U0001F629": (0.05, 0.88, 0.07),  # 😩
    "\U0001F97A": (0.20, 0.65, 0.15),  # 🥺
    "\U0001F613": (0.10, 0.78, 0.12),  # 😓
    "\U0001F625": (0.10, 0.78, 0.12),  # 😥
    "\U0001F630": (0.05, 0.88, 0.07),  # 😰
    "\U0001F63F": (0.05, 0.88, 0.07),  # 😿
    "\U0001F494": (0.03, 0.92, 0.05),  # 💔
    "\U0001F4A7": (0.10, 0.60, 0.30),  # 💧
    # ── Anger / frustration ───────────────────────────────────────────────────
    "\U0001F620": (0.03, 0.92, 0.05),  # 😠
    "\U0001F621": (0.02, 0.95, 0.03),  # 😡
    "\U0001F92C": (0.02, 0.96, 0.02),  # 🤬
    "\U0001F624": (0.05, 0.88, 0.07),  # 😤
    "\U0001F44E": (0.03, 0.92, 0.05),  # 👎
    "\U0001F926": (0.05, 0.80, 0.15),  # 🤦
    "\U0001F612": (0.05, 0.80, 0.15),  # 😒
    "\U0001F62C": (0.10, 0.65, 0.25),  # 😬
    "\U0001F644": (0.05, 0.75, 0.20),  # 🙄
    # ── Fear / disgust / shock ────────────────────────────────────────────────
    "\U0001F628": (0.05, 0.85, 0.10),  # 😨
    "\U0001F631": (0.05, 0.85, 0.10),  # 😱
    "\U0001F633": (0.10, 0.60, 0.30),  # 😳
    "\U0001F976": (0.08, 0.72, 0.20),  # 🥶
    "\U0001F922": (0.03, 0.92, 0.05),  # 🤢
    "\U0001F92E": (0.02, 0.95, 0.03),  # 🤮
    "\U0001F927": (0.05, 0.80, 0.15),  # 🤧
    "\U0001F912": (0.05, 0.82, 0.13),  # 🤒
    "\U0001F915": (0.05, 0.82, 0.13),  # 🤕
    "\U0001F637": (0.05, 0.75, 0.20),  # 😷
    "\U0001F974": (0.10, 0.65, 0.25),  # 🥴
    "\U0001F635": (0.05, 0.80, 0.15),  # 😵
    "\U0001F480": (0.05, 0.75, 0.20),  # 💀
    "\u2620":     (0.05, 0.75, 0.20),  # ☠
    # ── Sarcasm-adjacent / ambiguous ─────────────────────────────────────────
    "\U0001F608": (0.30, 0.55, 0.15),  # 😈
    "\U0001F47F": (0.10, 0.78, 0.12),  # 👿
    "\U0001F921": (0.40, 0.35, 0.25),  # 🤡
    "\U0001F47B": (0.35, 0.35, 0.30),  # 👻
    "\U0001F4A9": (0.20, 0.60, 0.20),  # 💩
    "\U0001FAB6": (0.65, 0.10, 0.25),  # 🪶
    "\U0001FAE1": (0.65, 0.10, 0.25),  # 🫡
    "\U0001F4A5": (0.55, 0.30, 0.15),  # 💥  ← NEW
}

_EMOJI_FALLBACK: tuple[float, float, float] = (0.33, 0.33, 0.34)


# ══════════════════════════════════════════════════════════════════════════════
#  EMOJI DETECTION HELPERS
# ══════════════════════════════════════════════════════════════════════════════

def _is_emoji_codepoint(cp: int) -> bool:
    ranges = [
        (0x1F600, 0x1F64F), (0x1F300, 0x1F5FF), (0x1F680, 0x1F6FF),
        (0x1F700, 0x1F77F), (0x1F780, 0x1F7FF), (0x1F800, 0x1F8FF),
        (0x1F900, 0x1F9FF), (0x1FA00, 0x1FA6F), (0x1FA70, 0x1FAFF),
        (0x2600,  0x26FF),  (0x2700,  0x27BF),  (0xFE00,  0xFE0F),
        (0x1F1E0, 0x1F1FF), (0x231A,  0x231B),  (0x23E9,  0x23F3),
        (0x23F8,  0x23FA),  (0x25AA,  0x25AB),  (0x25B6,  0x25B6),
        (0x25C0,  0x25C0),  (0x25FB,  0x25FE),  (0x2614,  0x2615),
        (0x2648,  0x2653),  (0x267F,  0x267F),  (0x2693,  0x2693),
        (0x26A1,  0x26A1),  (0x26AA,  0x26AB),  (0x26BD,  0x26BE),
        (0x26C4,  0x26C5),  (0x26D4,  0x26D4),  (0x26EA,  0x26EA),
        (0x26F2,  0x26F3),  (0x26F5,  0x26F5),  (0x26FA,  0x26FA),
        (0x26FD,  0x26FD),  (0x2702,  0x2702),  (0x2705,  0x2705),
        (0x2708,  0x270D),  (0x270F,  0x270F),  (0x2712,  0x2712),
        (0x2714,  0x2714),  (0x2716,  0x2716),  (0x271D,  0x271D),
        (0x2721,  0x2721),  (0x2728,  0x2728),  (0x2733,  0x2734),
        (0x2744,  0x2744),  (0x2747,  0x2747),  (0x274C,  0x274C),
        (0x274E,  0x274E),  (0x2753,  0x2755),  (0x2757,  0x2757),
        (0x2763,  0x2764),  (0x2795,  0x2797),  (0x27A1,  0x27A1),
        (0x27B0,  0x27B0),  (0x27BF,  0x27BF),  (0x2934,  0x2935),
        (0x2B05,  0x2B07),  (0x2B1B,  0x2B1C),  (0x2B50,  0x2B50),
        (0x2B55,  0x2B55),  (0x3030,  0x3030),  (0x303D,  0x303D),
        (0x3297,  0x3297),  (0x3299,  0x3299),  (0x200D,  0x200D),
    ]
    return any(lo <= cp <= hi for lo, hi in ranges)


_TRANSPARENT_CHARS = frozenset(" \t\n\r\u200D\uFE0F\uFE0E\u20E3")


def is_emoji_only(text: str) -> bool:
    """Return True if the text contains ONLY emojis (and whitespace/joiners)."""
    if not text or not text.strip():
        return False
    found_emoji = False
    for char in text:
        if char in _TRANSPARENT_CHARS:
            continue
        cp = ord(char)
        if _is_emoji_codepoint(cp):
            found_emoji = True
            continue
        cat = unicodedata.category(char)
        if cat in ("Mn", "Cf", "So"):
            continue
        return False
    return found_emoji


def extract_emojis(text: str) -> list[str]:
    """Extract all emoji clusters (including ZWJ sequences and skin tones)."""
    emojis: list[str] = []
    chars = list(text)
    i = 0
    while i < len(chars):
        cp = ord(chars[i])
        if _is_emoji_codepoint(cp):
            cluster = chars[i]
            j = i + 1
            while j < len(chars):
                ncp = ord(chars[j])
                if ncp in (0x200D, 0xFE0F, 0xFE0E, 0x20E3) or (0x1F3FB <= ncp <= 0x1F3FF):
                    cluster += chars[j]; j += 1
                elif 0x1F1E0 <= ncp <= 0x1F1FF and j == i + 1:
                    cluster += chars[j]; j += 1
                else:
                    break
            emojis.append(cluster)
            i = j
        else:
            i += 1
    return emojis


def classify_emoji_comment(text: str) -> dict:
    """
    Emoji-only fast path.
    Returns unified {sentiment, confidence, source}.
    """
    emojis = extract_emojis(text)
    if not emojis:
        return {"sentiment": "Neutral", "confidence": 0.0, "source": "emoji_lexicon"}

    total_pos = total_neg = total_neu = 0.0
    for em in emojis:
        weights = (
            EMOJI_SENTIMENT.get(em)
            or EMOJI_SENTIMENT.get(em[0])
            or _EMOJI_FALLBACK
        )
        total_pos += weights[0]
        total_neg += weights[1]
        total_neu += weights[2]

    n   = len(emojis)
    pos = total_pos / n
    neg = total_neg / n
    neu = total_neu / n

    scores    = {"Positive": pos, "Negative": neg, "Neutral": neu}
    sentiment = max(scores, key=scores.__getitem__)
    return {
        "sentiment":  sentiment,
        "confidence": round(scores[sentiment], 4),
        "source":     "emoji_lexicon",
    }


# ══════════════════════════════════════════════════════════════════════════════
#  IMPROVEMENT ①  EMOJI HYBRID BOOST
#  Applied to MIXED comments (text + emojis) after the LLM has spoken.
#  Logic:
#    • LLM=Neutral  + emoji strongly ≠ Neutral  → override with emoji label
#    • LLM=X        + emoji agrees with X        → blend confidence upward
#    • LLM=X        + emoji disagrees            → trust LLM (keeps its result)
# ══════════════════════════════════════════════════════════════════════════════

def emoji_sentiment_score(text: str) -> tuple[float, float, float] | None:
    """
    Returns (pos, neg, neu) averaged over all *known* emojis in the text,
    or None if no known emojis are found.
    Unknown emojis (not in lexicon) are skipped rather than penalised.
    """
    emojis = extract_emojis(text)
    if not emojis:
        return None

    total_pos = total_neg = total_neu = 0.0
    found = 0
    for em in emojis:
        weights = EMOJI_SENTIMENT.get(em) or EMOJI_SENTIMENT.get(em[0]) or None
        if weights:
            total_pos += weights[0]
            total_neg += weights[1]
            total_neu += weights[2]
            found += 1

    if found == 0:
        return None

    return total_pos / found, total_neg / found, total_neu / found


def apply_emoji_boost(comment: str, llm_result: dict) -> dict:
    """
    Post-process an LLM result using emoji signals embedded in the comment.

    Three cases:
      1. LLM said Neutral, emoji is confident about Pos/Neg → override
      2. LLM and emoji agree                               → boost confidence
      3. LLM gave Pos/Neg, emoji disagrees                 → keep LLM result
    """
    scores = emoji_sentiment_score(comment)
    if scores is None:
        return llm_result  # no known emojis present, nothing to do

    pos, neg, neu = scores
    label_scores  = {"Positive": pos, "Negative": neg, "Neutral": neu}
    emoji_label   = max(label_scores, key=label_scores.__getitem__)
    emoji_conf    = label_scores[emoji_label]
    llm_label     = llm_result["sentiment"]

    # ── Case 1: LLM neutral, emoji confident about a polarity ─────────────────
    if (llm_label == "Neutral"
            and emoji_label != "Neutral"
            and emoji_conf >= EMOJI_OVERRIDE_THRESHOLD):
        return {
            "sentiment":  emoji_label,
            "confidence": round(emoji_conf, 4),
            "source":     "hf_llm_batch+emoji_override",
        }

    # ── Case 2: LLM and emoji agree → blend confidence upward ─────────────────
    if llm_label == emoji_label:
        blended = min(1.0, llm_result["confidence"] * 0.5 + emoji_conf * 0.5)
        return {**llm_result, "confidence": round(blended, 4)}

    # ── Case 3: disagreement → stick with LLM ─────────────────────────────────
    return llm_result


# ══════════════════════════════════════════════════════════════════════════════
#  IMPROVEMENT ②  RICHER DARJA-AWARE SYSTEM PROMPT
# ══════════════════════════════════════════════════════════════════════════════

BATCH_SYSTEM_PROMPT = """You are an expert sentiment analysis AI specialising in Tunisian Darja — a code-switching dialect that freely mixes Arabic, French, and English in the same sentence.

── Key Tunisian Darja sentiment signals you MUST recognise ──────────────────

POSITIVE intensifiers / praise:
  wallah (oath used as exclamation of admiration), barcha (a lot / very much),
  bezzef (a lot), 3ajib / 3ajiib (amazing), ma7la / ma7leeha (how beautiful/nice),
  behi (good), mriguel (great), top, bravo, merci, chapeau, waw (wow),
  brasmi / b rasmi (I swear / truly), rabbi y7afdhek (God bless you),
  mabrook (congratulations), 5ater / khater (because / since → positive context),
  sa7 (correct / right → agreement = positive)

NEGATIVE signals:
  khayeb / khaybe (bad), ma3jebnich / ma ye3jebnich (I don't like it),
  service 0 / note 0 / 0/10 (zero rating = very negative), mouche behi (not good),
  nti7a (disaster), dégoûtant / degoûtant (disgusting), nul (worthless),
  barra (away / get out), m9arfes (disgusting/bad), khsara (what a loss)

NEUTRAL / informational:
  wselni (it arrived), yetfahem (understandable), normal, ok, 5ater (because),
  sma3t (I heard), cheft (I saw)

SPECIAL RULES:
  • "ma7la" alone = Positive (means "how beautiful/nice")
  • "wallah" as exclamation of wonder = Positive
  • "service 0" / "note 0" = Negative (zero-star rating)
  • Repeated letters (merciiii, wawwww, 3ajiiiib) amplify the dominant sentiment
  • A mix of Darja + positive emoji (❤️ ❣️ 😍 🥰 💕) = almost always Positive

─────────────────────────────────────────────────────────────────────────────
Return ONLY a valid JSON array — one object per comment, same order, no extra text.
Each object must have exactly two keys:
  "id"        → integer, same as the input number
  "sentiment" → one of: "Positive", "Negative", "Neutral"

Examples:
Input:
0: "brasmi ma7la southa wallah ❣️"
1: "C'est nul, ma ye3jebnich jemla w service 0."
2: "wselni lyoum el behi."
3: "3ajib barcha merci bezzef ❤️"
4: "khayeb mouche behi khidma 0"
5: "ok normal"
6: "wawwww ma7leeeha rabbi y7afdhek"
7: "nti7a dégoûtant m9arfes"

Output:
[{"id":0,"sentiment":"Positive"},{"id":1,"sentiment":"Negative"},{"id":2,"sentiment":"Positive"},{"id":3,"sentiment":"Positive"},{"id":4,"sentiment":"Negative"},{"id":5,"sentiment":"Neutral"},{"id":6,"sentiment":"Positive"},{"id":7,"sentiment":"Negative"}]

Do NOT include markdown fences, explanation, or any text outside the JSON array."""


# ══════════════════════════════════════════════════════════════════════════════
#  BATCH LLM CLASSIFICATION
# ══════════════════════════════════════════════════════════════════════════════

def extract_batch_sentiments(response_text: str, n: int) -> list[str]:
    """
    Parse a JSON array like [{"id":0,"sentiment":"Positive"}, ...]
    from the raw model output.  Falls back to "Neutral" for any missing entry.
    """
    results = ["Neutral"] * n

    try:
        clean = re.sub(r"```(?:json)?", "", response_text).strip()
        match = re.search(r'\[.*\]', clean, re.DOTALL)
        if match:
            data = json.loads(match.group(0))
            if isinstance(data, list):
                for item in data:
                    idx  = item.get("id")
                    sent = str(item.get("sentiment", "Neutral")).capitalize()
                    if isinstance(idx, int) and 0 <= idx < n:
                        if sent in ("Positive", "Negative", "Neutral"):
                            results[idx] = sent
                return results
    except Exception:
        pass

    # Last-resort keyword scan (very rough)
    lower = response_text.lower()
    if "positive" in lower and "negative" not in lower:
        return ["Positive"] * n
    if "negative" in lower and "positive" not in lower:
        return ["Negative"] * n
    return results


def classify_text_llm_batch(comments: list[str], model, tokenizer) -> list[dict]:
    """
    Classify a list of text comments in ONE LLM call.
    Empty comments are handled locally (no tokens wasted).
    Returns [{sentiment, confidence, source}, …] in the same order.
    """
    output: list[dict | None] = [None] * len(comments)

    text_indices:  list[int] = []
    text_comments: list[str] = []

    for i, c in enumerate(comments):
        if not str(c).strip():
            output[i] = {"sentiment": "Neutral", "confidence": 0.0, "source": "empty"}
        else:
            text_indices.append(i)
            text_comments.append(str(c))

    if not text_comments:
        return output  # type: ignore[return-value]

    numbered_block = "\n".join(f'{i}: "{c}"' for i, c in enumerate(text_comments))

    messages = [
        {"role": "system", "content": BATCH_SYSTEM_PROMPT},
        {"role": "user",   "content": f"Comments:\n{numbered_block}"},
    ]

    prompt = tokenizer.apply_chat_template(
        messages, tokenize=False, add_generation_prompt=True
    )
    inputs = tokenizer([prompt], return_tensors="pt").to(model.device)

    with torch.no_grad():
        outputs = model.generate(
            **inputs,
            max_new_tokens=512,
            do_sample=False,
            pad_token_id=tokenizer.eos_token_id,
        )

    generated_ids = outputs[0][inputs.input_ids.shape[1]:]
    response_text = tokenizer.decode(generated_ids, skip_special_tokens=True).strip()

    sentiments = extract_batch_sentiments(response_text, len(text_comments))

    for local_idx, (global_idx, sent) in enumerate(zip(text_indices, sentiments)):
        output[global_idx] = {
            "sentiment":  sent,
            "confidence": 1.0,
            "source":     "hf_llm_batch",
        }

    return output  # type: ignore[return-value]


# ══════════════════════════════════════════════════════════════════════════════
#  CHECKPOINT HELPER
# ══════════════════════════════════════════════════════════════════════════════

def save_checkpoint(
    df: pd.DataFrame,
    sentiments: list,
    confidences: list,
    sources: list,
    output_path: str,
) -> None:
    """Write all rows processed so far to the output CSV."""
    n = len(sentiments)
    out_df = df.iloc[:n].copy()
    out_df["predicted_sentiment"] = sentiments
    out_df["confidence"]          = confidences
    out_df["prediction_source"]   = sources
    out_df.to_csv(output_path, index=False, encoding="utf-8-sig")
    print(f"\n  💾  Checkpoint saved — {n:,} rows → {output_path}")


# ══════════════════════════════════════════════════════════════════════════════
#  TERMINAL COLOUR HELPERS
# ══════════════════════════════════════════════════════════════════════════════

_LABEL_COLOUR = {
    "Positive": "\033[92m",   # green
    "Negative": "\033[91m",   # red
    "Neutral":  "\033[93m",   # yellow
}
_RESET = "\033[0m"

def _coloured(label: str) -> str:
    return f"{_LABEL_COLOUR.get(label, '')}{label}{_RESET}"


# ══════════════════════════════════════════════════════════════════════════════
#  MAIN PIPELINE
# ══════════════════════════════════════════════════════════════════════════════
class Args:
        input = DEFAULT_INPUT
        output = DEFAULT_OUTPUT
        column = DEFAULT_COLUMN
def main():
    # parser = argparse.ArgumentParser(
    #     description="Batch Sentiment Analysis via HuggingFace BitsAndBytes"
    # )
    # parser.add_argument("--input",  default=DEFAULT_INPUT,  help="Path to input CSV")
    # parser.add_argument("--output", default=DEFAULT_OUTPUT, help="Path to output CSV")
    # parser.add_argument("--column", default=DEFAULT_COLUMN, help="Name of the text column")
    args =Args()

    # ── Load data ─────────────────────────────────────────────────────────────
    print(f"Loading data from {args.input}...")
    try:
        df = pd.read_csv(args.input)
    except Exception as e:
        sys.exit(f"ERROR: Could not read CSV: {e}")

    total_rows = len(df)
    print(f"Total rows: {total_rows:,}")

    # ── Resume logic ──────────────────────────────────────────────────────────
    results_sentiment:  list = []
    results_confidence: list = []
    results_source:     list = []
    already_done = 0

    if os.path.exists(args.output):
        try:
            existing = pd.read_csv(args.output, encoding="utf-8-sig")
            if "predicted_sentiment" in existing.columns and len(existing) > 0:
                already_done       = len(existing)
                results_sentiment  = existing["predicted_sentiment"].tolist()
                results_confidence = existing["confidence"].tolist()
                results_source     = existing["prediction_source"].tolist()
                print(f"  ↩  Resuming from row {already_done:,} "
                      f"({total_rows - already_done:,} remaining)")
        except Exception as e:
            print(f"  ⚠  Could not read existing output ({e}), starting fresh.")

    # ── Load model ────────────────────────────────────────────────────────────
    # quantization_config = BitsAndBytesConfig(
    #     load_in_4bit=True,
    #     bnb_4bit_compute_dtype=torch.float16,
    #     bnb_4bit_quant_type="nf4",
    #     bnb_4bit_use_double_quant=True,
    # )
    quantization_config = BitsAndBytesConfig(
    load_in_8bit=True
)
    print(f"\nLoading model {MODEL_ID} into GPU (4-bit)...")
    tokenizer = AutoTokenizer.from_pretrained(MODEL_ID)
    model     = AutoModelForCausalLM.from_pretrained(
        MODEL_ID,
        device_map="auto",
         # torch_dtype=torch.float16,  
         quantization_config=quantization_config,
    )
    print("Model loaded.\n")

    # ── Slice to unprocessed comments ─────────────────────────────────────────
    all_comments        = df[args.column].tolist()
    comments_to_process = all_comments[already_done:]

    if not comments_to_process:
        print("Nothing left to process. Output is already complete.")
        return

    # ── Main loop ─────────────────────────────────────────────────────────────
    num_batches = (len(comments_to_process) + BATCH_SIZE - 1) // BATCH_SIZE
    pbar = tqdm(total=len(comments_to_process), desc="Comments", unit="cmt")

    for batch_idx in range(num_batches):
        batch_start  = batch_idx * BATCH_SIZE
        batch        = comments_to_process[batch_start : batch_start + BATCH_SIZE]
        batch_output: list[dict | None] = [None] * len(batch)

        # ── Within each batch:
        #    emoji-only  → lexicon (fast, no GPU)
        #    mixed/text  → LLM, then emoji boost  ← NEW
        # ─────────────────────────────────────────────────────────────────────
        llm_indices:  list[int] = []
        llm_comments: list[str] = []

        for i, comment in enumerate(batch):
            comment_str = str(comment)
            if is_emoji_only(comment_str):
                batch_output[i] = classify_emoji_comment(comment_str)
            else:
                llm_indices.append(i)
                llm_comments.append(comment_str)

        # Single LLM call for all text/mixed comments in this batch
        if llm_comments:
            llm_results = classify_text_llm_batch(llm_comments, model, tokenizer)
            for local_i, global_i in enumerate(llm_indices):
                raw_result = llm_results[local_i]
                # ── IMPROVEMENT ①: apply emoji boost to every LLM result ──────
                batch_output[global_i] = apply_emoji_boost(
                    llm_comments[local_i], raw_result
                )

        # ── Collect results + live terminal print ─────────────────────────────
        for i, (comment, res) in enumerate(zip(batch, batch_output)):
            global_row = already_done + batch_start + i + 1   # 1-based display

            results_sentiment.append(res["sentiment"])
            results_confidence.append(res["confidence"])
            results_source.append(res["source"])

            preview = str(comment).replace("\n", " ")[:70]
            if len(str(comment)) > 70:
                preview += "…"

            print(
                f"  [{global_row:>6}/{total_rows}] "
                f"{_coloured(res['sentiment']):<20} "
                f"src={res['source']:<28} "
                f"| {preview}"
            )

        pbar.update(len(batch))

        # ── Checkpoint save every SAVE_EVERY comments ─────────────────────────
        processed_so_far = already_done + batch_start + len(batch)
        prev_checkpoint  = (processed_so_far - len(batch)) // SAVE_EVERY
        curr_checkpoint  = processed_so_far // SAVE_EVERY
        is_last_batch    = (batch_idx == num_batches - 1)

        if curr_checkpoint > prev_checkpoint or is_last_batch:
            save_checkpoint(
                df,
                results_sentiment,
                results_confidence,
                results_source,
                args.output,
            )

    pbar.close()

    # ── Final summary ─────────────────────────────────────────────────────────
    from collections import Counter
    counts = Counter(results_sentiment)
    total  = len(results_sentiment)
    print(f"\n{'═'*60}")
    print(f"✅  Done!  {total:,} rows saved to {args.output}")
    print(f"   Positive : {counts['Positive']:>7,}  ({counts['Positive']/total*100:.1f}%)")
    print(f"   Negative : {counts['Negative']:>7,}  ({counts['Negative']/total*100:.1f}%)")
    print(f"   Neutral  : {counts['Neutral']:>7,}  ({counts['Neutral']/total*100:.1f}%)")
    print(f"{'═'*60}")

    # ── Source breakdown (shows how often emoji boost fired) ──────────────────
    source_counts = Counter(results_source)
    print("\n  Prediction source breakdown:")
    for src, cnt in source_counts.most_common():
        print(f"    {src:<35} {cnt:>7,}  ({cnt/total*100:.1f}%)")
    print(f"{'═'*60}\n")


main()