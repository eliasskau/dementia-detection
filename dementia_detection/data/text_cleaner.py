"""
text_cleaner.py
---------------
Cleans a raw *PAR utterance extracted from a CHAT (.cha) file, leaving
only the spoken words of the participant.

Strips:
  - Timestamps               e.g.  3754_5640
  - CHAT retracing codes     e.g.  [/]  [//]  [///]
  - CHAT event codes         e.g.  [+ gram]  [* m:0]  [% comment]
  - Angle-bracket scopes     e.g.  <a child is trying to get> [//]
  - Filled-pause markers     e.g.  &-uh  &-um  &=laughs
  - Unintelligible tokens    e.g.  xxx  yyy  www
  - Pause notation           e.g.  (.)  (..)  (...)
  - Trailing word completion e.g.  tryin(g)  → trying
  - Replacement annotations  e.g.  outta [: out of]  → out of
  - CHAT unicode markers     e.g.  ↑ ↓ ≠ ∬ ⁰ • ‡ °
  - The *PAR: prefix itself
"""

import re


def clean_participant_text(raw: str) -> str:
    """
    Clean a single PAR utterance string and return plain spoken text.

    Parameters
    ----------
    raw : str
        A raw *PAR line (with or without the '*PAR:' prefix) from a .cha file.

    Returns
    -------
    str
        The cleaned, plain-text utterance.  Empty string if nothing remains.
    """
    text = raw

    # 1. Strip the *PAR: prefix if present
    text = re.sub(r'^\*PAR:\s*', '', text)

    # 2. Expand replacement annotations: word [: replacement] → replacement
    #    e.g.  outta [: out of]  →  out of
    text = re.sub(r'\S+\s*\[:\s*([^\]]+)\]', r'\1', text)

    # 3. Remove angle-bracket retrace spans + their event code
    #    e.g.  <a child is trying> [//]  →  (empty)
    text = re.sub(r'<[^>]*>\s*\[[^\]]*\]', '', text)

    # 4. Remove all remaining square-bracket annotations
    #    e.g.  [/]  [//]  [+ gram]  [* m:0]  [% aside]
    text = re.sub(r'\[[^\]]*\]', '', text)

    # 5. Remove any leftover angle brackets
    text = re.sub(r'[<>]', '', text)

    # 6. Expand word-completion parens: tryin(g) → trying
    text = re.sub(r'\((\w+)\)', r'\1', text)

    # 7. Remove CHAT overlap / continuation / quotation markers
    #    Order matters: longest patterns first.
    #
    #  Quotation delimiters  +"/.  +"  +".   (keep the words, strip the marker)
    text = re.sub(r'\+"\s*/?\.?', '', text)
    #  Self-interruption / trailing-off  +//.  +//?  +/.  +/?
    text = re.sub(r'\+//[.?]?', '', text)
    text = re.sub(r'\+/[.?]', '', text)
    #  Overlap  +<   lazy continuation  +,   trailing  +...  +..
    text = re.sub(r'^\+[<,]?\s*', '', text)          # +< or +, at start
    text = re.sub(r'\+\.\.\.?', '', text)             # +... or +..
    #  Non-verbal &+word
    text = re.sub(r'&\+\S+', '', text)
    # Stray trailing % (orphaned annotation tier marker)
    text = re.sub(r'\s*%\s*$', '', text)

    # 8. Remove filled-pause / non-verbal markers:  &-uh  &-um  &=laughs
    text = re.sub(r'&[-=]\S+', '', text)

    # 8. Remove unintelligible tokens
    text = re.sub(r'\b(xxx|yyy|www)\b', '', text, flags=re.IGNORECASE)

    # 9. Remove CHAT pause notation  (.)  (..)  (...)
    text = re.sub(r'\(\.+\)', '', text)

    # 10. Remove timestamps  e.g.  3754_5640
    text = re.sub(r'\b\d+_\d+\b', '', text)

    # 10b. Remove ASCII control characters (e.g. 0x15 NAK used as field sep)
    text = re.sub(r'[\x00-\x1f\x7f]', '', text)

    # 11. Remove CHAT prosody / disfluency unicode symbols
    text = re.sub(r'[↑↓≠∬⁰•‡°]', '', text)

    # 12. Remove stray punctuation left by removed spans (keep . ? !)
    text = re.sub(r'\s+([.,?!])', r'\1', text)   # close gaps before punct
    text = re.sub(r'^[.,?!\s]+', '', text)        # strip leading punct

    # 13. Collapse whitespace
    text = re.sub(r'[ \t]+', ' ', text).strip()

    return text
