"""Provider-neutral answer style formatter.

This module implements a small, deterministic post-processing step that
removes retrieval-style attribution wording from generated answers
(e.g. "According to the document...", "Based on the provided context...",
"The document states that...").  It strips only recognised attribution
phrases and preserves the rest of the answer exactly -- including
formatting, markdown, code blocks, and any citations.

The formatter is intentionally conservative:
- It modifies nothing unless a known attribution phrase is present.
- It only strips leading attribution phrases; it never summarises,
  rewrites, or changes the factual content of an answer.
- It is fully provider-neutral (no provider, model, or SDK references).

In addition to attribution removal, the formatter applies a **safe,
code-fence-aware whitespace normalisation** (``normalize_whitespace``):
trailing whitespace on non-code lines is trimmed and runs of excessive
blank lines are collapsed to a single blank line.  Whitespace inside
fenced code blocks is preserved verbatim, and markdown lists, tables,
headings and other structural elements are never altered.  Answers with
no whitespace issues and no attribution phrase are returned unchanged.

Finally, a **conservative markdown consistency pass** (``normalize_markdown``)
is applied:
- An unclosed triple-backtick code fence is closed with a matching fence.
- Obvious broken numbered-list line breaks are restored (a list marker such
  as ``2.`` that directly abuts a preceding word with no space is moved to its
  own line).
Both fixes only touch formatting, never content, and are skipped whenever the
text is ambiguous (false negatives are preferred over false positives).
"""

from __future__ import annotations

import re
from re import Pattern
from typing import Final

#: Leading prepositional attribution phrases.  Each is matched at the
#: start of the answer and removed together with any trailing
#: comma/whitespace.  These are written as general *classes* of source
#: references (document/CV/proposal names, page numbers, team names,
#: "as mentioned in ...") rather than one-off phrasings, so that most
#: retrieval-oriented preambles are normalised without listing every
#: possible wording.
_PREPOSITIONAL_PATTERNS: Final[tuple[Pattern[str], ...]] = (
    # "According to the <source>," -- covers document, SRS, CV, proposal,
    # provided/retrieved context, etc. (multi-word source names allowed).
    re.compile(
        r"According to (?:the )?[a-z0-9_]+(?: [a-z0-9_]+)*[,:]?\s*",
        re.IGNORECASE,
    ),
    # "Based on the <source>," -- covers provided/retrieved/uploaded context.
    re.compile(
        r"Based on (?:the )?(?:provided |retrieved |uploaded )?"
        r"[a-z0-9_]+(?: [a-z0-9_]+)*[,:]?\s*",
        re.IGNORECASE,
    ),
    # "By team <name>," attribution.
    re.compile(r"By team [a-z0-9_]+[,:]?\s*", re.IGNORECASE),
    # "On page <n>," page references (with or without a separating space).
    re.compile(r"On page\s?\d+[,:]?\s*", re.IGNORECASE),
    # "As mentioned in the <source>," attribution.
    re.compile(
        r"As mentioned in (?:the )?[a-z0-9_]+(?: [a-z0-9_]+)*[,:]?\s*",
        re.IGNORECASE,
    ),
)

#: Leading "X states that" / "X mentions that" attribution phrases.  The
#: continuation of the sentence is re-capitalised to keep the answer
#: grammatically correct.  A single general pattern covers both singular
#: and plural subjects and both "state(s)" and "mention(s)".
_STATES_THAT_PATTERNS: Final[tuple[Pattern[str], ...]] = (
    re.compile(
        r"The (?:srs |provided )?[a-z0-9_]+s? (?:states?|mentions?) that[:]?\s*",
        re.IGNORECASE,
    ),
)

#: Full leading reference sentences pointing at a section, e.g.
#: "This is mentioned in the System Architecture Diagram section, ...".
#: The referencing preamble is stripped; any content that follows is
#: preserved and re-capitalised.
_REFERENCE_SENTENCE_PATTERNS: Final[tuple[Pattern[str], ...]] = (
    re.compile(
        r"This (?:is|was) mentioned in (?:the )?(?:[a-z0-9_]+ )*"
        r"[a-z0-9_]+(?: section)?[,:]?\s*",
        re.IGNORECASE,
    ),
)


def strip_attribution(text: str) -> str:
    """Remove leading retrieval-style attribution phrases from *text*.

    Only leading attribution patterns are removed.  The remainder of the
    answer -- including formatting, markdown, code blocks, and citations --
    is preserved verbatim.  Answers without a recognised attribution
    phrase are returned unchanged.

    Parameters
    ----------
    text:
        The assistant answer text to clean.

    Returns
    -------
    str
        The cleaned answer text.
    """
    if not text:
        return text

    stripped = text.lstrip()

    # Prepositional attribution ("According to the ...", "Based on the ...").
    # The remainder is re-capitalised to keep the answer grammatically correct.
    for pattern in _PREPOSITIONAL_PATTERNS:
        match = pattern.match(stripped)
        if match:
            remainder = stripped[match.end():].lstrip()
            return normalize_markdown(normalize_whitespace(_capitalize(remainder)))

    # "X states that ..." attribution -- capitalise the continuation.
    for pattern in _STATES_THAT_PATTERNS:
        match = pattern.match(stripped)
        if match:
            remainder = stripped[match.end():].lstrip()
            return normalize_markdown(normalize_whitespace(_capitalize(remainder)))

    # Full reference sentences ("This is mentioned in the <section>, ...").
    for pattern in _REFERENCE_SENTENCE_PATTERNS:
        match = pattern.match(stripped)
        if match:
            remainder = stripped[match.end():].lstrip()
            return normalize_markdown(normalize_whitespace(_capitalize(remainder)))

    # No attribution removed: normalise whitespace and markdown only (never
    # changes content meaning).  If nothing needs cleaning, the original text
    # is returned so answers without issues remain byte-for-byte identical.
    return normalize_markdown(normalize_whitespace(text))


#: Markdown characters that indicate a structural/formatting element when
#: they appear at the start of a line.  When the remainder after stripping
#: an attribution begins with one of these, we must not "capitalise" it
#: (that would corrupt the markdown).  Note ``1.`` is matched separately
#: because it is a two-char sequence.
_MARKDOWN_LEAD_CHARS: Final[str] = "-*#>+`|~_=["

#: Regex matching a leading ordered-list marker (e.g. ``1.``, ``12.``).
_ORDERED_LIST_RE: Final[Pattern[str]] = re.compile(r"^\d+\.[ \t]*")


def normalize_whitespace(text: str) -> str:
    """Safely normalise whitespace in *text* without changing meaning.

    Two conservative cleanups are applied:
    - Trailing whitespace is removed from each non-code line.
    - Runs of more than one blank line are collapsed to a single blank line.

    Whitespace inside fenced code blocks (````` ``` ````) is preserved
    verbatim, and markdown lists, tables, headings and other structural
    elements are never restructured.  Only whitespace is affected; the
    content, order and meaning of every line are unchanged.

    Parameters
    ----------
    text:
        The answer text to normalise.

    Returns
    -------
    str
        The normalised text.
    """
    if not text:
        return text

    # Whitespace-only input is returned unchanged (it has no content to
    # normalise, and trimming it would alter the answer).
    if not text.strip():
        return text

    lines = text.split("\n")
    out: list[str] = []
    in_code: bool = False
    blank_run: int = 0

    for line in lines:
        stripped = line.strip()

        # Toggle fenced code-block state on a triple-backtick fence line.
        if stripped.startswith("```"):
            in_code = not in_code
            # Fence lines are kept verbatim (so its indentation is kept).
            out.append(line)
            blank_run = 0
            continue

        if in_code:
            # Preserve code lines exactly (including internal blank lines
            # and indentation).
            out.append(line)
            blank_run = 0
            continue

        # Non-code line.
        if stripped == "":
            blank_run += 1
            if blank_run <= 1:
                out.append("")
            continue

        # Non-blank line: trim trailing whitespace only.
        out.append(line.rstrip())
        blank_run = 0

    # Remove any trailing blank lines at the very end of the answer.
    while out and out[-1] == "":
        out.pop()

    return "\n".join(out)


#: Matches a numbered-list marker (``2.``, ``12.``) that directly abuts a
#: preceding word character with **no space in between** -- a strong signal
#: that a line break was lost between two numbered list items (e.g.
#: "Feature one2. Feature two").  The lookbehind requires the character
#: immediately before the number to be an alphabetic letter, so normal text
#: like "version 2.0", "in 2024.", or "Section 2. Introduction" (where a
#: space or digit precedes the number) is never treated as a list break.
_BROKEN_LIST_RE: Final[Pattern[str]] = re.compile(r"(?<=[A-Za-z])(\d+\.\s)")


#: Validates a single uppercase Roman numeral in the range 1..3999.  This is
#: intentionally strict (not a loose ``[IVXLCDM]+`` run) so that ordinary text
#: made of those letters (e.g. "VI. Let's", "IL.") is never mistaken for a
#: list marker.
_ROMAN_NUMERAL_RE: Final[Pattern[str]] = re.compile(
    r"^(?=[MDCLXVI])M{0,3}(CM|CD|D?C{0,3})(XC|XL|L?X{0,3})(IX|IV|V?I{0,3})$"
)

#: Locates every candidate Roman-list marker (``I.``, ``IV.``, ``XII.`` ...)
#: anywhere on a line.  The leading ``\b`` prevents matching a numeral that is
#: part of a longer word (e.g. the ``I`` in ``VII`` or ``DIV``).
_ROMAN_LIST_MARKER_RE: Final[Pattern[str]] = re.compile(r"\b([IVXLCDM]+)\.")


def _is_roman_numeral(candidate: str) -> bool:
    """Return True if *candidate* is a valid uppercase Roman numeral (1..3999)."""
    return _ROMAN_NUMERAL_RE.match(candidate) is not None


def _fix_broken_roman_lists(line: str) -> str:
    """Restore obvious lost line breaks before inline Roman-list markers.

    A line is treated as a malformed inline Roman list **only** when all three
    conditions hold:
    - it contains at least two valid Roman-numeral markers, **and**
    - at least one marker directly follows a period+whitespace (``. ``), which
      is the strong signal that a previous list item ended mid-line, **and**
    - the markers are not inside a fenced code block (handled by the caller).

    A line break is then inserted before every marker that follows a
    period+whitespace.  This is conservative:
    - ``Chapter I.``, ``Version II.``, ``Section III.`` are untouched because
      their markers follow a word, not a period.
    - A single marker, or markers that do not follow a period, are left alone.
    - Roman numerals are never converted to Arabic numbers.
    """
    if not line:
        return line

    markers = [
        m for m in _ROMAN_LIST_MARKER_RE.finditer(line)
        if _is_roman_numeral(m.group(1))
    ]
    # A list needs at least two markers, one of which ends the previous item.
    if len(markers) < 2:
        return line
    if not any(line[m.start() - 2:m.start()] == ". " for m in markers):
        return line

    # Insert a line break before each marker that follows a period+whitespace.
    # Iterate right-to-left so earlier positions are not shifted.
    result = line
    for m in reversed(markers):
        start = m.start()
        if result[start - 2:start] == ". ":
            # Drop the space and put a newline in its place so no trailing
            # space is left before the line break.
            result = result[:start - 1] + "\n" + result[start:]
    return result


def _capitalize(value: str) -> str:
    """Return *value* with its first alphabetic character uppercased.

    Structural markdown at the start of the answer (bullets, numbered
    lists, headings, blockquotes, inline code, horizontal rules, code
    fences, etc.) is preserved as-is; only the first actual alphabetic
    character in a plain leading run is uppercased.  If the value starts
    with a markdown structure symbol, no case change is applied so the
    formatting is never corrupted.
    """
    if not value:
        return value

    # Preserve leading structural markdown (e.g. "- ", "1. ", "# ", "> ",
    # "```", "`x`", "|", "---", "**", "__").  Do not touch the case.
    if value[0] in _MARKDOWN_LEAD_CHARS:
        return value

    # Ordered-list marker like "1. " or "12. ".
    if _ORDERED_LIST_RE.match(value):
        return value

    # Find the first alphabetic character and capitalise only it, leaving
    # any leading punctuation/quotes/spaces (e.g. '"hello' -> '"Hello')
    # untouched.
    for idx, ch in enumerate(value):
        if ch.isalpha():
            return value[:idx] + ch.upper() + value[idx + 1:]

    # No alphabetic character found (symbols, punctuation only) -- leave
    # the value unchanged.
    return value


def _close_unclosed_fence(text: str) -> str:
    """Close a single unclosed triple-backtick code fence, if present.

    A fenced code block is unclosed when the number of triple-backtick
    fence lines is odd.  In that case a closing `` ``` `` is appended so the
    following content is not swallowed into the code block by the renderer.
    If the fences are balanced (even count) the text is returned unchanged.
    Code content and language tags are never modified.

    Parameters
    ----------
    text:
        The answer text to check.

    Returns
    -------
    str
        The text with any unclosed code fence closed.
    """
    if not text:
        return text

    fence_count = 0
    for line in text.split("\n"):
        if line.strip().startswith("```"):
            fence_count += 1

    if fence_count % 2 == 1:
        return text.rstrip() + "\n```"

    return text


def _fix_numbered_lists(text: str) -> str:
    """Restore obvious lost line breaks before numbered-list markers.

    Only applies to non-code lines.  A numbered marker (``2.`` etc.) that
    directly abuts a preceding word with no space (e.g. "one2. Feature two")
    is moved to its own line.  Content is never rewritten; only a single line
    break is inserted, and only when the marker is clearly a list boundary.
    """
    if not text:
        return text

    lines = text.split("\n")
    out: list[str] = []
    in_code: bool = False

    for line in lines:
        stripped = line.strip()
        if stripped.startswith("```"):
            in_code = not in_code
            out.append(line)
            continue
        if in_code:
            # Never touch code content.
            out.append(line)
            continue
        # Non-code line: fix Arabic broken markers first, then conservatively
        # restore lost line breaks before any malformed inline Roman markers.
        result = _BROKEN_LIST_RE.sub(lambda m: "\n" + m.group(1), line)
        out.append(_fix_broken_roman_lists(result))

    return "\n".join(out)


def normalize_markdown(text: str) -> str:
    """Conservatively normalise obvious markdown inconsistencies.

    Two formatting-only fixes are applied, in order:
    1. An unclosed triple-backtick code fence is closed (``_close_unclosed_fence``).
    2. Obvious broken numbered-list line breaks are restored
       (``_fix_numbered_lists``).

    Both fixes never modify content, never rewrite sentences, never change
    code or language tags, and are skipped whenever the text is ambiguous
    (false negatives are preferred over false positives).

    Parameters
    ----------
    text:
        The answer text to normalise.

    Returns
    -------
    str
        The markdown-normalised text.
    """
    if not text:
        return text
    return _fix_numbered_lists(_close_unclosed_fence(text))
