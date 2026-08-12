"""Provider-neutral system prompt templates.

This module stores and retrieves system prompt templates.  It is
completely independent of any LLM provider, model, or UI framework.
All prompts are plain text — no provider-specific instructions,
no model names, no SDK references.

Future expansion (not yet implemented)
--------------------------------------
- ``prompt("rag")``         -> RAG-specific system prompt
- ``prompt("agent")``       -> Agent workflow instructions
- ``prompt("creative")``    -> Creative writing persona
- ``prompt(name, **params)``-> Dynamic prompt formatting
"""

from __future__ import annotations

from typing import Final

# ---------------------------------------------------------------------------
# Default prompt
# ---------------------------------------------------------------------------
_DEFAULT_SYSTEM_PROMPT: Final[str] = (
    "You are a helpful, respectful, and honest assistant. "
    "Answer concisely, completely and naturally, as if you already know the "
    "information. Never invent details; if unsure, say so.\n\n"
    "When context is provided (uploaded documents):\n"
    "- Answer only from the retrieved context; do not guess missing details.\n"
    "- If not in context, clearly say it's unavailable.\n"
    "- Never name documents, files, pages, or sources, or use phrases like "
    "'According to the document' unless asked.\n\n"
    "Formatting:\n"
    "- Use Arabic numbering (1., 2., 3.) when presenting ordered lists in the "
    "answer, including lists based on retrieved documents.\n"
    "- Put each list item on its own line. Never place multiple list items on "
    "the same line.\n"
    "  Good:\n"
    "  1. Objective one.\n"
    "  2. Objective two.\n"
    "  3. Objective three.\n"
    "  Bad (never do this):\n"
    "  I. Objective one. II. Objective two. III. Objective three.\n"
    "- Preserve Roman numerals only when they are part of document structures "
    "such as chapters, sections, versions, phases, headings, or quoted text.\n"
    "- Do not preserve Roman numeral list markers from documents when "
    "presenting them as an answer list.\n\n"
    "Based on question type:\n"
    "- Simple facts: give a direct, short answer.\n"
    "- Explanations: use short paragraphs or bullets.\n"
    "- Programming: use markdown code blocks with language tags.\n\n"
    "Avoid disclaimers, repeated explanations, and filler text."
)
# ---------------------------------------------------------------------------
# Prompt registry (extensible)
# ---------------------------------------------------------------------------
_PROMPT_REGISTRY: Final[dict[str, str]] = {
    "default": _DEFAULT_SYSTEM_PROMPT,
}


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------
def get_default_prompt() -> str:
    """Return the default system prompt.

    Returns
    -------
    str
        The default chatbot system prompt.
    """
    return _DEFAULT_SYSTEM_PROMPT


def get_prompt(name: str) -> str:
    """Retrieve a named prompt from the registry.

    Parameters
    ----------
    name:
        The prompt identifier.  ``"default"`` always returns the
        default system prompt.

    Returns
    -------
    str
        The prompt text.

    Raises
    ------
    KeyError
        If *name* is not a registered prompt.
    """
    return _PROMPT_REGISTRY[name]


class PromptManager:
    """Manages access to system prompt templates.

    Usage
    -----
        manager = PromptManager()
        prompt = manager.get("default")
    """

    @staticmethod
    def get(name: str = "default") -> str:
        """Return the prompt identified by *name*.

        Parameters
        ----------
        name:
            Prompt identifier.  Defaults to ``"default"``.

        Returns
        -------
        str
            The prompt text.

        Raises
        ------
        KeyError
            If the prompt is not registered.
        """
        return get_prompt(name)

    @staticmethod
    def default() -> str:
        """Shorthand for ``get("default")``."""
        return get_default_prompt()
