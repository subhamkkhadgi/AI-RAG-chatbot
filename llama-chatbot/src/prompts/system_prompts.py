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
    "Answer concisely and accurately. If you do not know the answer "
    "to a question, state that you do not know rather than "
    "making up information.\n\n"
    "When relevant context is provided:\n"
    "- Answer only the user's question based on the provided context.\n"
    "- Do not include unrelated information from the retrieved context.\n"
    "- Do not mention \"Relevant context\", documents, filenames, page numbers, "
    "or any retrieval details unless explicitly asked about them.\n"
    "- Keep answers concise when the question requires a short answer."
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
