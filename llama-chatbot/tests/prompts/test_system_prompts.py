"""Tests for the provider-neutral prompt management layer.

All tests are pure unit tests — no providers, no API keys,
no network access.
"""

from __future__ import annotations

import pytest

from src.prompts.system_prompts import (
    PromptManager,
    get_default_prompt,
    get_prompt,
)


class TestDefaultPrompt:
    """Verify the default system prompt exists and is well-formed."""

    def test_default_prompt_exists(self) -> None:
        """The default prompt must be a non-empty string."""
        prompt = get_default_prompt()
        assert isinstance(prompt, str)
        assert len(prompt) > 0

    def test_default_prompt_content_is_helpful(self) -> None:
        """The default prompt should contain expected behavioural
        instructions."""
        prompt = get_default_prompt()
        assert "helpful" in prompt.lower()
        assert "respectful" in prompt.lower()
        assert "honest" in prompt.lower()

    def test_default_prompt_no_provider_mentions(self) -> None:
        """The default prompt must not mention any LLM provider."""
        prompt = get_default_prompt()
        assert "groq" not in prompt.lower()
        assert "ollama" not in prompt.lower()
        assert "openai" not in prompt.lower()
        assert "llama" not in prompt.lower()

    def test_default_prompt_no_model_names(self) -> None:
        """The default prompt must not hardcode any model name."""
        prompt = get_default_prompt()
        for forbidden in ("3.1", "3.2", "gpt-", "claude", "gemini"):
            assert forbidden not in prompt

    def test_default_prompt_is_concise(self) -> None:
        """The default prompt should be reasonably short (< 400 chars)."""
        prompt = get_default_prompt()
        assert len(prompt) < 400


class TestPromptRetrieval:
    """Verify prompt lookup behaves correctly."""

    def test_retrieve_default_by_name(self) -> None:
        """get_prompt('default') must return the same text as
        get_default_prompt()."""
        assert get_prompt("default") == get_default_prompt()

    def test_retrieve_default_via_manager(self) -> None:
        """PromptManager.get('default') must return the default
        prompt."""
        assert PromptManager.get("default") == get_default_prompt()

    def test_manager_default_shorthand(self) -> None:
        """PromptManager.default() must return the default prompt."""
        assert PromptManager.default() == get_default_prompt()

    def test_manager_get_with_default_param(self) -> None:
        """PromptManager.get() (no args) must return the default
        prompt."""
        assert PromptManager.get() == get_default_prompt()


class TestUnknownPrompt:
    """Verify behaviour when requesting an unregistered prompt."""

    def test_unknown_prompt_raises_key_error(self) -> None:
        """Requesting a non-existent prompt name must raise
        KeyError."""
        with pytest.raises(KeyError):
            get_prompt("non_existent_prompt")

    def test_unknown_prompt_via_manager_raises_key_error(self) -> None:
        """PromptManager.get with an unknown name must raise
        KeyError."""
        with pytest.raises(KeyError):
            PromptManager.get("non_existent_prompt")

    def test_empty_string_raises_key_error(self) -> None:
        """An empty prompt name must raise KeyError (not silently
        return something)."""
        with pytest.raises(KeyError):
            get_prompt("")


class TestNoSideEffects:
    """Verify the prompt layer does not modify external state."""

    def test_get_default_prompt_is_idempotent(self) -> None:
        """Calling get_default_prompt() multiple times must return the
        same value."""
        a = get_default_prompt()
        b = get_default_prompt()
        c = get_default_prompt()
        assert a == b == c

    def test_prompt_registry_not_modified_by_getter(self) -> None:
        """Retrieving a prompt must not affect the registry."""
        before = get_prompt("default")
        get_prompt("default")
        get_prompt("default")
        after = get_prompt("default")
        assert before == after


class TestNoDependencies:
    """Verify no unwanted imports exist in the prompt layer."""

    def test_no_provider_imports(self) -> None:
        """PromptManager should not import any provider code."""
        import src.prompts.system_prompts as mod

        source = mod.__file__
        if source:
            with open(source) as f:
                content = f.read()
            assert "import " in content  # it should import *something*
            assert "from src.providers" not in content
            assert "import src.providers" not in content

    def test_no_streamlit_imports(self) -> None:
        """PromptManager should not import Streamlit."""
        import src.prompts.system_prompts as mod

        source = mod.__file__
        if source:
            with open(source) as f:
                content = f.read()
            assert "streamlit" not in content.lower()

    def test_no_environment_loading(self) -> None:
        """PromptManager should not load environment variables."""
        import src.prompts.system_prompts as mod

        source = mod.__file__
        if source:
            with open(source) as f:
                content = f.read()
            assert "load_dotenv" not in content
            assert ".env" not in content
            assert "environ" not in content
