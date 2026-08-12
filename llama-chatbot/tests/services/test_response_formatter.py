"""Unit tests for the provider-neutral answer style formatter.

These are pure unit tests — no providers, no API keys, no network.
"""

from __future__ import annotations

import pytest

from src.services.response_formatter import strip_attribution


class TestAttributionRemoval:
    """Verify recognised attribution phrases are removed."""

    def test_according_to_the_document(self) -> None:
        """'According to the document,' should be removed."""
        assert (
            strip_attribution(
                "According to the document, the purpose of TES is to automate "
                "teacher evaluation."
            )
            == "The purpose of TES is to automate teacher evaluation."
        )

    def test_according_to_the_srs_document(self) -> None:
        """'According to the SRS document,' should be removed."""
        assert (
            strip_attribution(
                "According to the SRS document, the system must log all events."
            )
            == "The system must log all events."
        )

    def test_according_to_the_srs(self) -> None:
        """'According to the SRS,' should be removed."""
        assert (
            strip_attribution("According to the SRS, three modules are defined.")
            == "Three modules are defined."
        )

    def test_based_on_the_document(self) -> None:
        """'Based on the document,' should be removed."""
        assert (
            strip_attribution(
                "Based on the document, the API uses REST endpoints."
            )
            == "The API uses REST endpoints."
        )

    def test_based_on_the_provided_context(self) -> None:
        """'Based on the provided context,' should be removed."""
        assert (
            strip_attribution(
                "Based on the provided context, the limit is 100 requests."
            )
            == "The limit is 100 requests."
        )

    def test_the_document_states_that(self) -> None:
        """'The document states that' should be removed and the remainder
        capitalised."""
        assert (
            strip_attribution(
                "The document states that the process is fully automated."
            )
            == "The process is fully automated."
        )

    def test_the_srs_document_states_that(self) -> None:
        """'The SRS document states that' should be removed and the
        remainder capitalised."""
        assert (
            strip_attribution(
                "The SRS document states that ten users are supported."
            )
            == "Ten users are supported."
        )

    def test_according_to_the_provided_document(self) -> None:
        """'According to the provided document,' should be removed."""
        assert (
            strip_attribution(
                "According to the provided document, the system is secure."
            )
            == "The system is secure."
        )

    def test_according_to_the_provided_cv(self) -> None:
        """'According to the provided CV,' should be removed."""
        assert (
            strip_attribution(
                "According to the provided CV, the candidate has 5 years."
            )
            == "The candidate has 5 years."
        )

    def test_according_to_the_provided_proposal(self) -> None:
        """'According to the provided proposal,' should be removed."""
        assert (
            strip_attribution(
                "According to the provided proposal, the budget is $10k."
            )
            == "The budget is $10k."
        )

    def test_by_team_iris(self) -> None:
        """'By team IRIS,' should be removed."""
        assert (
            strip_attribution(
                "By team IRIS, the solution was delivered on time."
            )
            == "The solution was delivered on time."
        )

    def test_the_document_mentions_that(self) -> None:
        """'The document mentions that' should be removed and the remainder
        capitalised."""
        assert (
            strip_attribution(
                "The document mentions that two modules are planned."
            )
            == "Two modules are planned."
        )

    def test_on_page_reference(self) -> None:
        """'On page 26,' should be removed."""
        assert (
            strip_attribution(
                "On page 26, the system is described as fast."
            )
            == "The system is described as fast."
        )

    def test_on_page_reference_no_space(self) -> None:
        """'On page26,' (no space) should be removed."""
        assert (
            strip_attribution(
                "On page26, the system is described as fast."
            )
            == "The system is described as fast."
        )

    def test_according_to_the_cv(self) -> None:
        """'According to the CV,' should be removed."""
        assert (
            strip_attribution(
                "According to the CV, the candidate has 5 years."
            )
            == "The candidate has 5 years."
        )

    def test_according_to_the_proposal(self) -> None:
        """'According to the proposal,' should be removed."""
        assert (
            strip_attribution(
                "According to the proposal, the budget is 10k."
            )
            == "The budget is 10k."
        )

    def test_as_mentioned_in_the_document(self) -> None:
        """'As mentioned in the document,' should be removed."""
        assert (
            strip_attribution(
                "As mentioned in the document, the system is secure."
            )
            == "The system is secure."
        )

    def test_this_is_mentioned_in_section(self) -> None:
        """'This is mentioned in the <section> section,' should be
        removed and the remainder capitalised."""
        assert (
            strip_attribution(
                "This is mentioned in the System Architecture Diagram section, "
                "the design is modular."
            )
            == "The design is modular."
        )


class TestNoChangeWhenNoAttribution:
    """Verify answers without attribution phrases are unchanged."""

    def test_plain_answer_unchanged(self) -> None:
        """A normal answer without attribution should be returned as-is."""
        text = "The purpose of TES is to automate teacher evaluation."
        assert strip_attribution(text) == text

    def test_trailing_mention_unchanged(self) -> None:
        """An attribution phrase in the middle/end of an answer should not
        be removed (only leading phrases are stripped)."""
        text = "The system is described in the document, which lists features."
        assert strip_attribution(text) == text

    def test_empty_string_unchanged(self) -> None:
        """An empty string should be returned unchanged."""
        assert strip_attribution("") == ""

    def test_whitespace_only_unchanged(self) -> None:
        """Whitespace-only input should be returned unchanged."""
        assert strip_attribution("   ") == "   "


class TestPreservation:
    """Verify formatting, markdown, and code blocks are preserved."""

    def test_code_block_preserved(self) -> None:
        """Markdown code blocks should be preserved."""
        text = "```python\nprint('hello')\n```"
        assert strip_attribution(text) == text

    def test_bullets_preserved(self) -> None:
        """Bullet lists should be preserved."""
        text = "- First point\n- Second point"
        assert strip_attribution(text) == text

    def test_leading_bullet_attribution_not_stripped(self) -> None:
        """An attribution phrase after a leading bullet is not at the very
        start of the string, so it is intentionally left unchanged."""
        text = "- According to the document, item one\n- item two"
        assert strip_attribution(text) == text

    def test_formatting_after_removal_preserved(self) -> None:
        """After removing a leading attribution, the remaining formatting
        (bullets, markdown) is preserved."""
        result = strip_attribution(
            "According to the document:\n- item one\n- item two"
        )
        assert result == "- item one\n- item two"


class TestDeterministic:
    """Verify the formatter is deterministic."""

    def test_same_input_same_output(self) -> None:
        """Repeated calls with the same input yield the same result."""
        text = "According to the document, the system is fast."
        assert strip_attribution(text) == strip_attribution(text)


class TestNoDependencies:
    """Verify the formatter has no unwanted dependencies."""

    def test_no_streamlit_import(self) -> None:
        """The formatter must not import Streamlit."""
        import src.services.response_formatter as mod

        source = mod.__file__
        if source:
            with open(source) as f:
                content = f.read()
            assert "streamlit" not in content.lower()

    def test_no_provider_import(self) -> None:
        """The formatter must not import any provider."""
        import src.services.response_formatter as mod

        source = mod.__file__
        if source:
            with open(source) as f:
                content = f.read()
            assert "ollama" not in content.lower()
            assert "groq" not in content.lower()
            assert "openai" not in content.lower()


class TestWhitespaceNormalization:
    """Verify safe whitespace normalisation."""

    def test_trailing_whitespace_removed(self) -> None:
        """Trailing whitespace on non-code lines is trimmed."""
        assert (
            strip_attribution("The system is fast.   ")
            == "The system is fast."
        )

    def test_excessive_blank_lines_collapsed(self) -> None:
        """Runs of more than one blank line are collapsed to one."""
        assert (
            strip_attribution(
                "First paragraph.\n\n\n\n\nSecond paragraph."
            )
            == "First paragraph.\n\nSecond paragraph."
        )

    def test_single_blank_line_preserved(self) -> None:
        """Intentional single blank lines between paragraphs are kept."""
        text = "First paragraph.\n\nSecond paragraph."
        assert strip_attribution(text) == text

    def test_code_block_whitespace_preserved(self) -> None:
        """Whitespace inside fenced code blocks is preserved verbatim."""
        text = (
            "```python\n"
            "def f():\n"
            "    return 1\n"
            "\n\n"
            "x = f()   \n"
            "```"
        )
        assert strip_attribution(text) == text

    def test_trailing_blank_lines_removed(self) -> None:
        """Trailing blank lines at the end of the answer are removed."""
        assert (
            strip_attribution("The system is fast.\n\n\n")
            == "The system is fast."
        )

    def test_no_content_meaning_change(self) -> None:
        """Only whitespace changes; content and order are preserved."""
        text = "Line one\n\n\nLine two."
        assert strip_attribution(text) == "Line one\n\nLine two."


class TestCapitalizationPreservesMarkdown:
    """Verify capitalization never corrupts markdown structure."""

    def test_bullet_after_attribution_not_capitalized(self) -> None:
        """A bullet list after a removed attribution is not corrupted."""
        assert (
            strip_attribution(
                "According to the document:\n- item one\n- item two"
            )
            == "- item one\n- item two"
        )

    def test_numbered_list_after_attribution_not_capitalized(self) -> None:
        """A numbered list after a removed attribution is not corrupted."""
        assert (
            strip_attribution(
                "Based on the document:\n1. first\n2. second"
            )
            == "1. first\n2. second"
        )

    def test_code_fence_after_attribution_not_capitalized(self) -> None:
        """A code fence after a removed attribution is not corrupted."""
        assert (
            strip_attribution(
                "Based on the document:\n```python\nprint('hi')\n```"
            )
            == "```python\nprint('hi')\n```"
        )

    def test_heading_after_attribution_not_capitalized(self) -> None:
        """A heading after a removed attribution is not corrupted."""
        assert (
            strip_attribution(
                "The document states that:\n# Overview"
            )
            == "# Overview"
        )

    def test_inline_code_preserved(self) -> None:
        """An answer beginning with inline code is preserved unchanged (a
        leading markdown backtick is a structural symbol and is not
        re-capitalised)."""
        result = strip_attribution("According to the document: `python` is fast.")
        assert result == "`python` is fast."

    def test_plain_alpha_capitalized(self) -> None:
        """Plain text after attribution is correctly capitalised."""
        assert (
            strip_attribution(
                "According to the document, the system is fast."
            )
            == "The system is fast."
        )


class TestMarkdownConsistency:
    """Verify conservative markdown consistency improvements."""

    def test_unclosed_fence_closed(self) -> None:
        """A single unclosed triple-backtick fence is closed."""
        text = "```python\nprint('hello')"
        assert strip_attribution(text) == "```python\nprint('hello')\n```"

    def test_balanced_fence_unchanged(self) -> None:
        """Balanced code fences are left unchanged."""
        text = "```python\nprint('hello')\n```"
        assert strip_attribution(text) == text

    def test_fence_language_tag_preserved(self) -> None:
        """The opening language tag is preserved when closing a fence."""
        text = "Here is the code:\n```python\nx = 1\nprint(x)"
        result = strip_attribution(text)
        assert result.endswith("```")
        assert "```python\nx = 1\nprint(x)\n```" in result

    def test_code_content_not_modified(self) -> None:
        """Code content inside an unclosed fence is never rewritten."""
        text = "```python\nx = 1  \nprint(x)"
        result = strip_attribution(text)
        # The trailing spaces inside the code line are preserved (fence-aware).
        assert "x = 1  \n" in result
        assert result.endswith("```")

    def test_broken_numbered_list_fixed(self) -> None:
        """A broken numbered list ('one2.') gets a line break restored."""
        text = "Feature one2. Feature two3. Feature three"
        assert (
            strip_attribution(text)
            == "Feature one\n2. Feature two\n3. Feature three"
        )

    def test_missing_list_separation_fixed(self) -> None:
        """A list with missing separation before '2.' gets a line break."""
        text = "Inventory tracking for admins2. User management features"
        assert (
            strip_attribution(text)
            == "Inventory tracking for admins\n2. User management features"
        )

    def test_normal_number_text_unchanged(self) -> None:
        """Numbers with a preceding space/digit are not treated as lists."""
        text = "Version 2.0 is released in 2024."
        assert strip_attribution(text) == "Version 2.0 is released in 2024."

    def test_date_unchanged(self) -> None:
        """Dates and ordinal-like text are not turned into lists."""
        text = "The report was published on 5. March and updated 12. December."
        assert (
            strip_attribution(text)
            == "The report was published on 5. March and updated 12. December."
        )

    def test_valid_numbered_list_unchanged(self) -> None:
        """A properly formatted numbered list is preserved."""
        text = "1. Feature one\n2. Feature two\n3. Feature three"
        assert strip_attribution(text) == text

    def test_broken_list_inside_text_after_attribution(self) -> None:
        """Attribution removal composes with list fixing."""
        text = (
            "According to the document, the system has:\n"
            "Feature one2. Feature two"
        )
        assert (
            strip_attribution(text)
            == "The system has:\nFeature one\n2. Feature two"
        )

    def test_broken_list_in_code_block_untouched(self) -> None:
        """Broken-looking markers inside code are never modified."""
        text = "```\nFeature one2. Feature two\n```"
        assert strip_attribution(text) == text
