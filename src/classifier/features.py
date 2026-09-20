"""Feature extraction from prompts for the complexity classifier.

Converts raw prompt text into a numeric feature vector that the
sklearn model can consume. Features are designed around the
programming / code-generation domain.
"""

from __future__ import annotations

import re
from typing import Optional

import numpy as np


# ── Keyword groups (domain: programming & code learning) ────────────

SIMPLE_KEYWORDS = [
    "what is", "define", "syntax", "print", "hello world", "convert",
    "format", "extract", "list", "rename", "translate to", "rewrite",
    "fix typo", "correct syntax", "boilerplate", "template", "basic",
    "simple", "variable", "constant", "import", "how to declare",
]

MODERATE_KEYWORDS = [
    "explain", "summarize", "debug", "review", "classify", "difference",
    "compare", "pros and cons", "when to use", "example of", "refactor",
    "best practice", "error handling", "edge case", "test case",
    "walk through", "step by step", "how does", "why does", "trace",
    "code review", "improve", "optimize this", "readable",
]

COMPLEX_KEYWORDS = [
    "architect", "design pattern", "system design", "implement from scratch",
    "algorithm", "data structure", "concurrent", "async", "distributed",
    "microservice", "scale", "performance", "benchmark", "trade-off",
    "security", "vulnerability", "machine learning", "neural network",
    "analyze complexity", "big o", "dynamic programming", "graph",
    "multi-step", "recursive", "backtracking", "advanced", "production",
    "deploy", "ci/cd", "pipeline", "orchestrate", "integrate",
]

OUTPUT_FORMAT_MARKERS = {
    "json": ["json", "dictionary", "dict", "key-value"],
    "table": ["table", "csv", "spreadsheet", "columns", "rows"],
    "code": ["function", "class", "method", "module", "script", "program"],
    "markdown": ["markdown", "documentation", "readme", "doc"],
    "essay": ["essay", "paragraph", "write about", "discuss"],
}


class FeatureExtractor:
    """Extracts a numeric feature vector from a prompt string.

    Feature vector layout (12 features):
        [0]  token_count          — approximate word count
        [1]  char_count           — character length
        [2]  sentence_count       — number of sentences
        [3]  question_count       — number of question marks
        [4]  constraint_count     — number of explicit constraints/requirements
        [5]  simple_keyword_score — hit ratio against simple keywords
        [6]  moderate_keyword_score
        [7]  complex_keyword_score
        [8]  has_code_block       — 1 if prompt contains code fences
        [9]  has_context          — 1 if prompt provides context/data
        [10] output_complexity    — 0 (free text) to 3 (structured output)
        [11] instruction_depth   — nested instruction count
    """

    FEATURE_NAMES = [
        "token_count",
        "char_count",
        "sentence_count",
        "question_count",
        "constraint_count",
        "simple_keyword_score",
        "moderate_keyword_score",
        "complex_keyword_score",
        "has_code_block",
        "has_context",
        "output_complexity",
        "instruction_depth",
    ]

    def extract(self, prompt: str) -> np.ndarray:
        """Return a 1-D numpy array of features for the given prompt."""
        lower = prompt.lower()

        return np.array(
            [
                self._token_count(prompt),
                len(prompt),
                self._sentence_count(prompt),
                self._question_count(prompt),
                self._constraint_count(lower),
                self._keyword_score(lower, SIMPLE_KEYWORDS),
                self._keyword_score(lower, MODERATE_KEYWORDS),
                self._keyword_score(lower, COMPLEX_KEYWORDS),
                float(self._has_code_block(prompt)),
                float(self._has_context(lower)),
                self._output_complexity(lower),
                self._instruction_depth(lower),
            ],
            dtype=np.float64,
        )

    def extract_batch(self, prompts: list[str]) -> np.ndarray:
        """Extract features for a batch of prompts → (N, 12) array."""
        return np.vstack([self.extract(p) for p in prompts])

    # ── Individual feature extractors ───────────────────────────────────

    @staticmethod
    def _token_count(text: str) -> int:
        return len(text.split())

    @staticmethod
    def _sentence_count(text: str) -> int:
        return max(len(re.split(r"[.!?]+", text)) - 1, 1)

    @staticmethod
    def _question_count(text: str) -> int:
        return text.count("?")

    @staticmethod
    def _constraint_count(text: str) -> int:
        """Count explicit constraints like 'must', 'should', 'ensure', numbered lists."""
        markers = [
            "must", "should", "ensure", "require", "constraint",
            "do not", "don't", "avoid", "make sure", "important",
            "at least", "at most", "no more than", "exactly",
        ]
        count = sum(1 for m in markers if m in text)
        # Also count numbered requirements: "1.", "2.", etc.
        count += len(re.findall(r"\b\d+\.\s", text))
        return count

    @staticmethod
    def _keyword_score(text: str, keywords: list[str]) -> float:
        """Fraction of keywords found in the text."""
        if not keywords:
            return 0.0
        hits = sum(1 for kw in keywords if kw in text)
        return hits / len(keywords)

    @staticmethod
    def _has_code_block(text: str) -> bool:
        return "```" in text or "    " in text

    @staticmethod
    def _has_context(text: str) -> bool:
        """Check if the prompt provides inline context/data."""
        context_markers = [
            "given the following", "here is", "below is", "consider this",
            "the code is", "this is my", "i have", "the error is",
            "the output is", "the input is", "the data",
        ]
        return any(m in text for m in context_markers)

    @staticmethod
    def _output_complexity(text: str) -> float:
        """Score 0–3 based on how structured the expected output is."""
        score = 0.0
        for fmt, markers in OUTPUT_FORMAT_MARKERS.items():
            if any(m in text for m in markers):
                if fmt in ("json", "table"):
                    score = max(score, 3.0)
                elif fmt == "code":
                    score = max(score, 2.0)
                elif fmt in ("markdown", "essay"):
                    score = max(score, 1.0)
        return score

    @staticmethod
    def _instruction_depth(text: str) -> int:
        """Count nested or chained instructions (then, after that, next, also)."""
        chaining = [
            "then ", "after that", "next ", "also ", "additionally",
            "furthermore", "finally ", "step 1", "step 2", "step 3",
            "first ", "second ", "third ",
        ]
        return sum(1 for c in chaining if c in text)
