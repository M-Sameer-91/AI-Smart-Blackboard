"""Provider-neutral interface for optional Blackboard AI explanations."""

from abc import ABC, abstractmethod


class AIProviderError(RuntimeError):
    """A safe, user-displayable AI provider failure."""


class AIProvider(ABC):
    @property
    @abstractmethod
    def is_available(self) -> bool:
        """Whether this provider has sufficient local configuration."""

    @abstractmethod
    def explain(self, board_context: str) -> str:
        """Return a concise student-facing explanation for structured board context."""
