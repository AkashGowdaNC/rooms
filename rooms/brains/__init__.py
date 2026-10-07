"""Inference backends for agents.

An AgentBrain is the unit that turns a conversation history into the next
agent response. Concrete brains live next to this module; the abstract base
class here is the contract the rest of the framework will depend on.
"""

from abc import ABC, abstractmethod
from typing import Any, Dict, List, Optional


class AgentBrain(ABC):
    """Abstract base class for agent inference backends.

    Implementations receive the same context the current ``Agent`` passes to
    LiteLLM (a list of ``{"role": ..., "content": ...}`` messages) and return
    a structured result with the response text and any skill-execution events
    raised during generation.
    """

    @abstractmethod
    def generate(
        self,
        context_messages: List[Dict[str, str]],
        override_params: Optional[Dict[str, Any]] = None,
    ) -> Dict[str, Any]:
        """Generate a response from the given context.

        Args:
            context_messages: The conversation history, already including the
                system message as the first element.
            override_params: Optional runtime parameters (e.g. temperature)
                that override the brain's configured defaults.

        Returns:
            A dict with at least ``content`` (a string, possibly empty) and
            ``skill_events`` (a list, possibly empty). The shape matches
            ``Agent.generate_response_with_events`` today so callers can
            swap implementations without touching their parsing code.
        """
        raise NotImplementedError