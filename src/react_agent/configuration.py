"""Define the configurable parameters for the agent."""

from __future__ import annotations

from dataclasses import dataclass, field, fields
from typing import Annotated, Optional

from langchain_core.runnables import RunnableConfig, ensure_config

from react_agent import prompts
# import prompts


@dataclass(kw_only=True)
class Configuration:
    """The configuration for the agent."""
    # Set dynamically based on agent_type
    # prompts: dict = field(init=False)

    # Define the dictionary of system prompts
    prompts: dict = field(
        default_factory=lambda: {
            "system_prompt": prompts.SYSTEM_PROMPT,
            "supervisor_agent": prompts.SUPERVISOR_PROMPT,
            "researcher_agent": prompts.RESEARCHER_PROMPT,
            "summary_agent": prompts.SUMMARY_PROMPT,
            "marketing_prompt": prompts.MARKETING_PROMPT
        },
        metadata={
            "description": "A dictionary of prompts for different agent types."
        },
    )

    model: Annotated[str, {"__template_metadata__": {"kind": "llm"}}] = field(
        default="openai/gpt-4o",
        metadata={
            "description": "The name of the language model to use for the agent's main interactions. "
            "Should be in the form: provider/model-name."
        },
    )

    max_search_results: int = field(
        default=10,
        metadata={
            "description": "The maximum number of search results to return for each search query."
        },
    )

    @classmethod
    def from_runnable_config(
        cls, config: Optional[RunnableConfig] = None
    ) -> Configuration:
        """Create a Configuration instance from a RunnableConfig object."""
        config = ensure_config(config)
        configurable = config.get("configurable") or {}
        _fields = {f.name for f in fields(cls) if f.init}
        return cls(**{k: v for k, v in configurable.items() if k in _fields})

    def __post_init__(self):
        if not self.prompts:
            self.prompts = {
                "system_prompt": prompts.SYSTEM_PROMPT,
                "supervisor_agent": prompts.SUPERVISOR_PROMPT,
                "researcher_agent": prompts.RESEARCHER_PROMPT,
                "summary_agent": prompts.SUMMARY_PROMPT,
                "marketing_prompt": prompts.MARKETING_PROMPT
            }

    def get_prompt(self, agent_type: str) -> str:
        """Retrieve the prompt for the given agent type."""
        return self.prompts.get(agent_type, self.prompts["system_prompt"])
