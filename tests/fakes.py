"""Test doubles shared by the agent and API tests."""

from langchain_core.language_models import BaseChatModel
from langchain_core.messages import AIMessage
from langchain_core.outputs import ChatGeneration, ChatResult


class ScriptedModel(BaseChatModel):
    """Replays AIMessages in order and records every message list it was shown."""

    script: list
    seen: list = []

    @property
    def _llm_type(self) -> str:
        return "scripted"

    def bind_tools(self, tools, **kwargs):
        return self

    def _generate(self, messages, stop=None, run_manager=None, **kwargs):
        self.seen.append(list(messages))
        msg = self.script.pop(0) if self.script else AIMessage("(script exhausted)")
        return ChatResult(generations=[ChatGeneration(message=msg)])


def call(name: str, **args) -> AIMessage:
    return AIMessage("", tool_calls=[{"name": name, "args": args, "id": f"c-{name}-{len(args)}"}])
