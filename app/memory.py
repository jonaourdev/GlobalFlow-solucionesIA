from __future__ import annotations

from dataclasses import dataclass, field
from app.langchain_compat import ConversationBufferWindowMemory
from app.models import ExecutionTrace


@dataclass
class AgentMemory:

    traces: list[ExecutionTrace] = field(default_factory=list)

    def __post_init__(self) -> None:
        self.langchain_memory = None
        if ConversationBufferWindowMemory is not None:
            self.langchain_memory = ConversationBufferWindowMemory(
                k=6,
                memory_key="historial_flujo",
                return_messages=True,
            )

    def add(self, paso: str, agente: str, detalle: str) -> None:
        trace = ExecutionTrace(paso=paso, agente=agente, detalle=detalle)
        self.traces.append(trace)

        if self.langchain_memory is not None:
            self.langchain_memory.save_context(
                {"input": f"{paso} - {agente}"},
                {"output": detalle},
            )

    def summary(self) -> list[str]:
        return [f"{t.paso} | {t.agente}: {t.detalle}" for t in self.traces]
