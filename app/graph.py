from app.orchestrator import GlobalFlowOrchestrator


class RunnableGlobalFlow:
    def __init__(self) -> None:
        self.orchestrator = GlobalFlowOrchestrator()

    def invoke(self, state: dict):
        raw_text = state.get("raw_invoice_text", "")
        file_name = state.get("file_name", "entrada_manual")
        return self.orchestrator.run(raw_invoice_text=raw_text, file_name=file_name)


def build_graph():
    return RunnableGlobalFlow()
