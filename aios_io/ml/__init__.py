"""Advanced machine learning modules for AIOS-IO.

This subpackage consolidates experimental algorithms originally scattered in the
"Machine Learning Logic" directory so they can be imported and tested like
standard Python modules.
"""

# Expose commonly used classes to the package namespace for convenience.
try:  # pragma: no cover - optional heavy dependencies
    from .twmrto_compression import QuantumHuffmanEncoder, NeuralCompressionNetwork  # noqa: F401
except Exception:  # pragma: no cover
    QuantumHuffmanEncoder = NeuralCompressionNetwork = None  # type: ignore

try:  # pragma: no cover - optional heavy dependencies
    from .nlp_to_code_engine import CodeGenerationRequest, GeneratedCode  # noqa: F401
except Exception:  # pragma: no cover
    CodeGenerationRequest = GeneratedCode = None  # type: ignore

from .compression_service import encode_and_enqueue  # noqa: F401
from .orchestrator_hooks import trigger_ml_tasks  # noqa: F401

try:  # pragma: no cover - optional heavy dependencies
    from .rby_core_engine import RBYState, RBYQuantumProcessor, create_task  # noqa: F401
except Exception:  # pragma: no cover
    # Fallback implementations when numpy/torch are unavailable
    RBYState = None  # type: ignore
    RBYQuantumProcessor = None  # type: ignore
    from ..task import Task  # type: ignore
    from ..node import Node  # type: ignore

    async def create_task(node: Node, name: str | None = None) -> Task:  # type: ignore
        """Fallback task factory that performs no computation."""

        def action() -> None:
            pass

        task_name = name or f"rby_core_{node.node_id}"
        return Task(task_name, "R", action)
