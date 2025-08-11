"""Advanced machine learning modules for AIOS-IO.

This subpackage consolidates experimental algorithms originally scattered in the
"Machine Learning Logic" directory so they can be imported and tested like
standard Python modules.
"""

# Expose commonly used classes to the package namespace for convenience.
from .rby_core_engine import RBYState, RBYQuantumProcessor  # noqa: F401
from .twmrto_compression import QuantumHuffmanEncoder, NeuralCompressionNetwork  # noqa: F401
from .nlp_to_code_engine import CodeGenerationRequest, GeneratedCode  # noqa: F401
