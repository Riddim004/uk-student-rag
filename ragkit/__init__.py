"""ragkit — a small, tested, hybrid-retrieval RAG engine.

The engine is generic; the bundled data happens to be UK student-visa guidance.
Everything is dependency-injected (embedder, LLM) so the whole pipeline and its
retrieval-quality evaluation run under pytest with no API key and no network.
"""

__version__ = "2.0.0"
