"""CoRG: training-free remote sensing visual grounding with frozen MLLMs.

Modules: corg.interface (adaptive grounding interface), corg.consensus (hypothesis consensus),
corg.refine (consensus-guided refinement and geometric guard), corg.pipeline (the complete method).
"""
__all__ = ["CoRG"]


def __getattr__(name):
    if name == "CoRG":
        from .pipeline import CoRG
        return CoRG
    raise AttributeError(name)
