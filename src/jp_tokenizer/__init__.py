__all__ = ["HybridTokenizer"]

def __getattr__(name: str):
    if name == "HybridTokenizer":
        from .hybrid import HybridTokenizer #try importing locally, could fix torch issue
        return HybridTokenizer
    raise AttributeError(name)