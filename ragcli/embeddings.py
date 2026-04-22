from sentence_transformers import SentenceTransformer

DEFAULT_MODEL = "all-MiniLM-L6-v2"

_model_cache: SentenceTransformer | None = None


def get_model(model_name: str = DEFAULT_MODEL) -> SentenceTransformer:
    global _model_cache
    if _model_cache is None:
        _model_cache = SentenceTransformer(model_name)
    return _model_cache


def embed_texts(texts: list[str], model_name: str = DEFAULT_MODEL) -> list[list[float]]:
    model = get_model(model_name)
    embeddings = model.encode(texts, show_progress_bar=len(texts) > 50)
    return embeddings.tolist()


def embed_query(query: str, model_name: str = DEFAULT_MODEL) -> list[float]:
    model = get_model(model_name)
    return model.encode(query).tolist()
