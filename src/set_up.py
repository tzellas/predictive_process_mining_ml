from sentence_transformers import SentenceTransformer

class RAGConfig:
    def __init__(self, model_id):
        self.model_id = model_id
        self.model = self.resolve_model()
        self.dimension = self.get_dimension()
    
    def resolve_model(self):
        if self.model_id == "sentence-transformers/all-MiniLM-L12-v2":
            return SentenceTransformer(self.model_id)
        raise ValueError(f"Unsupported model_id: {self.model_id}")

    def get_dimension(self):
        if self.model_id == "sentence-transformers/all-MiniLM-L12-v2":
            return self.model.get_sentence_embedding_dimension()

    def encode(self, texts: list[str]):
        if self.model_id.startswith("sentence-transformers"):
            return self.model.encode(texts, normalize_embeddings=True)
        raise ValueError(f"Unsupported encode provider for model_id: {self.model_id}")

