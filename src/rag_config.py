import os
from dotenv import load_dotenv
from sentence_transformers import SentenceTransformer
from qdrant_client import QdrantClient, models
from qdrant_client.http.models import Distance, VectorParams
load_dotenv()

class RAGConfig:
    def __init__(self, model_id, dataset):
        self.model_id = model_id
        self.dataset = dataset
        self.model = self.resolve_model()
        self.dimension = self.get_dimension()
        self.client = QdrantClient(url=os.getenv("QDRANT_URL"),
                                    grpc_port=int(os.getenv("QDRANT_GRPC_PORT")),
                                    prefer_grpc=True,
                                )
        self.collection_name = self.initialize_collection()
        
    def resolve_model(self):
        if self.model_id == "sentence-transformers/all-MiniLM-L12-v2":
            return SentenceTransformer(self.model_id)
        raise ValueError(f"Unsupported model_id: {self.model_id}")

    def get_dimension(self):
        if self.model_id == "sentence-transformers/all-MiniLM-L12-v2":
            return self.model.get_sentence_embedding_dimension()

    def initialize_collection(self):
        collection_name=f"{self.model_id.replace("/", "_")}_{self.dataset.rsplit("/", 1)[-1].removesuffix(".xes")}"
        if self.client.collection_exists(collection_name):
            self.client.recreate_collection(
                collection_name=collection_name,
                vectors_config=VectorParams(size=self.dimension, distance=Distance.COSINE),
            )
        else:
            self.client.create_collection(
                collection_name=collection_name,
                vectors_config=VectorParams(size=self.dimension, distance=Distance.COSINE),
            )
        return collection_name
    
    def encode(self, texts: list[str]):
        if self.model_id.startswith("sentence-transformers"):
            return self.model.encode(texts, normalize_embeddings=True)
        raise ValueError(f"Unsupported encode provider for model_id: {self.model_id}")