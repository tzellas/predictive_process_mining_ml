import os
from pathlib import Path
from dotenv import load_dotenv
from sentence_transformers import SentenceTransformer
from qdrant_client import QdrantClient, models
from qdrant_client.http.models import Distance, VectorParams
load_dotenv()

EMBEDDING_MODELS = ["sentence-transformers/all-MiniLM-L12-v2"]
class RAGConfig:
    def __init__(self, model_id: str = None, dataset: str = None, collection_name: str = None):
        self.client = QdrantClient(host="localhost",
                                    grpc_port=int(os.getenv("QDRANT_GRPC_PORT")),
                                    prefer_grpc=True,
                                )
        
        if ((model_id or dataset) and collection_name) or (not collection_name and ((not model_id) or (not dataset))):
             raise ValueError(f"Please provide model_id and dataset or only collection name to get the configuration obj")
         
        if collection_name:
            success = False
            for model in EMBEDDING_MODELS:
                if collection_name.startswith(model.replace("/","_")):
                    self.collection_name = collection_name
                    self.model_id = model
                    self.dataset = collection_name.removeprefix(model.replace("/","_") + "_")
                    self.model = self.resolve_model()
                    self.dimension = self.get_dimension()
                    success = True
                    break
            if not success:
                raise ValueError(f"Wrong collection name.\n Try one of these: {"\n".join(c.name for c in self.client.get_collections().collections)}")
        else:        
            self.model_id = model_id
            self.dataset = dataset
            self.model = self.resolve_model()
            self.dimension = self.get_dimension()
            self.collection_name = self.initialize_collection()

    
    def resolve_model(self):
        if self.model_id == "sentence-transformers/all-MiniLM-L12-v2":
            return SentenceTransformer(self.model_id)
        raise ValueError(f"Unsupported model_id: {self.model_id}")

    def get_dimension(self):
        if self.model_id == "sentence-transformers/all-MiniLM-L12-v2":
            return self.model.get_sentence_embedding_dimension()

    def initialize_collection(self):
        collection_name=f"{self.model_id.replace("/", "_")}_{Path(self.dataset).stem}"
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