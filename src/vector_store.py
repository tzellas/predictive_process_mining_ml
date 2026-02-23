import pandas as pd
from qdrant_client import QdrantClient, models
from qdrant_client.http.models import Distance, VectorParams

def store_embeddings(data_csv_path, embed_model, collection_name, client = QdrantClient(":memory:")):

  df = pd.read_csv(data_csv_path)

  # split the prefix and last values from the prediction
  texts = df["prefix"].tolist()
  predictions = df["prediction"].tolist()

  embeddings = embed_model.encode(texts, normalize_embeddings=True)

  client.recreate_collection(
      collection_name=collection_name,
      vectors_config=VectorParams(size=embeddings.shape[1], distance=Distance.COSINE),
  )

  points = [
      models.PointStruct(
          id=i,
          vector=embeddings[i].tolist(),
          payload={"prefix": texts[i], "prediction": predictions[i]},
      )
      for i in range(len(texts))
  ]

  client.upsert(collection_name=collection_name, points=points)
  return client, collection_name