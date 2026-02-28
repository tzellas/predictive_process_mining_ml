import pandas as pd
from qdrant_client import QdrantClient, models
from qdrant_client.http.models import Distance, VectorParams

def store_embeddings(config, prefix_list):

    # split the prefix and last values from the prediction
    texts = []
    predictions = []

    for prefix in prefix_list:
        parts = prefix.rsplit(" - ", 1)
        if len(parts) != 2:
            continue
        texts.append(parts[0].strip()) 
        predictions.append(parts[1].strip())
    
    embeddings = config.encode(texts)

    points = [
        models.PointStruct(
            id=i,
            vector=embeddings[i].tolist(),
            payload={"prefix": texts[i], "prediction": predictions[i]},
        )
        for i in range(len(texts))
    ]

    config.client.upsert(collection_name=config.collection_name, points=points)
    return 


def retrieve_similar_prefixes(config, query_full_prefix, top_k: int = 5):
    qvec = config.encode([query_full_prefix])[0].tolist()

    res = config.client.query_points(
        collection_name=config.collection_name,
        query=qvec,
        limit=top_k,
        with_payload=True,
    )
    hits = res.points

    context = {}
    for rank, h in enumerate(hits, start=1):
        p = h.payload or {}
       
        context[f"trace_{rank}"] = {
            "prefix": p.get("prefix", ""),
            "prediction": p.get("prediction", ""),
            "score": round(float(h.score), 4),
        }

    return context, hits
