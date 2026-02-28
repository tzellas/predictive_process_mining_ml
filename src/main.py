from llm_api import api_call
from event_log_preprocessing import *
from rag_config import *
from vector_api import *

def main():
    # model_id = "sentence-transformers/all-MiniLM-L12-v2"
    # dataset = "/home/apost/projects/Diplomatiki/data/xes_logs/BPI_Challenge_2012.xes"
    # config = RAGConfig(model_id, dataset)
    
    # prefix_list = log_processor(dataset, True)
    # store_embeddings(prefix_list=prefix_list, config=config)

    # print(f"Done. Collection: {[c.name for c in config.client.get_collections().collections]}")

    qprefix = "A_SUBMITTED,A_PARTLYSUBMITTED,A_PREACCEPTED,W_Completeren aanvraag,W_Completeren aanvraag,W_Completeren aanvraag,W_Completeren aanvraag,W_Completeren aanvraag,W_Completeren aanvraag,W_Completeren aanvraag,A_ACCEPTED,O_SELECTED,A_FINALIZED,O_CREATED,O_SENT,W_Completeren aanvraag,W_Nabellen offertes,W_Nabellen offertes,W_Nabellen offertes,O_SENT_BACK,W_Nabellen offertes,W_Valideren aanvraag,W_Nabellen incomplete dossiers,W_Nabellen incomplete dossiers,W_Nabellen incomplete dossiers,W_Nabellen incomplete dossiers,W_Nabellen incomplete dossiers,W_Nabellen incomplete dossiers,W_Nabellen incomplete dossiers,W_Valideren aanvraag,W_Nabellen incomplete dossiers,W_Nabellen incomplete dossiers,W_Valideren aanvraag,W_Valideren aanvraag,W_Nabellen incomplete dossiers,W_Nabellen incomplete dossiers,W_Nabellen incomplete dossiers,W_Nabellen incomplete dossiers,W_Nabellen incomplete dossiers,W_Nabellen incomplete dossiers,W_Nabellen incomplete dossiers,W_Nabellen incomplete dossiers,W_Nabellen incomplete dossiers,W_Nabellen incomplete dossiers,W_Nabellen incomplete dossiers,W_Nabellen incomplete dossiers,W_Nabellen incomplete dossiers,W_Nabellen incomplete dossiers,W_Nabellen incomplete dossiers,W_Valideren aanvraag,W_Valideren aanvraag,A_APPROVED,O_ACCEPTED - Values: {'orre': '10609', 'litr': 'COMPLETE', 'titi': '2011-11-15 12:50:28.812000+00:00'}"
    collection_name = "sentence-transformers_all-MiniLM-L12-v2_BPI_Challenge_2012"
    config = RAGConfig(collection_name=collection_name)
    ctxt, hits = retrieve_similar_prefixes(config, qprefix)
    print("########Context Ready#########")
    print(api_call(ctxt, qprefix))

if __name__ == "__main__":
    main()