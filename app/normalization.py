import sys, os
import pandas as pd
import torch
from nlp4bia.datasets.benchmark.distemist import DistemistLoader, DistemistGazetteer
from nlp4bia.linking.retrievers import DenseRetriever
from sentence_transformers import SentenceTransformer
from simple_inference import join_all_entities


def run_nel_inference(gazetteer, input_file, output_file, model, k=10, store_vector_db=None, vector_db_file='vector_db.pt', input_mentions=None, save_output=True):

    print('Loading model...')
    st_model = SentenceTransformer(model)
    print('Model loaded.')

    gazetteer_df = pd.read_csv(gazetteer, sep='\t')
    if input_mentions is None:
        input_df = pd.read_csv(input_file, sep='\t')
        mentions = input_df.span.unique().tolist()
    else:
        mentions = input_mentions
        input_df = pd.DataFrame({"span": mentions})

    terms = gazetteer_df["term"].tolist()
        
    #term2code = gazetteer_df.set_index("term")["code"].to_dict()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    if os.path.exists(vector_db_file):
        print("Loading vector database from file...")
        vector_db = torch.load(vector_db_file, map_location=device)
    else:
        print("Vector database not found. Computing vector database...")
        vector_db = st_model.encode(
            terms,
            show_progress_bar=True, 
            convert_to_tensor=True,
            batch_size=4096,
            device=device.type  # make sure encoding runs on the same device
        )


    if store_vector_db is not None:
        torch.save(vector_db, store_vector_db)
        print(f"Vector database stored at {store_vector_db}")

    biencoder = DenseRetriever(df_candidates=gazetteer_df, vector_db=vector_db, model_or_path=st_model)

    candidates = biencoder.retrieve_top_k(
        mentions, 
        k=k, 
        input_format="text",
        return_documents=True
    )
    candidates_df = pd.DataFrame(candidates).rename(columns={"mention": "span"})
    #print('\n', input_df, '\n')
    #print('\n', candidates_df, '\n')
    candidates_df = input_df.merge(candidates_df, on="span", how="left")
    #print('\n', candidates_df, '\n')
    candidates_df["similarity"] = candidates_df["similarity"].apply(
        lambda sims: [round(s, 4) for s in sims] if isinstance(sims, list) else round(sims, 4)
    )
    candidates_df = candidates_df.rename(columns={"similarity": "similarities"})
    cols_to_move = ["span", "codes", "terms", "similarities"]
    new_order = [col for col in candidates_df.columns if col not in cols_to_move] + cols_to_move
    output = candidates_df[new_order]
    
    if save_output:
        output.to_csv(output_file, sep="\t", index=False)
    
    return output


def nel_inference(nerl_results, nerl_models_config, combined=True):
    # nerl_results contains the output of the NER step. This function adds the NEL fields information

    for mentions_list, config in zip(nerl_results, nerl_models_config):
        mentions = []
        mention_dicts = []
        for mentions_text in mentions_list:
            for mention_dict in mentions_text:
                mentions.append(mention_dict['span'])
                mention_dicts.append(mention_dict)

        if len(mentions) == 0:
            continue

        nel_model_path = config["nel_model_path"]
        try:
            gazetteer_path = config["gazetteer_path"]
        except KeyError:
            gazetteer_path = None
        try:
            vectorized_gazetteer_path = config["vectorized_gazetteer_path"]
        except KeyError:
            vectorized_gazetteer_path = None

        output = run_nel_inference(gazetteer=gazetteer_path,
                                   input_file=None, # No input file, mentions are directly provided
                                   output_file=None, # No need to save output to file
                                   model=nel_model_path,
                                   k=1, # Only the top candidate is wanted
                                   vector_db_file=vectorized_gazetteer_path,
                                   input_mentions=mentions,
                                   save_output=False)

        #mention_dicts = [mention_dict for mentions_text in mentions_list for mention_dict in mentions_text]

        # Modify the mention_dicts in place to add the NEL fields
        for m, c, t, s in zip(mention_dicts, output.codes.to_list(), output.terms.to_list(), output.similarities.to_list()):
            m["code"] = c[0]
            m["term"] = t[0]
            m["nel_score"] = s[0]

    if combined:
        nerl_results = join_all_entities(nerl_results)

    return nerl_results



from optparse import OptionParser

def main(argv = None):
    parser = OptionParser()
    parser.add_option("-g", "--gazetteer", dest="gazetteer", help="gazetteer, tab-separated values extension (.tsv)", default="SpanishSnomed.tsv")
    parser.add_option("-i", "--input", dest="input_file", help="input file, tab-separated values extension (.tsv)", default="input.tsv")
    parser.add_option("-o", "--output", dest="output_file", help="output file, tab-separated values extension (.tsv)",default="output.tsv")
    parser.add_option("-m", "--model", dest="model", help="model to be used", default="ICB-UMA/ClinLinker-KB-GP")
    parser.add_option("-k", "--top_k", dest="k", type="int", help="number of top candidates to retrieve (default 10)", default=10)
    parser.add_option("-s", "--store_vector_db", dest="store_vector_db", help="path to the file if the vector database of the gazetteer terms wants to be stored to be reused", default=None)
    parser.add_option("-v", "--vector_db_file", dest="vector_db_file", help="path to the vector database of the gazetteer terms (default 'vector_db.pt')", default='vector_db.pt')
    #parser.add_option("-tsv", "--tsv_format", dest="tsv", type="bool", help="whether to store the output in TSV format", default=False)
    (options, args) = parser.parse_args(argv)

    run_nel_inference(
        gazetteer=options.gazetteer,
        input_file=options.input_file,
        output_file=options.output_file,
        model=options.model,
        k=options.k,
        store_vector_db=options.store_vector_db,
        vector_db_file=options.vector_db_file
    )


if __name__ == "__main__":
  sys.exit(main())
