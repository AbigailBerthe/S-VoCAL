from cleaner import OutputCleaner
from evaluation_metrics import compute_f1_score, f1_list, compute_f1_score_soft, compute_bertscore
import pandas as pd
from embeddings_eval import load_model, compare_gold_predicted, mean_cos
from datetime import datetime
import sys
import numpy as np



def load_and_clean(gold_df, predicted, datatype, attributes):
    if datatype == "baseline":
        clean = False
    else:
        clean = True


    if clean:
        predicted = OutputCleaner.extract_last_part(predicted, list_attributes=attributes)


    return gold_df, predicted



def main(datatype, rag, attributes, current_time, model_used):
    all_dfs = []
    
    print("starting")
    folder_gold = "../Data" 
    folder = f"../Data/{datatype}"
    gold_df = pd.read_json(f"{folder_gold}/S-VoCAL_dataset.jsonl", lines=True)

    if datatype == "baseline":
        path = f"{folder}/baseline.csv"
    else: 
        path = f"{folder}/{datatype}_{model_used}_rag_{rag}_{'_'.join(attributes)}_{current_time}.csv"
    predicted = pd.read_csv(path, sep=';', quotechar='"', encoding='utf-8')
    gold_df, predicted = load_and_clean(gold_df, predicted, datatype, attributes)
    print("dfs loaded")

    needs_embeddings = any(attr in attributes for attr in ['residence', 'origin', 'occupation', 'physical_health', 'type'])
    if needs_embeddings:
        tokenizer, model = load_model()
        print("model loaded")

    results = f"Evaluation date and time for file {path}: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n\n"
    results += f"Attributes evaluated: {', '.join(attributes)}\n\n"

    final_df = None
    for attr in attributes:
        score_df = None
        # Count the number of lines in gold havning the attribute
        col = next((c for c in gold_df.columns if c.lower() == attr.lower()), None)
        gold_count = gold_df[col].notna().sum()

        results += f"\nEvaluation for {attr} on {gold_count} instances\n"
        if attr in ['residence', 'origin', 'physical_health', 'occupation']:
            score_df = compare_gold_predicted(gold_df, predicted, tokenizer, model, [attr])
            mean_cosine_similarity = mean_cos(score_df, [attr])
            
            results += f"Mean cosine similarity for {attr}: {mean_cosine_similarity}\n"

            # Compute BERTScore
            filtered = score_df[score_df[f'{attr}_gold'].notna() & score_df[f'{attr}_pred'].notna()]
            gold_texts = filtered[f'{attr}_gold'].astype(str).tolist()
            pred_texts = filtered[f'{attr}_pred'].astype(str).tolist()

            if gold_texts and pred_texts:
                mean_bert_f1, bert_f1_list = compute_bertscore(gold_texts, pred_texts, lang='en')
                results += f"BERTScore F1 for {attr}: {mean_bert_f1:.4f}\n"
                filtered[f'bertscore_f1_{attr}'] = bert_f1_list

                merge_on = ['name']
                if 'book_title' in score_df.columns:
                    merge_on.append('book_title')
                else:
                    merge_on.append('book')

                columns_to_merge = merge_on + [f'bertscore_f1_{attr}']
                # Merge BERTScore back into score_df
                score_df = pd.merge(score_df, filtered[columns_to_merge], on=merge_on, how='left')

        elif attr in ['spoken_languages']:
            f1_score, score_df = f1_list(gold_df, predicted, label_pred=attr, label_gold=attr)
            results += f"Classification report for {attr}:\n {f1_score}\n"
        elif attr in ['age', 'gender']:
            f1_score, score_df = compute_f1_score(gold_df, predicted, attribute=attr)
            results += f"Classification report for {attr}:\n {f1_score}\n"
            if attr == 'age':
                weight_matrix = {
                    "child" : {"child": 1.0, "teenager": 0.8, "adult": 0.0, "senior": 0.0},
                    "teenager": {"child": 0.8, "teenager": 1.0, "adult": 0.8, "senior": 0.0},
                    "adult": {"teenager": 0.8, "adult": 1.0, "senior": 0.8, "child": 0.0},
                    "senior": {"adult": 0.8, "senior": 1.0, "teenager": 0.0, "child": 0.0}
                    
                    }
                f1_score_soft, soft_score_df = compute_f1_score_soft(gold_df, predicted, attribute=attr, weight_matrix=weight_matrix)
                results += f"Soft F1 score for {attr}: {f1_score_soft}\n"

                # merge soft_score_df with score_df by adding to score_df only the columns that are not already in score_df
                cols_to_add = [col for col in soft_score_df.columns if col not in score_df.columns or col in ['name', 'book_title']]
                merge_on = ['name']
                if 'book_title' in score_df.columns:
                    merge_on.append('book_title')
                else:
                    merge_on.append('book')

                score_df = pd.merge(score_df, soft_score_df[cols_to_add], on=merge_on, how='outer')

        elif attr == 'type':
            print(predicted['type'])
            # if type contains some string (not all) / older version, not used anymore
            if predicted['type'].apply(lambda x: isinstance(x, str)).any():
                results += (
                    "Detected legacy 'type' format as string. "
                    "This evaluation branch is deprecated and disabled. "
                    "Current pipeline expects 'type' as a dictionary with "
                    "{'is_human': bool, 'character_type': str}.\n"
                )
                raise RuntimeError(
                    "Legacy 'type' string format detected. "
                    "Please regenerate predictions using the current schema."
                )
                
            # if type contains dictionnary
            elif predicted['type'].apply(lambda x: isinstance(x, dict)).any():
                results += "Detected 'type' as a dictionary with 'is_human' and 'character_type' attributes.\n"
                # new case: type is a dictionnary with {"is_human": bool, "character_type": str}
                # extract both attr
                predicted['is_human'] = predicted['type'].apply(lambda x: x.get('is_human') if isinstance(x, dict) else None)
                predicted['character_type'] = predicted['type'].apply(lambda x: x.get('character_type') if isinstance(x, dict) else None)
                gold_df["is_human"] = gold_df["Type"].str.lower() == "human"

                total_count = predicted['is_human'].notna().sum()
                count_ones = predicted['is_human'].sum()  # True = humain
                percentage_ones = (count_ones / total_count) * 100 if total_count > 0 else 0
                results += f"Percentage of predicted humans: {percentage_ones:.2f}%\n"

                # compute f1
                f1_score, score_df = compute_f1_score(gold_df, predicted, attribute='is_human')
                results += f"Classification report for is_human:\n {f1_score}\n"

                # Prepare the data by renaming columns
                temp_pred = predicted.drop(columns=['type'], errors='ignore').rename(columns={'character_type': 'type'})

                predicted['cos_sim_type'] = np.nan  

                # Compute the score for all characters
                score_df_all = compare_gold_predicted(gold_df, temp_pred, tokenizer, model, ['type'])

                # Keep only the cases where gold != "human" and prediction == False
                score_df_nonhumans = score_df_all[
                    (score_df_all['type_gold'].str.lower() != 'human') &
                    (score_df_all['is_human_pred'] == 0)
                ]

                # Merge the cosine similarity scores back into the predicted DataFrame
                merge_cols = []
                if 'name' in predicted.columns:
                    merge_cols.append('name')
                elif 'character' in predicted.columns:
                    merge_cols.append('character')

                if 'book_title' in predicted.columns and 'book_title' in score_df_nonhumans.columns:
                    merge_cols.append('book_title')
                elif 'book' in predicted.columns and 'book' in score_df_nonhumans.columns:
                    merge_cols.append('book')


                predicted = predicted.merge(
                    score_df_nonhumans[merge_cols + ['cos_sim_type']],
                    on=merge_cols,
                    how='left',
                    suffixes=('', '_new')
                )

                predicted['cos_sim_type'] = predicted['cos_sim_type_new'].combine_first(predicted['cos_sim_type'])
                predicted = predicted.drop(columns=['cos_sim_type_new'])
                mean_cosine_similarity = mean_cos(score_df_nonhumans, ['type'])
                results += f"Mean cosine similarity for character_type (non-humans): {mean_cosine_similarity}\n"

                score_df = predicted.copy()
            else:
                # Print the types of data in the 'type' column
                unique_types = predicted['type'].apply(type).unique()
                print(f"Unexpected data types in 'type' column: {unique_types}")
                # print all predicted['type'] values
                print("Predicted 'type' values:", predicted['type'].unique())

        all_dfs.append(score_df)
    book_col = "book_title" if "book_title" in all_dfs[0].columns else "book"
    final_df = pd.concat(all_dfs, ignore_index=True)
    final_df = final_df.groupby([book_col, "name"], as_index=False).first()

    #Save final_df
    final_df.to_csv(f"../Data/evaluation/dataframes/{datatype}_final_scores_{model_used}_{rag}_{'_'.join(attributes)}_{datetime.now().strftime('%Y-%m-%d_%H-%M-%S')}.csv", index=False)

    results += f"\nEvaluation completed at {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n"
    print("Evaluation completed")


    
    
    with open(f"../Data/evaluation/{datatype}_evaluation_results.txt", "a") as f:
        f.write(results)


if __name__ == "__main__":
    print("Calling main() \n", flush = True)
    #check if the correct number of args is given
    if len(sys.argv) < 4:
        print("Usage: python script.py <attr1,attr2,...> <rag_strategy> <model_name>")
        sys.exit(1)

    # 1st argument is the list of attributes to be retrieved
    attributes = sys.argv[1].split(",")

    # 2nd arg is the method of excerpts selection (ex: rag ou rag_e5)
    rag = sys.argv[2]

    # 3rd arg is the datatype, usually raw or cleaned
    datatype = sys.argv[3]

    current_time = sys.argv[4]

    model = sys.argv[5]

    main(datatype, rag, attributes, current_time, model)