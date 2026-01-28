from sklearn.metrics import confusion_matrix, classification_report, ConfusionMatrixDisplay
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.metrics import f1_score
from sklearn.preprocessing import MultiLabelBinarizer
from cleaner import split_list
import numpy as np
from bert_score import score
import ast

def merge_dataframes(gold_df, predicted_df):
    """
    Merge the gold and predicted dataframes on the character name.
    """

    gold_df.columns = gold_df.columns.str.lower()
    predicted_df.columns = predicted_df.columns.str.lower()

    merged_df = pd.merge(
        gold_df,
        predicted_df,
        left_on=['name', 'gutenberg_content'],
        right_on=['character', 'book'],
        how='inner',
        suffixes=('_gold', '_pred')
    )

    return merged_df

def binary_detection(gold_df, predicted_df, attribute, value):
    """
    Returns the predicted dataframe with a new binary column (e.g. 'type_binary') indicating:
    - 1 if both gold and predicted agree on the presence or absence of 'value'
    - 0 otherwise

    Previously used for evaluating 'type = human' in experiments where the LLM did not explicitly
    state whether the character was human before providing the detailed type. The function checked
    agreement between gold and prediction regarding 'humanness'.
    It is no longer used in the current pipeline, since the LLM is now instructed to extract both
    (a) whether the character is human, and (b) the precise type if not human.
    """

    gold_col = f'{attribute}_gold'
    pred_col = f'{attribute}_pred'
    binary_col = f'{attribute}_binary'

    # Merge the dataframes
    merged_df = merge_dataframes(gold_df, predicted_df)

    # Check the expected columns are present
    for col in ['character', 'book', gold_col, pred_col]:
        if col not in merged_df.columns:
            raise ValueError(f"Missing column in merged_df : {col}")

    # Detete lines without gold
    filtered_df = merged_df[merged_df[gold_col].notna()].copy()
    print(f"Number of filtered lines for {attribute}: {len(filtered_df)}")

    # Apply binary logic
    gold_is_value = filtered_df[gold_col].str.lower() == value.lower()
    pred_is_value = filtered_df[pred_col].str.lower() == value.lower()

    filtered_df[binary_col] = (gold_is_value == pred_is_value).astype(int)

    # Make sure book and character columns are present
    if not {'character', 'book'}.issubset(filtered_df.columns):
        filtered_df.reset_index(inplace=True)

    # Merge witht he initial predicted df to get binary column
    predicted_df = predicted_df.merge(
        filtered_df[['character', 'book', binary_col]],
        on=['character', 'book'],
        how='left'
    )

    return predicted_df

def compute_f1_score(gold_df, predicted_df, attribute):
    """
    Compute F1-score for a single attribute (e.g. 'gender', 'type', ...).

    - Merges gold and predicted dataframes on ['character', 'book'].
    - Filters out rows where either gold or predicted is missing.
    - Normalises string labels to lowercase (for string attributes).
    - Handles boolean-like attributes ('true'/'false') as bools.
    - Adds a per-row exact-match score column: f'score_{attribute}'.
    
    Returns
    -------
    report : dict
        sklearn classification_report as a dict.
    merged_df : pd.DataFrame
        Global merged dataframe with an extra score column.
    """

    merged_df = merge_dataframes(gold_df, predicted_df)

    gold_col = f'{attribute}_gold'
    pred_col = f'{attribute}_pred'

    # Filter lines where both gold and pred are provided
    filtered_df = merged_df[merged_df[gold_col].notna() & merged_df[pred_col].notna()].copy()
    print(f"Number of filtered lines for {attribute}: {len(filtered_df)}", flush=True)

    if filtered_df.empty:
        print(f"No gold for {attribute}, F1 score ignored.")
        return None, merged_df

    print(f"Unique values in {gold_col}: {filtered_df[gold_col].unique()}", flush=True)
    print(f"Unique values in {pred_col}: {filtered_df[pred_col].unique()}", flush=True)

    # Detect if we are dealing with strings or boolean val
    is_string = filtered_df[pred_col].apply(lambda x: isinstance(x, str)).any()

    if is_string:
        # Normalise strings to lowercase for comparison
        filtered_df.loc[:, gold_col] = filtered_df[gold_col].astype(str).str.lower()
        filtered_df.loc[:, pred_col] = filtered_df[pred_col].astype(str).str.lower()
        gold_labels = filtered_df[gold_col]
        predicted_labels = filtered_df[pred_col]
    else:
        # Map 'true'/'false' to boolean
        gold_labels = (
            filtered_df[gold_col]
            .astype(str).str.strip().str.lower()
            .map({'true': True, 'false': False})
        )
        predicted_labels = (
            filtered_df[pred_col]
            .astype(str).str.strip().str.lower()
            .map({'true': True, 'false': False})
        )

    # Exact-match score per row
    filtered_df[f'score_{attribute}'] = (filtered_df[gold_col] == filtered_df[pred_col]).astype(float)

    # Merge the per-row score back on (character, book)
    merged_df = merged_df.merge(
        filtered_df[['character', 'book', f'score_{attribute}']],
        on=['character', 'book'],
        how='left'
    )

    # Classification report
    report = classification_report(gold_labels, predicted_labels, output_dict=True, zero_division=0)

    return report, merged_df

def compute_f1_score_soft(gold_df, predicted_df, attribute, weight_matrix):
    merged_df = merge_dataframes(gold_df, predicted_df)

    gold_col = f'{attribute}_gold'
    pred_col = f'{attribute}_pred'

    # Filter lines where both gold and prediction are present
    filtered_df = merged_df[merged_df[gold_col].notna() & merged_df[pred_col].notna()].copy()
    print(f"Lines filtered for {attribute}: {len(filtered_df)}")

    # Normalize case
    filtered_df[gold_col] = filtered_df[gold_col].str.strip().str.lower()
    filtered_df[pred_col] = filtered_df[pred_col].str.strip().str.lower()

    if filtered_df.empty:
        print(f"No gold pred for {attribute}.")
        return None, merged_df
    
    # Apply soft scoring
    tp_weighted = 0.0
    total_gold = {}
    total_pred = {}

    for idx, row in filtered_df.iterrows():
        g = row[gold_col]
        p = row[pred_col]
        weight = weight_matrix.get(g, {}).get(p, 0.0)
        tp_weighted += weight
        total_gold[g] = total_gold.get(g, 0) + 1
        total_pred[p] = total_pred.get(p, 0) + 1

    total_gold_count = sum(total_gold.values())
    total_pred_count = sum(total_pred.values())

    precision = tp_weighted / total_pred_count if total_pred_count > 0 else 0.0
    recall = tp_weighted / total_gold_count if total_gold_count > 0 else 0.0
    f1 = (2 * precision * recall) / (precision + recall) if precision + recall > 0 else 0.0

    # Save score in filtered_df
    filtered_df[f'soft_score_{attribute}'] = [
        weight_matrix.get(g, {}).get(p, 0.0)
        for g, p in zip(filtered_df[gold_col], filtered_df[pred_col])
    ]

    # Merge scores back to merged_df 
    merged_df = merged_df.merge(
    filtered_df[['character', 'book', f'soft_score_{attribute}']],
    on=['character', 'book'],
    how='left'
    )



    return f1, merged_df

def compute_bertscore(gold_texts, pred_texts, lang='en'):
    """
    Compute mean BERTScore F1 between gold and predicted texts.
    Returns: mean F1 score, and list of per-line scores if needed
    """
    P, R, F1 = score(pred_texts, gold_texts, lang=lang, verbose=False)
    return F1.mean().item(), F1  # means and scores line by line



def f1_list(gold_df, predicted_df, label_pred='spoken_languages', label_gold='spoken_languages'):
    """
    Compute a multilabel F1 score for an attribute represented as a list 
    (e.g., spoken_languages). The function:
      - normalizes gold and predicted list formats,
      - merges entries by character and book,
      - binarizes lists with MultiLabelBinarizer,
      - computes per-instance F1 scores,
      - and returns the global micro-averaged F1 score.
    """

    gold_df = gold_df.copy()
    predicted_df = predicted_df.copy()

    gold_df.columns = gold_df.columns.str.lower()
    predicted_df.columns = predicted_df.columns.str.lower()

    #print(gold_df[label_gold])
    gold_df[f'{label_pred}_gold'] = gold_df[label_gold].apply(split_list)

    # Clean predictions
    predicted_df[f'{label_pred}_pred'] = predicted_df[label_pred].apply(
    lambda x: x if isinstance(x, list)
    else ast.literal_eval(x) if isinstance(x, str) and x.startswith('[')
    else []
    )

    # Fusion by pair character book
    merged = pd.merge(
        predicted_df,
        gold_df[['name', 'gutenberg_content', f'{label_pred}_gold']],
        left_on=['character', 'book'],
        right_on=['name', 'gutenberg_content'],
        how='inner'
    )

    print(merged[f'{label_pred}_pred'].head())

    # Binarization and F1
    mlb = MultiLabelBinarizer()

    all_langs_gold = merged[f'{label_pred}_gold'].tolist()
    all_langs_pred = merged[f'{label_pred}_pred'].tolist()
    all_langs_union = all_langs_gold + all_langs_pred

    mlb.fit(all_langs_union)

    y_true = mlb.transform(all_langs_gold)
    y_pred = mlb.transform(all_langs_pred)

    individual_scores = []
    for true, pred in zip(y_true, y_pred):
        if true.sum() == 0 :
            individual_scores.append(np.nan)
        else:
            score = f1_score(true, pred, zero_division=0)
            individual_scores.append(score)

    merged[f'score_{label_pred}'] = individual_scores

    # Computation of micro F1 score with only gold provided
    filtered_y_true = [y for y in y_true if y.sum() > 0]
    filtered_y_pred = [p for y, p in zip(y_true, y_pred) if y.sum() > 0]
    f1 = f1_score(filtered_y_true, filtered_y_pred, average='micro') if filtered_y_true else np.nan
    
    return f1 , merged
