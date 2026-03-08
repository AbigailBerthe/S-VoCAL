# Requires transformers>=4.51.0
import torch
import torch.nn.functional as F
from torch import Tensor
from transformers import AutoTokenizer, AutoModel

import pandas as pd

# Attribute-specific instructions used to condition the embedding space for open-class attribute evaluation.
ATTRIBUTE_TASKS = {
    'origin': "Identify the character's origin.",
    'residence': "Identify the character's residence.",
    'occupation': "Identify the character's occupation.",
    'physical_health': "Identify the character's physical health.",
    'spoken_languages': "Identify the languages spoken by the character.",
    'type': "Identify the type of character.",
    'native_language': "Identify the character's native language."
}

# Attribute-specific query prefixes (left empty in our setup, the query content consists of the attribute values themselves)
ATTRIBUTE_QUERY = {
    'origin': " ",
    'spoken_languages': " ",
    'occupation': " ",
    'type': " ",
    'physical_health': " ",
    'residence': " ",
    'native_language' : " "
}

def last_token_pool(last_hidden_states: Tensor,
                 attention_mask: Tensor) -> Tensor:
    left_padding = (attention_mask[:, -1].sum() == attention_mask.shape[0])
    if left_padding:
        return last_hidden_states[:, -1]
    else:
        sequence_lengths = attention_mask.sum(dim=1) - 1
        batch_size = last_hidden_states.shape[0]
        return last_hidden_states[torch.arange(batch_size, device=last_hidden_states.device), sequence_lengths]


def get_detailed_instruct(task_description: str, query: str) -> str:
    return f'Instruct: {task_description}\nQuery:{query}'


def load_model(model_name = 'Qwen/Qwen3-Embedding-8B'):
    tokenizer = AutoTokenizer.from_pretrained(model_name, padding_side='left')
    model = AutoModel.from_pretrained('Qwen/Qwen3-Embedding-8B')
    print(f"Model device: {model.device}")


    max_length = 8192
    tokenizer.model_max_length = max_length

    return tokenizer, model

def create_prompt(attribute = 'spoken_languages', gold = ["English", "French"], pred = ["French"]):
    """
    Build instruction-conditioned prompts for gold and predicted attribute values.
    """
    # Same instruction, different query content (gold vs. predicted values)
    task = ATTRIBUTE_TASKS[attribute]
    query = ATTRIBUTE_QUERY[attribute]

    if gold is None:
        gold = []

    gold_text = get_detailed_instruct(task, query + ", ".join(gold))
    pred_text = get_detailed_instruct(task, query + ", ".join(pred))

    return gold_text, pred_text

def compute_cosine_similarity(gold_text, pred_text, tokenizer, model):
    input_texts = [gold_text, pred_text]

    batch = tokenizer(input_texts, padding=True, return_tensors="pt", truncation=True).to(model.device)
    outputs = model(**batch)
    embs = last_token_pool(outputs.last_hidden_state, batch['attention_mask'])
    # L2-normalization enables cosine similarity via dot product
    embs = F.normalize(embs, p=2, dim=1)

    score = (embs[0] @ embs[1].T).item()
    return score

def compare_gold_predicted(gold_df, predicted_df, tokenizer, model, attributes):
    """
    Align gold and predicted data and compute cosine similarities using instruction-conditioned embeddings.
    """

    # Lower all column names
    # Normalize column names to avoid casing mismatches
    gold_df.columns = gold_df.columns.str.lower()
    predicted_df.columns = predicted_df.columns.str.lower()

    # One row per character-book pair
    gold_df = gold_df.drop_duplicates(subset=['name', 'gutenberg_url'])
    predicted_df = predicted_df.drop_duplicates(subset=['character', 'book'])

    merged_df = pd.merge(
        gold_df,
        predicted_df,
        left_on=['name', 'gutenberg_url'],
        right_on=['character', 'book'],
        how='inner',
        suffixes=('_gold', '_pred')
    )

    for attribute in attributes:
        # Initialize similarity column for this attribute
        merged_df['cos_sim_'+ attribute] = 0.0
        for index, row in merged_df.iterrows():
            # convert gold to list if it is a string, separated with commas
            if isinstance(row[f'{attribute}_gold'], str):
                row[f'{attribute}_gold'] = row[f'{attribute}_gold'].split(', ')
            if isinstance(row[f'{attribute}_pred'], str):
                row[f'{attribute}_pred'] = row[f'{attribute}_pred'].split(', ')
            if(type(row[f'{attribute}_gold']) != float and type(row[f'{attribute}_pred']) == list):
                gold_text, pred_text = create_prompt(attribute, row[f'{attribute}_gold'], row[f'{attribute}_pred'])
                # Compute semantic similarity between gold and predicted values
                score = compute_cosine_similarity(gold_text, pred_text, tokenizer, model)
                merged_df.at[index, 'cos_sim_' + attribute] = score
                #print(score)
    
    return merged_df

def mean_cos(merged_df, attributes):
    """
    Compute mean cosine similarity per attribute, ignoring missing gold values.
    """
    mean_scores = {}
    for attribute in attributes:
        # only compute mean if 'gold_' + attribute is notna
        merged_df_notna = merged_df[merged_df[f'{attribute}_gold'].notna()]
        mean_score = merged_df_notna['cos_sim_' + attribute].mean()
        mean_scores[attribute] = mean_score
        print(f"Mean cosine similarity for {attribute}: {mean_score:.4f}")
    
    return mean_scores


        


