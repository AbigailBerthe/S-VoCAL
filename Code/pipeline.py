import requests
import json
import re
import pandas as pd
import time
import os
import unicodedata
from sentence_transformers import SentenceTransformer
import numpy as np
import sys
from datetime import datetime

def get_detailed_instruct(task_description: str, query: str) -> str:
    """
    Constructs a detailed instruction prompt.
    Args:
        task_description (str): Description of the task to be performed.
        query (str): The specific query to be addressed."""
    
    return f'Instruct: {task_description}\nQuery: {query}'

def send_request_ollama(model, prompt):
    """Send a request to the Ollama API and return the response content.
    Args:
        model (str): The model name to use.
        prompt (str): The prompt to send to the model."""
    
    # Define the URL for Ollama API
    IP = "localhost"
    url = f"http://{IP}:11434/api/chat"

    # Prompt
    messages = [
        {"role": "user", "content": prompt}
    ]

    # Ollama http request
    response = requests.post(
        url,
        json={
            "model": model,
            "messages": messages,
            "options": {
                "num_ctx": 40000,  # context length
            }
        },
        stream = False,
        headers={"Content-Type": "application/json"}
    )
   
   
    full_content = ""

    # Check the response status code
    if response.status_code == 200:
        lines = response.text.strip().split('\n')


        for line in lines:
            try:
                obj = json.loads(line)
                if "message" in obj and "content" in obj["message"]:
                    full_content += obj["message"]["content"]
            except json.JSONDecodeError:
                continue

    else:
        return f"Error {response.status_code}: {response.text}"
    return full_content
    

def remove_accents(text: str) -> str:
    """Remove accents from a given text."""

    return ''.join(
        c for c in unicodedata.normalize('NFD', text)
        if unicodedata.category(c) != 'Mn'
    )

def extract_character_info(book_text, character_name, characters_aliases=None, window_size=200):
    """
    Detect all mentions of a character and extract a window of tokens around each mention.

    Args:
        book_text (str): The full text of the book.
        character_name (str): The name of the character to search for.
        character_aliases (str) : Optional list of aliases for the character.
        window_size (int): The number of tokens (words) to include in the window around each mention.

    Returns:
        list: A list of text snippets containing the character mentions and their surrounding context.
    """
    if characters_aliases is None:
        characters_aliases = []

    # Normalisation of the Name and aliases (token-wise for exact alias matching)
    all_names = [character_name] + characters_aliases
    all_names_tokens = [
        remove_accents(name.lower()).split()
        for name in all_names
        if name and name.strip()
    ]

    # Split book into tokens (+ a normalized version)
    words = book_text.split()
    words_norm = [remove_accents(w.lower()) for w in words]

    # Define window parameters
    half = max(1, window_size // 2)
    snippets = []

    # Scan tokens and check if any alias matches starting at position i
    i = 0
    n = len(words_norm)
    while i < n:
        matched_len = 0
        for alias_tok in all_names_tokens:
            L = len(alias_tok)
            if L == 0 or i + L > n:
                continue
            # exact token sequence match (avoids substring false positives)
            if words_norm[i:i+L] == alias_tok:
                matched_len = L
                break

        if matched_len > 0:
            start = max(0, i - half)
            end = min(n, i + matched_len + half)
            window_text = " ".join(words[start:end])
            snippets.append(window_text)
            # advance at least by 1 (or skip past this alias)
            i += matched_len if matched_len > 0 else 1
        else:
            i += 1

    return snippets

def load_book_from_gutemberg(url, max_retries = 5, wait_seconds=3):
    """Loads a book from the Gutemberg Project by its ID."""
    for attempt in range(1, max_retries + 1):
        try:
            print(f"Attempt {attempt} - Downloading {url}")
            response = requests.get(url, timeout=10)

            if response.status_code == 200:
                return response.text
            else:
                print(f"HTTP error: {response.status_code} - Retrying...")

        except requests.exceptions.RequestException as e:
            print(f"Connection error: {e} - Retrying in {wait_seconds}s...")

        time.sleep(wait_seconds)

    return f"Error: Failed to retrieve book from {url} after {max_retries} attempts."



def create_query(attributes, book_text, character_name):
    queries = {
        "age": '"child" (before 12), "teenager" (before 18), "adult" (before 60), "senior" (after 60)',
        "gender": '"male" or "female"',
        "origin": '["continent", "country", "state", "city", "any other available location information"]',
        "native_language": '"language"',
        "residence": '["continent", "country", "state", "city", "any other available location information"]',
        "spoken_languages": '["language1", "language2", ...]',
        #"type": '"type of character (e.g., human, rat, troll, etc.)"',
        "type": '{"is_human": true/false, "character_type": "type (species) if not human, else empty string"}',
        "occupation": '["occupation1", "occupation2", ...]',
        "physical_health": '"condition (e.g., good, ill, injured)"'
    }

    readable_attributes = ' and '.join(attr.replace("_", " ") for attr in attributes)
    json_example = ',\n    '.join(f'"{attr}": {queries[attr]}' for attr in attributes)

    query = (
        f"You are a literary analyst, working on the book excerpts about the character {character_name} "
        f"provided below. Your task is to infer the {readable_attributes} of the character named "
        f"'{character_name}' based solely on the following book excerpts, separated by newlines:\n\n"
        f"{book_text}\n\n"
        f"Please provide the {readable_attributes} for the character '{character_name}' in the following format, "
        f"without any additional text or explanations. If any value is not inferrable from the excerpts, return null for that field.\n\n"
        f"Format:\n```json\n{{\n    {json_example}\n}}\n```\n"
        f"Do not add any explanation or extra text."
    )

    return query




def e5_selection(mentions, model, attributes, character_name, top_k=10):
    task = 'Given a query, retrieve relevant passages from which the answer to the query could be inferred'
    queries = []

    for attribute in attributes:
        if attribute == "age":
            query_age = get_detailed_instruct(task, f'Is {character_name} a child, a teenager, an adult or a senior?')
            queries.append(query_age)

        elif attribute == "gender":
            query_gender = get_detailed_instruct(task, f'Is {character_name} a male or a female?')
            queries.append(query_gender)

        elif attribute == "origin":
            query_origin = get_detailed_instruct(task, f'Where is {character_name} from?')
            queries.append(query_origin)

        elif attribute == "native_language":
            query_native_language = get_detailed_instruct(task, f'What is {character_name}\'s native language?')
            queries.append(query_native_language)

        elif attribute == "residence":
            query_residence = get_detailed_instruct(task, f'Where does {character_name} live?')
            queries.append(query_residence)

        elif attribute == "spoken_languages":
            query_spoken_languages = get_detailed_instruct(task, f'What languages does {character_name} speak?')
            queries.append(query_spoken_languages)

        elif attribute == "type":
            query_type = get_detailed_instruct(task, f'Where type of entity is {character_name} ?')
            queries.append(query_type)

        elif attribute == "occupation":
            query_occupation = get_detailed_instruct(task, f'What\'s {character_name}\'s occupation?')
            queries.append(query_occupation)

        elif attribute == "physical_health":
            query_health = get_detailed_instruct(task, f'How is {character_name}\'s health condition?')
            queries.append(query_health)

        else:
            print(f"Unknown attribute: {attribute} -- skipping.")

    input_texts = queries + mentions

    embeddings = model.encode(input_texts, convert_to_tensor=True, normalize_embeddings=True)
    num_queries = len(attributes)
    scores = (embeddings[:num_queries] @ embeddings[num_queries:].T) * 100

    scores = np.array(scores.tolist())
    top_indices = []
    for i in range(len(attributes)):  # une ligne par requête
        top_i = np.argsort(scores[i])[-top_k:][::-1]
        top_indices.extend(top_i)

    # Remove doubles
    top_indices = list(dict.fromkeys(top_indices))

    top_passages = [mentions[i] for i in top_indices]

    return top_passages

def main(attributes, rag, model):
    """
    attributes is the list of attributes to be retrieved
    rag is the type of RAG to be used (e5 retriever or all_mentions)
    model is the model to be used for inference
    """

    print(f"Attributes requested: {attributes}", flush=True)
    print(f"RAG strategy: {rag}", flush=True)
    print(f"Model: {model}", flush=True)
    n = 1
    start = time.time()

    if(rag == "e5"):
        # load the retrieval model
        model_retrieval = SentenceTransformer('intfloat/multilingual-e5-large-instruct')


    # Load jsonl file with gold character data
    df = pd.read_json("../Data/S-VoCAL_dataset.jsonl", lines=True)

    results = []
    output_counter = 0
    current_time = datetime.now().strftime('%Y-%m-%d_%H-%M-%S') 
    # Define output file name using the current time to distinguish runs
    output_file = f"../Data/raw/raw_{model}_rag_{rag}_{'_'.join(attributes)}_{current_time}.csv"
    print("OUTPUT_TIME:", current_time)

    for index, row in df.iterrows():
        print(f"character {n}", flush = True)
        n +=1
        book_content_link = row['gutenberg_content']
        book = load_book_from_gutemberg(book_content_link)

        if isinstance(book, str) and book.startswith("Error"):
            continue  # Ignore if the book could not be loaded

        character_name = row['Name']
        character_aliases = [alias.strip() for alias in row['Aliases'].split(',')] if pd.notna(row['Aliases']) else []

        # Extract mentions
        mentions = extract_character_info(book, character_name, character_aliases)

        # Select top passages based on RAG strategy
        if rag == "e5":
            top_passages = e5_selection(mentions, model_retrieval, attributes, character_name)
        if rag == "all_mentions":
            top_passages = mentions

        excerpts = "\n\n ".join(top_passages)

        if excerpts.strip() != "":
            query = create_query(attributes, excerpts, character_name)
            output = send_request_ollama(model, query) #qwen3:latest
            if n==3:
                #print the last 300 characters
                print(query[-300:], flush=True)
        else:
            output = None
        results.append({
            "Book": book_content_link,
            "Character": character_name,
            "Output": output,
            "Mentions_counter": len(mentions),
            "Selected_passages": top_passages
        })

        output_counter += 1

        # Save intermediate results every 10 characters
        if output_counter % 10 == 0:
            output_df = pd.DataFrame(results)
            if output_counter == 10 and not os.path.exists(output_file):
                output_df.to_csv(output_file, sep=";", encoding="utf-8", index=False, mode='w')
            else:
                output_df.to_csv(output_file, sep=";", encoding="utf-8", index=False, mode='a', header=False)
            results = []

    if results:
        output_df = pd.DataFrame(results)
        if output_counter <= 10 and not os.path.exists(output_file):
            output_df.to_csv(output_file, sep=";", encoding="utf-8", index=False, mode='w')
        else:
            output_df.to_csv(output_file, sep=";", encoding="utf-8", index=False, mode='a', header=False)

    end = time.time()
    print(f"Runtime: {end-start:.2f} sec")



print("Script started \n", flush=True)
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

    # 3rd arg is the models name (ex: qwen3:latest)
    model_name = sys.argv[3]

    main(attributes, rag, model_name)

