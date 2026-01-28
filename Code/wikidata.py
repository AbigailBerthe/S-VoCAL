import requests
import pandas as pd
import time
from collections import Counter
import json

"""
This script contains all steps used to build the initial S-VoCAL character dataset
from Wikidata and Project Gutenberg.

Important note:
- Some functions in this file were used for exploratory analysis only
  (e.g. inspecting available characters or attribute coverage in Wikidata).
- These exploratory functions are NOT part of the final dataset construction pipeline.
- The final dataset produced by this script corresponds to the dataset before manual curation and annotation.

The effective dataset creation pipeline relies on the following steps:
1) Retrieval of books and characters associated with Project Gutenberg
2) Extraction of character-level attributes from Wikidata
3) Aggregation and flattening into a tabular dataset
"""


def get_gutenberg_books():
    # URL of Sparql access point
    url = "https://query.wikidata.org/sparql"

    # Sparql request with author
    query = """
    SELECT DISTINCT ?book ?bookLabel ?authorLabel ?date WHERE {
    ?book wdt:P7937 wd:Q8261 .           # Genre = novel
    ?book wdt:P674 ?character .          # Book with at least 1 character
    ?book wdt:P577 ?date .               # Release date
    ?book wdt:P50 ?author .              # Author
    FILTER(?date < "1940-01-01T00:00:00Z"^^xsd:dateTime)
    FILTER EXISTS { ?book wdt:P2034 ?gutenbergID }  # with Gutenberg ID

    SERVICE wikibase:label { bd:serviceParam wikibase:language "en" }
    }
    LIMIT 1500
    """

    # En-têtes
    headers = {
        "Accept": "application/sparql-results+json"
    }

    # Request
    response = requests.get(url, params={"query": query}, headers=headers)
    data = response.json()

    # Extract in a list of dict
    rows = []
    for result in data["results"]["bindings"]:
        rows.append({
            "title": result["bookLabel"]["value"],
            "author": result["authorLabel"]["value"],
            "publication_date": result["date"]["value"],
            "wikidata_uri": result["book"]["value"]
        })

    # dataframe creation
    df = pd.DataFrame(rows)

    # Print
    print(df.head())
    print(f"Total books found: {len(df)}")
    return df


def get_characters_for_book(book_uri, book_title, author_name):
    """
    Get characters for a book given its Wikidata URI.
    :param book_uri: URI of the book in Wikidata
    :param book_title: Title of the book (for logging)
    :param author_name: Name of the author (for logging)
    :return: List of characters with their names and URIs
    """

    query = f"""
    SELECT ?character ?characterLabel WHERE {{
      <{book_uri}> wdt:P674 ?character .
      SERVICE wikibase:label {{ bd:serviceParam wikibase:language "en". }}
    }}
    """
    headers = {
        "Accept": "application/sparql-results+json"
    }
    url = "https://query.wikidata.org/sparql"
    response = requests.get(url, params={"query": query}, headers=headers)
    if response.status_code != 200:
        print(f"Erreur {response.status_code} pour {book_title}")
        return []

    results = response.json()["results"]["bindings"]
    return [
        {
            "character_name": char["characterLabel"]["value"],
            "character_uri": char["character"]["value"],
            "book_title": book_title,
            "book_author": author_name
        }
        for char in results
    ]


def get_properties_for_character(character_uri, max_retries=5):
    """
    Retrieve property IDs for a character from Wikidata.
    :param character_uri: URI of the character in Wikidata
    :param max_retries: Maximum number of retries for rate limiting
    :return: List of property IDs
    """

    query = f"""
    SELECT ?p WHERE {{
      <{character_uri}> ?p ?o .
      FILTER(STRSTARTS(STR(?p), "http://www.wikidata.org/prop/direct/"))
    }}
    """
    headers = {"Accept": "application/sparql-results+json"}
    url = "https://query.wikidata.org/sparql"

    retries = 0
    wait_time = 1.5

    while retries < max_retries:
        response = requests.get(url, params={"query": query}, headers=headers)
        if response.status_code == 200:
            results = response.json()["results"]["bindings"]
            return [r["p"]["value"].split("/")[-1] for r in results]  # ex: "P21", "P569"
        elif response.status_code == 429:
            # if rate limited, wait and retry (up to max_retries)
            print(f"[429] Too many requests for {character_uri}, wait {wait_time}s...")
            time.sleep(wait_time)
            retries += 1
            wait_time *= 2
        else:
            print(f"Error {response.status_code} for {character_uri}")
            return []

    print(f"No response after {max_retries} retries for {character_uri}")
    return []




def get_property_labels(property_ids):
    """
    Retrieve labels for a list of property IDs from Wikidata.
    :param property_ids: List of property IDs (e.g., ["P21", "P569"])
    :return: Dictionary mapping property IDs to their labels
    """

    labels = {}
    for pid in property_ids:
        query = f"""
            SELECT ?label WHERE {{
            wd:{pid} rdfs:label ?label .
            FILTER(LANG(?label) = "en")
            }}
        """
        url = "https://query.wikidata.org/sparql"
        headers = {"Accept": "application/sparql-results+json"}
        response = requests.get(url, params={"query": query}, headers=headers)
        if response.status_code == 200:
            results = response.json()["results"]["bindings"]
            if results:
                labels[pid] = results[0]["label"]["value"]
        time.sleep(0.2)
    return labels

def create_gutemberg_persos_and_properties():
    """
    Create a DataFrame of characters from books with Gutenberg IDs and count their properties.
    """
    df = get_gutenberg_books()

    characters_data = []

    # iterate on each book of the DataFrame
    for idx, row in df.iterrows():
        book_uri = row["wikidata_uri"]
        title = row["title"]
        author = row["author"]

        print(f"Fetching characters for: {title} by {author}")
        try:
            characters = get_characters_for_book(book_uri, title, author)
            characters_data.extend(characters)
        except Exception as e:
            print(f"Erreur avec {title}: {e}")
        
        time.sleep(2)  # respect Wikidata limits (important)

    # Create a new DataFrame with the characters
    characters_df = pd.DataFrame(characters_data)

    # Display a preview
    print(characters_df.head())
    print(f"Total characters found: {len(characters_df)}")
    # Save to CSV
    characters_df.to_csv("characters_gutembergID.csv", index=False, encoding="utf-8")
    # Count properties
    property_counter = Counter()
    # iterate on all characters of the DataFrame
    for idx, row in characters_df.iterrows():
        character_uri = row["character_uri"]
        print(f"Attributes for : {row['character_name']}")

        try:
            props = get_properties_for_character(character_uri)
            property_counter.update(props)
        except Exception as e:
            print(f"error pour {character_uri} : {e}")

        time.sleep(2)
    # Convert to DataFrame
    prop_df = pd.DataFrame(property_counter.items(), columns=["property_id", "count"]).sort_values(by="count", ascending=False)

    # Display most frequent properties
    print(prop_df.head(15))

    # Retrieve labels
    labels_dict = get_property_labels(prop_df["property_id"].tolist())

    # Add labels to df
    prop_df["label"] = prop_df["property_id"].map(labels_dict)

    # display enriched resuts
    print(prop_df)

    #Save dataframe into csv file
    prop_df.to_csv("properties_gutembergID.csv", index=False, encoding="utf-8")

    print(f"Total properties found: {len(prop_df)}")
    
def get_books_without_gutenberg_id():
    """
    Some books are available on Gutenberg but do not have entry for their Gutenberg ID in Wikidata.
    This function retrieves books published before 1940, meaning they are very likely to be in the public domain, without Gutenberg ID entry in wikidata.
    """


    query = """
    SELECT DISTINCT ?book ?bookLabel ?authorLabel ?date WHERE {
    ?book wdt:P7937 wd:Q8261 .           # Genre = novel
    ?book wdt:P674 ?character .          # Book with at least 1 character
    ?book wdt:P577 ?date .               # Release date
    ?book wdt:P50 ?author .              # Author
    FILTER(?date < "1940-01-01T00:00:00Z"^^xsd:dateTime)
    FILTER NOT EXISTS { ?book wdt:P2034 ?gutenbergID }  # Without gutenberg id

    SERVICE wikibase:label { bd:serviceParam wikibase:language "en" }
    }
    LIMIT 1500
    """

    url = "https://query.wikidata.org/sparql"

    headers = {
        "Accept": "application/sparql-results+json"
    }

    # Request
    response = requests.get(url, params={"query": query}, headers=headers)
    data = response.json()

    # Extract in a list of dict
    rows = []
    for result in data["results"]["bindings"]:
        rows.append({
            "title": result["bookLabel"]["value"],
            "author": result["authorLabel"]["value"],
            "publication_date": result["date"]["value"],
            "wikidata_uri": result["book"]["value"]
        })

    df = pd.DataFrame(rows)
    print(df.head())
    print(f"Total books found: {len(df)}")
    return df

def without_gutenberg_id():
    """
    Retrieve characters and their properties for books without Gutenberg IDs.
    """

    df = get_books_without_gutenberg_id() # Get the books

    # Only keep rows with unique title and author to avoid multiple requests for same book
    df = df.drop_duplicates(subset=["title", "author"])
    print(df.head())
    print(f"Total books without Gutenberg ID: {len(df)}")
    df.to_csv("books_without_gutenbergID.csv", index=False, encoding="utf-8")
    
    characters = []
    for idx, row in df.iterrows():
        characters.extend(get_characters_for_book(row["wikidata_uri"], row["title"], row["author"]))
        time.sleep(2)
    # to df
    characters_df = pd.DataFrame(characters)
    print(characters_df.head())
    print(f"Total characters found: {len(characters_df)}")

    characters_df.to_csv("characters_without_gutembergID.csv", index=False, encoding="utf-8")

    property_counter = Counter()
    # Iterate on all characters of the DataFrame
    for idx, row in characters_df.iterrows():
        character_uri = row["character_uri"]
        print(f"Attributes for : {row['character_name']}")

        try:
            props = get_properties_for_character(character_uri)
            property_counter.update(props)
        except Exception as e:
            print(f"Error for {character_uri} : {e}")

        time.sleep(2)
    # Convert into df
    prop_df = pd.DataFrame(property_counter.items(), columns=["property_id", "count"]).sort_values(by="count", ascending=False)

    # Display most frequent properties
    print(prop_df.head(15))

    # Retrieve labels
    labels_dict = get_property_labels(prop_df["property_id"].tolist())

    # Add labels to dataframe
    prop_df["label"] = prop_df["property_id"].map(labels_dict)

    # to csv
    prop_df.to_csv("properties_without_gutembergID.csv", index=False, encoding="utf-8")

    # Display enriched results
    print(prop_df.head(15))
    print(f"Total properties found: {len(prop_df)}")


def check_books_by_title_and_lastname(df, wait=2.0):
    """
    Check if books exist on Gutenberg by searching with title and author's last name
    """
    exists_flags = []

    for idx, row in df.iterrows():
        title = row['title'].strip()
        # On extrait le nom de famille (dernier mot)
        author = row.get('author', '')
        last_name = author.strip().split()[-1].lower() if isinstance(author, str) and author.strip() else ''
        url = f"https://gutendex.com/books/?search={requests.utils.quote(title)}"

        print(f"Recherche : \"{title}\" | Nom de famille : \"{last_name}\"")

        try:
            response = requests.get(url, timeout=10)
            if response.status_code == 200:
                data = response.json()
                exists = False

                for result in data.get('results', []):
                    result_title = result.get('title', '').strip().lower()
                    if title.lower() in result_title:
                        result_authors = [a['name'].lower() for a in result.get('authors', [])]
                        if any(last_name in a for a in result_authors):
                            exists = True
                            break
                exists_flags.append(exists)
            else:
                print(f"Erreur HTTP {response.status_code}")
                exists_flags.append(False)

        except Exception as e:
            print(f"Erreur : {e}")
            exists_flags.append(False)

        time.sleep(wait)

    df['exists_on_gutenberg'] = exists_flags
    return df

def check_gutenberg_without_id():
    """
    Regroup entire process of verification for books without Gutenberg ID
    """
    #load the CSV file
    df = pd.read_csv("books_without_gutenbergID.csv", encoding="utf-8")
    #check the books by title and last name
    df = check_books_by_title_and_lastname(df, wait=1.0)
    #save the result to a new CSV file
    df.to_csv("books_without_gutenbergID_checked.csv", index=False, encoding="utf-8")
    # print the number of books found
    print(f"Books found : {df['exists_on_gutenberg'].sum()} out of {len(df)}")
    # print the 5 first books found
    print(df[df['exists_on_gutenberg']].head())
    # print the 5 first books not found
    print(df[~df['exists_on_gutenberg']].head())

def check_attributes_characters_no_id():
    """
    Build Wikidata properties available for characters
    from books that do not have a Gutenberg ID in Wikidata but are
    confirmed to exist on Project Gutenberg.

    The function:
    - loads books found on Wikidata previously checked as existing on 
    Gutenberg without a Gutenberg ID available in Wikidata,
    - retrieves all associated characters from Wikidata,
    - counts which Wikidata properties are present across these characters,
    - outputs CSV files listing the characters and property frequency statistics.

    This function is used to explore the attributes available for characters, and is not intended
    for extracting full attribute values.
    """
    

    #Load the CSV file
    df = pd.read_csv("books_without_gutenbergID_checked.csv", encoding = "utf-8")

    # Only keep those with 'True' in exists_on_gutenberg
    df = df[df['exists_on_gutenberg'] == True]

    characters_data = []

    # Iterate on each book of the DataFrame
    for idx, row in df.iterrows():
        book_uri = row["wikidata_uri"]
        title = row["title"]
        author = row["author"]

        print(f"Fetching characters for: {title} by {author}")
        try:
            characters = get_characters_for_book(book_uri, title, author)
            characters_data.extend(characters)
        except Exception as e:
            print(f"Erreur avec {title}: {e}")
        
        time.sleep(2)  

    characters_df = pd.DataFrame(characters_data)

    print(characters_df.head())
    print(f"Total characters found: {len(characters_df)}")

    # Save to CSV
    characters_df.to_csv("characters_no_gutembergID.csv", index=False, encoding="utf-8")

    property_counter = Counter()

    for idx, row in characters_df.iterrows():
        character_uri = row["character_uri"]
        print(f"Attributes for : {row['character_name']}")

        try:
            props = get_properties_for_character(character_uri)
            property_counter.update(props)
        except Exception as e:
            print(f"Erreur pour {character_uri} : {e}")

        time.sleep(2)

    # convert into DataFrame
    prop_df = pd.DataFrame(property_counter.items(), columns=["property_id", "count"]).sort_values(by="count", ascending=False)

    print(prop_df.head(15))

    # Retrieve labels of properties
    labels_dict = get_property_labels(prop_df["property_id"].tolist())

    # Add labels to dataframe
    prop_df["label"] = prop_df["property_id"].map(labels_dict)

    print(prop_df)

    #Save Dataframe into CSV file
    prop_df.to_csv("properties_no_gutembergID.csv", index=False, encoding="utf-8")

    print(f"Total properties found: {len(prop_df)}")

def retrieve_characters_attributes(character_name, character_uri):
    """
    Retrieve attributes for a character from Wikidata.
    :param character_name: Name of the character
    :param character_uri: URI of the character in Wikidata
    """
    print(f"Retrieval for {character_name} ({character_uri})...")

    query = f"""
    SELECT ?p ?propLabel ?value ?valueLabel WHERE {{
      <{character_uri}> ?p ?value .
      ?property wikibase:directClaim ?p .
      SERVICE wikibase:label {{ bd:serviceParam wikibase:language "en". }}
      OPTIONAL {{ ?property rdfs:label ?propLabel FILTER(LANG(?propLabel) = "en") }}
      OPTIONAL {{ ?value rdfs:label ?valueLabel FILTER(LANG(?valueLabel) = "en") }}

    }}
    """

    url = "https://query.wikidata.org/sparql"
    headers = {"Accept": "application/sparql-results+json"}
    response = requests.get(url, params={"query": query}, headers=headers)

    if response.status_code != 200:
        print(f"Erreur {response.status_code} pour {character_name}")
        return {
            "character_name": character_name,
            "character_uri": character_uri,
            "description": None,
            "also_known_as": [],
            "attributes": {},
            "note": f"Erreur HTTP {response.status_code}"
        }

    results = response.json()["results"]["bindings"]

    attributes = {}
    description, alt_labels = get_basic_info(character_uri)

    for r in results:
        # -- Prop
        if "p" in r and "value" in r:
            p_uri = r["p"]["value"]
            p_id = p_uri.split("/")[-1]
            prop = r.get("propLabel", {}).get("value", p_id)

            val_uri = r["value"]["value"]
            val_id = val_uri.split("/")[-1] if val_uri.startswith("http://www.wikidata.org/entity/") else val_uri
            value = r.get("valueLabel", {}).get("value", val_id)

            # if P1559 : personal name, get also the language code
            if p_id == "P1559":
                lang_code = r["value"].get("xml:lang", None)
                value = {"text": value, "lang": lang_code}

            if prop in attributes:
                if isinstance(attributes[prop], list):
                    if value not in attributes[prop]:
                        attributes[prop].append(value)
                else:
                    if value != attributes[prop]:
                        attributes[prop] = [attributes[prop], value]
            else:
                attributes[prop] = value


    return {
        "character_name": character_name,
        "character_uri": character_uri,
        "description": description,
        "also_known_as": list(alt_labels),
        "attributes": attributes
    }


def get_basic_info(character_uri):
    """
    Retrieve basic info (description, alt labels) for a character from Wikidata.
    """
    query = f"""
    SELECT ?altLabel ?desc WHERE {{
      OPTIONAL {{ <{character_uri}> skos:altLabel ?altLabel FILTER(LANG(?altLabel) = "en") }}
      OPTIONAL {{ <{character_uri}> schema:description ?desc FILTER(LANG(?desc) = "en") }}
    }}
    """

    url = "https://query.wikidata.org/sparql"
    headers = {"Accept": "application/sparql-results+json"}
    response = requests.get(url, params={"query": query}, headers=headers)

    alt_labels = set()
    description = None

    if response.status_code == 200:
        results = response.json()["results"]["bindings"]
        for r in results:
            if "altLabel" in r:
                alt_labels.add(r["altLabel"]["value"])
            if "desc" in r and description is None:
                description = r["desc"]["value"]
    else:
        print(f"error HTTP {response.status_code} for {character_uri}")

    return description, list(alt_labels)

def create_characters_attributes_json():
    """
    Aggregate all characters (with and without Gutenberg ID) and retrieve their
    full set of Wikidata attributes.

    For each unique character URI, this function queries Wikidata to extract
    descriptions, aliases, and all available properties, and saves the resulting
    structured data as a JSON file .
    """

    # no id
    df_noid = pd.read_csv("characters_no_gutembergID.csv", encoding="utf-8")
    # with id
    df_id = pd.read_csv("characters_gutembergID.csv", encoding="utf-8")
    #concat
    df = pd.concat([df_noid, df_id], ignore_index=True)
    #remove duplicates
    df = df.drop_duplicates(subset=["character_name", "character_uri"])

    print(f"Total characters: {len(df)}")
    # Loop over characters and create json with attributes
    characters_attributes = []
    for idx, row in df.iterrows():
        attributes = retrieve_characters_attributes(row["character_name"], row["character_uri"])
        characters_attributes.append(attributes)
        time.sleep(2)

    # Save to JSON file well formatted
    with open("characters_attributes_wikidata.json", "w", encoding="utf-8") as f:
        json.dump(characters_attributes, f, ensure_ascii=False, indent=4)

def attributes_names_count():
    """
    Count the frequency of Wikidata attribute names across all characters.

    This function loads the character-level Wikidata JSON file, aggregates
    how often each attribute appears, and outputs a ranked list of the most
    frequent properties.
    """

    # Load the JSON file
    with open("characters_attributes_wikidata.json", "r", encoding="utf-8") as f:
        data = json.load(f)

    # Count the number of attributes for each character
    attributes_count = Counter()
    for character in data:
        attributes_count.update(character["attributes"].keys())

    # Convert to DataFrame and sort by count
    df = pd.DataFrame(attributes_count.items(), columns=["attribute", "count"]).sort_values(by="count", ascending=False)
    print(df.head(50))



def table_creation():
    """
    Create the final structured character attribute dataset from Wikidata.

    This function is the main dataset construction step: it transforms the raw
    Wikidata character JSON into a clean tabular dataset (one row per character),
    ready to be used as the gold reference for annotation, evaluation, and experiments.
    """

    # Load the JSON data
    with open("characters_attributes_wikidata.json", "r") as f:
        data = json.load(f)

    records = []

    for character in data:
        name = character.get("character_name")
        uri = character.get("character_uri")
        description = character.get("description")
        aliases = ", ".join(character.get("also_known_as", []))
        attributes = character.get("attributes", {})

        flat_attributes = {}
        for k, v in attributes.items():
            if k == "name in native language" and isinstance(v, dict):
                flat_attributes["name_in_native_language"] = v.get("text", "")
                flat_attributes["native_language_from_name"] = v.get("lang", "")
            elif isinstance(v, list):
                flat_attributes[k] = ", ".join(
                    str(item) if not isinstance(item, dict) else str(item.get("text", str(item)))
                    for item in v
                )
            elif isinstance(v, dict):
                flat_attributes[k] = v.get("text", str(v))
            else:
                flat_attributes[k] = str(v)

        record = {
            "character_name": name,
            "character_uri": uri,
            "description": description,
            "also_known_as": aliases,
            **flat_attributes
        }

        records.append(record)

    df = pd.DataFrame(records)
    print(df.head())

    # Save to CSV with semicolon as separator
    df.to_csv("characters_attributes_wikidata.csv", sep=";", index=False, encoding="utf-8")



def main():
    # The following functions together create the dataset, before manual adjustment 

    #Retrieve books with a Gutenberg ID already registered in Wikidata, extract their characters, and analyze available Wikidata properties
    create_gutemberg_persos_and_properties()

    #Retrieve books published before 1940 without a Gutenberg ID in Wikidata and extract their associated characters
    without_gutenberg_id()

    # Check which of these books are actually available on Project Gutenberg using title and author matching.
    check_gutenberg_without_id()

    # Analyze Wikidata properties available for characters from books confirmed to exist on Gutenberg but without a Gutenberg ID in Wikidata
    check_attributes_characters_no_id()
    # print(retrieve_characters_attributes("Ármin Vámbéry", "http://www.wikidata.org/entity/Q2404481"))

    # Retrieve full Wikidata attributes (descriptions, aliases, properties) for all characters (with and without Gutenberg IDs) and store them as JSON.
    create_characters_attributes_json()

    #Build the final tabular dataset by flattening the Wikidata JSON into a structured table (one row per character)
    table_creation()
    

if __name__ == "__main__":
    main()