import glob
import json
import os
import tqdm
from datetime import datetime
from pathlib import Path

# Purpose: Traverse Politifact dataset folders, load individual tweet JSON files,
# extract a compact subset of tweet + user fields, and write one JSON list per
# source folder as JSON Lines (JSONL). Each output line corresponds to all
# tweets found in a single base_folder (e.g., one claim/story folder).
# Code assumes you have the original dataset in a folder 'FakeNewsNet/data/raw_data'

# No need to execute this script if you already have the anonymous ordered JSONL files to be rehydrated by rebuild_dataset.py

def load_json(filepath):
    """Load a JSON file and return its parsed Python object."""
    with open(filepath, 'r', encoding='utf-8') as f:
        return json.load(f)


def get_tweet_data(original_dict):
    """Extract essential fields from a tweet JSON dict."""
    # Base tweet fields we care about (add more if downstream needs them)
    keys_to_keep = ['created_at', 'id_str', 'text', 'favorite_count']

    # Mapping of user sub-dict fields -> flattened output keys
    user_keys_to_keep = {
        'id_str': 'user_id',
        'screen_name': 'user_screen_name',
        'followers_count':'followers_count',
        'friends_count':'following_count',
        'verified':'verified'
    }

    filtered_dict = {}
    for key in keys_to_keep:
        if key in original_dict:
            filtered_dict[key] = original_dict[key]

    if 'user' in original_dict and isinstance(original_dict['user'], dict):
        # Flatten selected user fields into top-level keys
        user_details = original_dict['user']
        for original_key, new_key in user_keys_to_keep.items():
            if original_key in user_details:
                filtered_dict[new_key] = user_details[original_key]

    return filtered_dict


def create_propagation_paths_jsonl(base_folder):
    """Create propagation path entries by iterating over the `tweets` folder inside base_folder."""

    tweets_folder = f"{base_folder}/tweets"  # Folder containing tweets
    
    if not os.path.exists(tweets_folder):
        # Gracefully skip folders that do not contain a tweets directory
        return

    propagation_paths = []

    for tweet_file in os.listdir(tweets_folder):  # Iterate all files in tweets folder
        if tweet_file.endswith(".json"):  # Only process JSON files
            tweet_path = os.path.join(tweets_folder, tweet_file)
            tweet_json = load_json(tweet_path)  # Load tweet JSON
            tweet_data = get_tweet_data(tweet_json)  # Extract compact representation
            propagation_paths.append(tweet_data)  # Accumulate for this base folder

    # Sort tweets chronologically by created_at before returning
    propagation_paths.sort(
        key=lambda x: datetime.strptime(x['created_at'], "%a %b %d %H:%M:%S %z %Y")
        if 'created_at' in x else datetime.max  # fallback if missing
    )

    return propagation_paths


def process_folders_json(input_pattern, output_file):
    """
    Processes data folders and writes one list of propagation paths per line (JSONL).
    """
    # Expand the glob pattern into concrete base folders (e.g., each claim/story)
    with open(output_file, 'w', encoding='utf-8') as file:
        for base_folder in tqdm.tqdm(glob.glob(input_pattern)):
            propagation_paths = create_propagation_paths_jsonl(base_folder)
            if propagation_paths:  # Only write non-empty results
                # Each line is a JSON array of simplified tweets for one base folder
                file.write(json.dumps(propagation_paths, ensure_ascii=False) + "\n")



if __name__ == "__main__":
    # Generate propagation path JSONL files for real & fake Politifact subsets.
    # Adjust the glob patterns or output filenames as needed.

    ROOT = Path().cwd()
    DATA_DIR = ROOT / 'FakeNewsNet' / 'data'

    process_folders_json(f'{DATA_DIR}/raw_data/politifact/real/*', f'{DATA_DIR}/ordered_real_propagation_paths.jsonl')
    process_folders_json(f'{DATA_DIR}/raw_data/politifact/fake/*', f'{DATA_DIR}/ordered_fake_propagation_paths.jsonl')