import json
import requests
import os
import time
from typing import List, Dict, Any, Optional
from pathlib import Path

# =========================
# Config
# =========================

BEARER_TOKEN = os.environ.get("TWITTER_BEARER_TOKEN")  # <-- set this environment variable

API_URL = "https://api.twitter.com/2/tweets"

# Required fields to map the output to the desired format
TWEET_FIELDS = "created_at,public_metrics,author_id"

EXPANSIONS = "author_id"

USER_FIELDS = "username,public_metrics,verified"

HEADERS = {
    "Authorization": f"Bearer {BEARER_TOKEN}"
}

# =========================
# Helpers
# =========================

def chunked(lst: List[str], n: int) -> List[List[str]]:
    """Splits a list into chunks of size n."""
    return [lst[i:i+n] for i in range(0, len(lst), n)]


def backoff_sleep(retry: int) -> None:
    """Exponential wait with slight jitter."""
    base = 2
    delay = min(60, (base ** retry))  # max 60s
    time.sleep(delay + (0.1 * retry))


def fetch_tweets_by_ids(ids: List[str]) -> Dict[str, Any]:
    """
    Calls /2/tweets?ids=... returning the JSON.
    Handles 429 with backoff; raises for unrecoverable errors.
    """
    params = {
        "ids": ",".join(ids),
        "tweet.fields": TWEET_FIELDS,
        "expansions": EXPANSIONS,
        "user.fields": USER_FIELDS
    }

    retries = 0
    while True:
        resp = requests.get(API_URL, headers=HEADERS, params=params, timeout=30)
        if resp.status_code == 200:
            return resp.json()
        if resp.status_code == 429:
            retries += 1
            backoff_sleep(retries)
            continue

        # Unrecoverable errors: try to print server detail
        try:
            detail = resp.json()
        except Exception:
            detail = resp.text
        raise RuntimeError(f"API error {resp.status_code}: {detail}")

def map_tweet_to_target(tweet: Dict[str, Any], user_by_id: Dict[str, Dict[str, Any]]) -> Dict[str, Any]:
    """
    Converts a v2 tweet into the required format:
    {
      'created_at', 'id_str', 'text',
      'favorite_count', 'user_id', 'user_screen_name',
      'followers_count', 'following_count', 'verified'
    }
    """
    author_id = tweet.get("author_id")
    u = user_by_id.get(author_id, {}) if author_id else {}

    return {
        "created_at": tweet.get("created_at"),
        "id_str": str(tweet.get("id")),
        "text": tweet.get("text"),
        "favorite_count": (tweet.get("public_metrics") or {}).get("like_count"),
        "user_id": str(author_id) if author_id is not None else None,
        "user_screen_name": u.get("username"),
        "followers_count": (u.get("public_metrics") or {}).get("followers_count"),
        "following_count": (u.get("public_metrics") or {}).get("following_count"),
        "verified": u.get("verified")
    }


def rehydrate_batch(ids_in_order: List[str]) -> List[Optional[Dict[str, Any]]]:
    """
    Rehydrates a batch (<=100 IDs recommended). Keeps the order of the input IDs.
    If an ID is unrecoverable (deleted, private, etc.) it places None in its place.
    """
    raw = fetch_tweets_by_ids(ids_in_order)

    # Map users
    includes = raw.get("includes", {}) or {}
    users = includes.get("users", []) or []
    user_by_id = {u["id"]: u for u in users if "id" in u}

    # Map retrieved tweets
    tweets = raw.get("data", []) or []
    tweet_by_id = {str(t["id"]): t for t in tweets if "id" in t}

    # Rebuild in the original sequence
    out = []
    for tid in ids_in_order:
        t = tweet_by_id.get(str(tid))
        if not t:
            out.append(None)  # not found / unavailable
        else:
            out.append(map_tweet_to_target(t, user_by_id))
    return out

def rehydrate_nested(anonymized_nested: List[List[Dict[str, Any]]],
                     save_path: Optional[str] = None) -> List[List[Optional[Dict[str, Any]]]]:
    """
    Rehydrates a list of lists of dictionaries (each with 'id_str') and
    returns a parallel structure with the rehydrated dictionaries.

    If save_path is provided, saves in .jsonl format (one line per sub-list).
    """
    if not BEARER_TOKEN:
        raise EnvironmentError("Set the environment variable TWITTER_BEARER_TOKEN with a valid Bearer Token.")

    rehydrated_all: List[List[Optional[Dict[str, Any]]]] = []

    for sublist in anonymized_nested:
        ids = [str(d.get("id_str")) for d in sublist if d and d.get("id_str") is not None]

        # Calls in chunks of 100
        chunk_results: List[Optional[Dict[str, Any]]] = []
        for chunk in chunked(ids, 100):
            chunk_results.extend(rehydrate_batch(chunk))

        # chunk_results is in the sequence of IDs; we need to realign it to the length of sublist
        id_to_result = {ids[i]: chunk_results[i] for i in range(len(ids))}
        rehydrated_sublist: List[Optional[Dict[str, Any]]] = []
        for item in sublist:
            tid = str(item.get("id_str")) if item else None
            rehydrated_sublist.append(id_to_result.get(tid) if tid is not None else None)

        rehydrated_all.append(rehydrated_sublist)

        # Optional progressive saving for large files
        if save_path:
            with open(save_path, "a", encoding="utf-8") as f:
                f.write(json.dumps(rehydrated_sublist, ensure_ascii=False) + "\n")


    return rehydrated_all



if __name__ == "__main__":

    ROOT = Path().cwd()  
    DATA_REPO = ROOT / "FakeNewsNet" / "data"

    anonymized_path_fake = DATA_REPO / "anonymized_fake_propagation_paths.jsonl"  # <-- your input file
    output_path_fake = DATA_REPO / "ordered_fake_propagation_paths.jsonl"      # <-- optional: save line-by-line

    # Read the jsonl: each line is a list of dictionaries with 'id_str'
    with open(anonymized_path_fake, "r", encoding="utf-8") as f:

        anonymized_data_fake = [json.loads(line) for line in f]

    # If the output file already exists, print a message instead of deleting it
    if os.path.exists(output_path_fake):
        print(f"File {output_path_fake} is already present in folder. Delete it to proceed.")
    else:
        rehydrated_data_fake = rehydrate_nested(anonymized_data_fake, save_path=output_path_fake)

    anonymized_path_real = DATA_REPO / "anonymized_real_propagation_paths.jsonl"  # <-- your input file
    output_path_real = DATA_REPO / "ordered_real_propagation_paths.jsonl"      # <-- optional: save line-by-line

    # Read the jsonl: each line is a list of dictionaries with 'id_str'
    with open(anonymized_path_real, "r", encoding="utf-8") as f:
        anonymized_data_real = [json.loads(line) for line in f]


    # If the output file already exists, print a message instead of deleting it
    if os.path.exists(output_path_real):
        print(f"File {output_path_real} is already present in folder. Delete it to proceed.")
    else:
        rehydrated_data_real = rehydrate_nested(anonymized_data_real, save_path=output_path_real)