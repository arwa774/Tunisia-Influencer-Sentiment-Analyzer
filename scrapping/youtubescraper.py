"""
=============================================================
  YouTube Comment Scraper — Tunisian YouTubers
  Scrapes comments from videos of each channel
  Target: min 100, max 1000 comments per YouTuber
  Output: tunisian_youtubers_comments.csv
         Columns: username, comment
=============================================================
"""

from googleapiclient.discovery import build
from googleapiclient.errors import HttpError
import json
import csv
import os
import pandas as pd
from dotenv import load_dotenv

# Load .env file
load_dotenv()

# Get the string
api_keys_str = os.getenv("API_KEYS")

# Convert to list
api_keys = api_keys_str.split(",")

# ─────────────────────────────────────────────────
#  CONFIGURATION
# ─────────────────────────────────────────────────

API_KEYS  = api_keys # 🔑 Replace with your YouTube Data API v3 key
  # 🔑 Your YouTube Data API v3 keys
VIDEOS_BATCH_SIZE       = 10         # How many videos to fetch per batch
MAX_COMMENTS_PER_VIDEO  = 100        # Max comments pulled from one video
MIN_COMMENTS_PER_YOUTUBER = 100      # Keep fetching videos until we reach this
MAX_COMMENTS_PER_YOUTUBER = 1000     # Stop this YouTuber once we reach this
OUTPUT_CSV = "tunisian_youtubers_comments.csv"

df = pd.read_csv("dataset_ml.csv")
YOUTUBERS = list(df["username"])


# ─────────────────────────────────────────────────
#  API KEY ROTATION
# ─────────────────────────────────────────────────
_key_index = 0                               # tracks which key is active
youtube    = build("youtube", "v3", developerKey=API_KEYS[_key_index])


def _rotate_key():
    """
    Switch to the next API key and rebuild the YouTube client.
    Raises RuntimeError if all keys are exhausted.
    """
    global _key_index, youtube
    _key_index += 1
    if _key_index >= len(API_KEYS):
        raise RuntimeError(
            "❌ All API keys are quota-exhausted. Stop the scraper and try again tomorrow."
        )
    print(f"🔄 Quota hit — switching to API key #{_key_index + 1} of {len(API_KEYS)}...")
    youtube = build("youtube", "v3", developerKey=API_KEYS[_key_index])


def _is_quota_error(e: HttpError) -> bool:
    """Return True when the HTTP error signals a quota / daily-limit problem."""
    try:
        reason = e.error_details[0].get("reason", "")
    except Exception:
        reason = ""
    return e.resp.status == 403 and reason in ("quotaExceeded", "dailyLimitExceeded")


def api_call(make_request):
    """
    Execute `make_request()` (a zero-argument lambda that calls the API).
    On a quota error, rotate the key and retry automatically.
    On any other HTTP error, re-raise immediately.

    Why a lambda?
      Each time we retry we need a FRESH request object built against
      the NEW youtube client. Passing a lambda instead of a pre-built
      request ensures the global `youtube` variable is evaluated again
      after the key rotation.

    Example usage:
        response = api_call(lambda: youtube.search().list(...).execute())
    """
    while True:
        try:
            return make_request()
        except HttpError as e:
            if _is_quota_error(e):
                _rotate_key()   # updates global `youtube`, then loop retries
            else:
                raise           # comments disabled, video private, etc. — bubble up


# ─────────────────────────────────────────────────
#  CHECKPOINT SYSTEM  (unchanged)
# ─────────────────────────────────────────────────
CHECKPOINT_FILE = "youtube_checkpoint.json"

if os.path.exists(CHECKPOINT_FILE):
    with open(CHECKPOINT_FILE, "r", encoding="utf-8") as f:
        try:
            checkpoint_data = json.load(f)
        except json.JSONDecodeError:
            checkpoint_data = {}
else:
    checkpoint_data = {}

completed_channels = set(checkpoint_data.get("completed_channels", []))
completed_videos   = set(checkpoint_data.get("completed_videos",   []))


def save_checkpoint():
    with open(CHECKPOINT_FILE, "w", encoding="utf-8") as f:
        json.dump({
            "completed_channels": list(completed_channels),
            "completed_videos":   list(completed_videos),
        }, f, indent=4)


# ─────────────────────────────────────────────────
#  HELPER FUNCTIONS
# ─────────────────────────────────────────────────

def find_channel_id(query: str) -> tuple[str, str]:
    response = api_call(
        lambda: youtube.search().list(
            q=query, type="channel", part="snippet", maxResults=1
        ).execute()
    )
    items = response.get("items", [])
    if not items:
        raise ValueError(f"No channel found for query: '{query}'")
    snippet    = items[0]["snippet"]
    return snippet["channelId"], snippet["channelTitle"]


def get_videos_batch(
    channel_id: str,
    max_results: int = 10,
    page_token: str = None
) -> tuple[list[dict], str | None]:
    """
    Fetch one page of videos from a channel.
    Returns (list_of_videos, next_page_token_or_None).
    """
    params = {
        "channelId":  channel_id,
        "type":       "video",
        "part":       "snippet",
        "order":      "date",
        "maxResults": max_results,
    }
    if page_token:
        params["pageToken"] = page_token

    response = api_call(lambda: youtube.search().list(**params).execute())

    videos = [
        {
            "video_id": item["id"]["videoId"],
            "title":    item["snippet"]["title"],
            "url":      f"https://www.youtube.com/watch?v={item['id']['videoId']}",
        }
        for item in response.get("items", [])
    ]
    return videos, response.get("nextPageToken")


def get_comments(video_id: str, max_comments: int) -> list[dict]:
    """
    Pull up to `max_comments` top-level comments from a single video.
    Skips silently if comments are disabled.
    """
    comments   = []
    next_page  = None

    while len(comments) < max_comments:
        params = {
            "videoId":    video_id,
            "part":       "snippet",
            "order":      "relevance",
            "maxResults": min(100, max_comments - len(comments)),
            "textFormat": "plainText",
        }
        if next_page:
            params["pageToken"] = next_page

        try:
            response = api_call(lambda: youtube.commentThreads().list(**params).execute())
        except HttpError as e:
            # 403 commentsDisabled or 404 — just skip this video
            print(f"    ⚠️  Skipping video {video_id}: {e}")
            break

        for item in response.get("items", []):
            top = item["snippet"]["topLevelComment"]["snippet"]
            comments.append({"text": top["textDisplay"]})

        next_page = response.get("nextPageToken")
        if not next_page:
            break

    return comments


# ─────────────────────────────────────────────────
#  PER-YOUTUBER SCRAPER
# ─────────────────────────────────────────────────

def scrape_youtuber(youtuber: str, csvfile) -> int:
    """
    Scrape comments for one YouTuber and write them to the open CSV file.

    Comment-count logic
    ───────────────────
    • We accumulate comments across ALL videos of a channel (not per video).
    • Videos are fetched in batches of VIDEOS_BATCH_SIZE.
    • After each batch we check:
        - total >= MAX_COMMENTS_PER_YOUTUBER → stop immediately (cap hit).
        - total >= MIN_COMMENTS_PER_YOUTUBER → stop (minimum satisfied).
        - total <  MIN_COMMENTS_PER_YOUTUBER AND more pages exist → fetch next batch.
        - total <  MIN_COMMENTS_PER_YOUTUBER AND no more pages   → stop (channel exhausted).

    Returns the number of comments written for this YouTuber.
    """
    print(f"\n{'=' * 60}")
    print(f"🔍 Searching: '{youtuber}'...")

    try:
        channel_id, channel_title = find_channel_id(youtuber)
        print(f"✅ Found: {channel_title}  (ID: {channel_id})")
    except ValueError as e:
        print(f"❌ {e}")
        csv.writer(csvfile).writerow([youtuber, f"[ERROR] {e}"])
        csvfile.flush()
        completed_channels.add(youtuber)
        save_checkpoint()
        return 0

    writer          = csv.writer(csvfile)
    total_comments  = 0
    next_page_token = None
    batch_num       = 0

    while True:
        # ── 1. Fetch the next batch of videos ──────────────────────────
        batch_num += 1
        print(f"\n  📹 Video batch #{batch_num}  ({VIDEOS_BATCH_SIZE} videos)...")
        videos, next_page_token = get_videos_batch(
            channel_id, VIDEOS_BATCH_SIZE, next_page_token
        )

        if not videos:
            print(f"  ℹ️  No more videos available for {channel_title}.")
            break

        print(f"  ✅ {len(videos)} videos in batch")

        # ── 2. Pull comments from each video in the batch ──────────────
        for idx, video in enumerate(videos, 1):
            video_id = video["video_id"]

            if video_id in completed_videos:
                print(f"    [SKIP] Already scraped: {video['url']}")
                continue

            # Hard cap: don't even start a new video if we're already full
            if total_comments >= MAX_COMMENTS_PER_YOUTUBER:
                break

            # Only ask for as many comments as we still need
            remaining = MAX_COMMENTS_PER_YOUTUBER - total_comments
            to_fetch  = min(MAX_COMMENTS_PER_VIDEO, remaining)

            print(f"    💬 [{idx}/{len(videos)}] {video['title'][:55]}...")
            comments = get_comments(video_id, to_fetch)

            for c in comments:
                text = c["text"].replace("\n", " ").replace("\r", " ").strip()
                writer.writerow([channel_title, text])

            csvfile.flush()
            total_comments += len(comments)

            print(f"         ✅ +{len(comments)} comments  |  total: {total_comments}")

            completed_videos.add(video_id)
            save_checkpoint()

            # Cap hit mid-batch — stop processing this YouTuber now
            if total_comments >= MAX_COMMENTS_PER_YOUTUBER:
                print(f"\n  🏁 Cap reached ({MAX_COMMENTS_PER_YOUTUBER}). Moving to next YouTuber.")
                break

        # ── 3. Decide whether to fetch another batch ────────────────────
        if total_comments >= MAX_COMMENTS_PER_YOUTUBER:
            # Already broke out above; confirm and exit
            break

        if total_comments >= MIN_COMMENTS_PER_YOUTUBER:
            print(f"\n  ✅ Minimum met: {total_comments} comments collected.")
            break

        # Below minimum — do we have more pages?
        if next_page_token:
            print(
                f"\n  🔁 Only {total_comments} comment(s) so far "
                f"(< {MIN_COMMENTS_PER_YOUTUBER}). Fetching more videos..."
            )
            # Loop continues → fetches next batch
        else:
            print(
                f"\n  ⚠️  Channel exhausted. Only {total_comments} comment(s) "
                f"collected for {channel_title} (target was {MIN_COMMENTS_PER_YOUTUBER})."
            )
            break

    completed_channels.add(youtuber)
    save_checkpoint()
    print(f"  📊 Done with {channel_title}: {total_comments} total comments")
    return total_comments


# ─────────────────────────────────────────────────
#  MAIN
# ─────────────────────────────────────────────────

def main():
    print("🚀 Starting Tunisian YouTubers Comment Scraper")
    print(f"📋 Channels to scrape      : {len(YOUTUBERS)}")
    print(f"🎬 Videos per batch        : {VIDEOS_BATCH_SIZE}")
    print(f"💬 Comment target / channel: {MIN_COMMENTS_PER_YOUTUBER} – {MAX_COMMENTS_PER_YOUTUBER}")
    print(f"🔑 API keys loaded         : {len(API_KEYS)}")

    total_rows = 0

    if not os.path.exists(OUTPUT_CSV):
        with open(OUTPUT_CSV, "w", encoding="utf-8", newline="") as f:
            csv.writer(f).writerow(["username", "comment"])

    with open(OUTPUT_CSV, "a", encoding="utf-8", newline="") as csvfile:
        for youtuber in YOUTUBERS:
            if youtuber in completed_channels:
                print(f"\n[SKIP] '{youtuber}' already fully processed.")
                continue
            try:
                total_rows += scrape_youtuber(youtuber, csvfile)
            except RuntimeError as e:
                # All API keys exhausted — abort gracefully
                print(f"\n{e}")
                print("🛑 Stopping scraper. Progress is saved in checkpoint.")
                break

    print(f"\n{'=' * 60}")
    print(f"✅ SCRAPING COMPLETE")
    print(f"{'=' * 60}")
    print(f"📄 Output : {OUTPUT_CSV}")
    print(f"📝 Rows written this session: {total_rows}")


if __name__ == "__main__":
    main()