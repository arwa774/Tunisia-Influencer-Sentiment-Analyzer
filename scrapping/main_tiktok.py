

import csv
import os
import asyncio
import json
import pandas as pd

from url_finder import *
from scrap_titok import *

OUTPUT_FILE = "tiktok_comments.csv"
CHECKPOINT_FILE = "checkpoint.json"
YOUR_PROXY = "http://xavdqtie:q3f3y1ctkiha@142.111.67.146:5611"

# ─── Thresholds ───────────────────────────────────────────────────────────────
MIN_COMMENTS_PER_USER   = 100    # keep scraping beyond 10 videos if below this
MAX_COMMENTS_PER_USER   = 1000   # stop early if we already hit this
DEFAULT_VIDEO_BATCH     = 10     # the "normal" video target per user
# ─────────────────────────────────────────────────────────────────────────────

df = pd.read_csv("dataset_ml_1.csv")
list_username = df["username"]

# --- CHECKPOINT SYSTEM (unchanged) ---
if os.path.exists(CHECKPOINT_FILE):
    with open(CHECKPOINT_FILE, "r", encoding="utf-8") as f:
        checkpoint_data = json.load(f)
else:
    checkpoint_data = {"completed_users": [], "completed_videos": []}

completed_users  = set(checkpoint_data["completed_users"])
completed_videos = set(checkpoint_data["completed_videos"])

def save_checkpoint():
    with open(CHECKPOINT_FILE, "w", encoding="utf-8") as f:
        json.dump({
            "completed_users":  list(completed_users),
            "completed_videos": list(completed_videos)
        }, f, indent=4)

# Write CSV header once
if not os.path.exists(OUTPUT_FILE):
    with open(OUTPUT_FILE, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=["user_name", "comment", "comment_likes"])
        writer.writeheader()


# --- MAIN SCRAPING LOOP ---
for username in list_username:

    if username in completed_users:
        print(f"[SKIP] @{username} already fully processed.")
        continue

    profile_url = f"https://www.tiktok.com/@{username}"
    videos_urls = get_tiktok_video_urls(profile_url)

    if not videos_urls:
        print(f"[!] No videos found for @{username}. Skipping.")
        completed_users.add(username)
        save_checkpoint()
        continue

    user_comment_count = 0   # total comments collected for this TikToker
    videos_scraped     = 0   # how many videos we've actually processed

    with open(OUTPUT_FILE, "a", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=["user_name", "comment", "comment_likes"])

        for i, url_data in enumerate(videos_urls, 1):
            video_url = url_data["url"]

            # ── EARLY-EXIT: already at or above the max ───────────────────
            if user_comment_count >= MAX_COMMENTS_PER_USER:
                print(f"[MAX] @{username} reached {user_comment_count} comments. Moving on.")
                break

            # ── NORMAL STOP: scraped ≥ DEFAULT_VIDEO_BATCH videos AND
            #    already above the minimum threshold ───────────────────────
            if videos_scraped >= DEFAULT_VIDEO_BATCH and user_comment_count >= MIN_COMMENTS_PER_USER:
                print(f"[DONE] @{username}: {videos_scraped} videos / "
                      f"{user_comment_count} comments — thresholds met.")
                break

            # ── Checkpoint skip ───────────────────────────────────────────
            if video_url in completed_videos:
                print(f"[SKIP] [{i}/{len(videos_urls)}] Already scraped: {video_url}")
                videos_scraped += 1   # counts toward the video budget
                continue

            # ── How many comments do we still want for this user? ─────────
            remaining_budget = MAX_COMMENTS_PER_USER - user_comment_count

            print(f"[SCRAPING] [{i}/{len(videos_urls)}] "
                  f"(user total so far: {user_comment_count}) — {video_url}")

            comments = asyncio.run(
                scrape_tiktok_comments(
                    video_url=video_url,
                    max_comments=remaining_budget,   # never over-collect
                    proxy=YOUR_PROXY
                )
            )

            for entry in comments:
                writer.writerow({
                    "user_name":     username,
                    "comment":       entry.get("text", ""),
                    "comment_likes": entry.get("likes", 0),
                })

            f.flush()

            batch_size          = len(comments)
            user_comment_count += batch_size
            videos_scraped     += 1

            print(f"  → +{batch_size} comments | user total: {user_comment_count}")

            completed_videos.add(video_url)
            save_checkpoint()

        # ── Warn if we exhausted all videos and still below minimum ───────
        if user_comment_count < MIN_COMMENTS_PER_USER:
            print(f"[WARN] @{username}: only {user_comment_count} comments after "
                  f"all {len(videos_urls)} available videos (below {MIN_COMMENTS_PER_USER} min).")

    completed_users.add(username)
    save_checkpoint()
    print(f"[+] Finished @{username} — {videos_scraped} videos, "
          f"{user_comment_count} total comments.\n")

print(f"[+] Done! Results saved to {OUTPUT_FILE}")