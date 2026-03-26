#!/usr/bin/env python3
"""
TikTok Profile Video URL Extractor
Uses yt-dlp to extract all video URLs from a TikTok user profile.
Usage: python tiktok_scraper.py <tiktok_profile_url>
"""

import sys
import json
import subprocess
import importlib.util

# ── Auto-install yt-dlp if missing ──────────────────────────────────────────
def ensure_ytdlp():
    if importlib.util.find_spec("yt_dlp") is None:
        print("[*] Installing yt-dlp...")
        subprocess.check_call([sys.executable, "-m", "pip", "install", "-q", "yt-dlp"])
    import yt_dlp
    return yt_dlp

yt_dlp = ensure_ytdlp()

# ── Core extractor ───────────────────────────────────────────────────────────
def get_tiktok_video_urls(profile_url: str) -> list[dict]:
    """
    Extract all video URLs from a TikTok profile.

    Returns a list of dicts:
        {
            "video_id": str,
            "url":      str,   # direct TikTok watch URL
            "title":    str,
            "likes":    int,
            "views":    int,
            "date":     str,   # YYYYMMDD
        }
    """
    ydl_opts = {
        # ── Stealth / anti-detection settings ────────────────────────────
        "quiet":             True,
        "no_warnings":       True,
        "simulate":          True,   # never download — metadata only
        "extract_flat":      True,   # fast: just collect entries, no deep fetch
        "skip_download":     True,
        "ignoreerrors":      True,

        # Rotate user-agent to a real browser string
        "http_headers": {
            "User-Agent": (
                "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
                "AppleWebKit/537.36 (KHTML, like Gecko) "
                "Chrome/123.0.0.0 Safari/537.36"
            ),
            "Referer":    "https://www.tiktok.com/",
            "Accept-Language": "en-US,en;q=0.9",
        },

        # Reasonable rate-limit to avoid bans
        "sleep_interval":      1,
        "max_sleep_interval":  3,
        "ratelimit":           500_000,  # 500 KB/s cap on metadata requests
    }

    results = []

    with yt_dlp.YoutubeDL(ydl_opts) as ydl:
        info = ydl.extract_info(profile_url, download=False)

        if info is None:
            print("[!] Could not fetch profile. The account may be private or the URL is wrong.")
            return results

        entries = info.get("entries", [])

        for entry in entries:
            if not entry:
                continue

            video_id = entry.get("id", "")
            video_url = entry.get("url") or entry.get("webpage_url") or f"https://www.tiktok.com/@_/video/{video_id}"

            results.append({
                "video_id": video_id,
                "url":      video_url,
                "title":    entry.get("title",       "N/A"),
                "likes":    entry.get("like_count",   0),
                "views":    entry.get("view_count",   0),
                "date":     entry.get("upload_date",  "N/A"),
            })

    return results


# ── Pretty printer ───────────────────────────────────────────────────────────
def print_results(videos: list[dict]):
    if not videos:
        print("[!] No videos found.")
        return

    print(f"\n{'='*60}")
    print(f"  Found {len(videos)} video(s)")
    print(f"{'='*60}\n")

    for i, v in enumerate(videos, 1):
        print(f"[{i:>3}] {v['url']}")
        print(f"       Title : {v['title'][:60]}")
        print(f"       Date  : {v['date']}  |  Views: {v['views']:,}  |  Likes: {v['likes']:,}")
        print()


# ── Save to JSON ─────────────────────────────────────────────────────────────
def save_json(videos: list[dict], path: str = "tiktok_videos.json"):
    with open(path, "w", encoding="utf-8") as f:
        json.dump(videos, f, ensure_ascii=False, indent=2)
    print(f"[+] Results saved to {path}")


# ── Entry point ──────────────────────────────────────────────────────────────
def main():
    if len(sys.argv) < 2:
        print("Usage: python tiktok_scraper.py <profile_url>")
        print("Example: python tiktok_scraper.py https://www.tiktok.com/@samy.chaffai")
        sys.exit(1)

    profile_url = sys.argv[1].strip()
    print(f"[*] Fetching videos from: {profile_url}")

    videos = get_tiktok_video_urls(profile_url)
    print_results(videos)

    if videos:
        save_json(videos)

        # Also print plain URL list for easy copy-paste
        print("\n── Plain URL list ──────────────────────────────────────")
        for v in videos:
            print(v["url"])


if __name__ == "__main__":
    main()