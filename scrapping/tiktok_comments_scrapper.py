import asyncio
import re
import json
import random
from httpx import AsyncClient
from urllib.parse import urlencode

async def scrape_tiktok_comments(
    video_url: str,
    max_comments: int = 100,
    proxy: str = None
      ):
    video_id = re.search(r'/video/(\d+)', video_url).group(1)
    comments = []
    cursor = 0
    collected = 0

    async with AsyncClient(
        headers={
            "accept": "text/html,application/xhtml+xml,application/xml;q=0.9,image/avif,image/webp,*/*;q=0.8",
            "accept-language": "en-US,en;q=0.9",
            "user-agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/141.0.0.0 Safari/537.36",
            "referer": "https://www.tiktok.com/",
        },
        proxy=proxy,
        timeout=30.0
    ) as client:
        try:
            # Step 1: Load video page to extract tokens (msToken, aid, region)
            page_resp = await client.get(video_url)
            text = page_resp.text

            params = {
                "aweme_id": video_id,
                "cursor": cursor,
                "count": 30,
                "current_region": "US",
                "aid": "1988"
            }

            # Extract dynamic tokens from HTML (required by TikTok in 2026)
            for pattern, key in [
                (r'"aid":(\d+)', "aid"),
                (r'"msToken":"([^"]+)"', "msToken"),
                (r'"region":"([^"]+)"', "region"),
            ]:
                match = re.search(pattern, text)
                if match:
                    params[key] = match.group(1) if key != "aid" else match.group(1)

            while collected < max_comments:
                api_url = "https://www.tiktok.com/api/comment/list/?" + urlencode(params)
                api_resp = await client.get(
                    api_url,
                    headers={"accept": "application/json", "referer": video_url}
                )

                if api_resp.status_code != 200:
                    print(f"Blocked or error: {api_resp.status_code}")
                    break

                data = api_resp.json()
                batch = data.get("comments", [])

                for c in batch:
                    if collected >= max_comments:
                        break
                    comments.append({
                        "text": c.get("text", ""),
                        "author_username": c.get("user", {}).get("unique_id", ""),
                        "author_nickname": c.get("user", {}).get("nickname", ""),
                        "likes": c.get("digg_count", 0),
                        "replies": c.get("reply_comment_total", 0),
                        "timestamp": c.get("create_time"),
                        "comment_id": c.get("cid")
                    })
                    collected += 1

                # Pagination
                if not data.get("has_more", False) or not batch:
                    break
                cursor += len(batch)
                params["cursor"] = cursor

                # HUMAN-LIKE DELAY (critical to avoid detection)
                await asyncio.sleep(random.uniform(8, 18))

            print(f"Successfully scraped {collected} comments")
            return comments

        except Exception as e:
            print(f"Error: {e}")
            return comments

# ====================== USAGE ======================
if __name__ == "__main__":
    VIDEO_URL = "https://www.tiktok.com/@samy.chaffai/video/7242711134845930757"  # ← CHANGE THIS
    YOUR_PROXY = "http://xavdqtie:q3f3y1ctkiha@142.111.67.146:5611"

    comments = asyncio.run(
        scrape_tiktok_comments(VIDEO_URL, max_comments=300, proxy=YOUR_PROXY)
    )

    with open("tiktok_comments.json", "w", encoding="utf-8") as f:
        json.dump(comments, f, indent=2, ensure_ascii=False)

    print(f"✅ Saved {len(comments)} comments to tiktok_comments.json")