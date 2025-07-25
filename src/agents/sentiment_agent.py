"""Kimi LLM news-sentiment agent."""
import httpx
from pydantic import BaseModel

from utils.logger import get_logger

log = get_logger("SENTIMENT_AGENT")

class SentimentScore(BaseModel):
    score: float  # -1 → bearish, 1 → bullish
    reasoning: str

class SentimentAgent:
    def __init__(self, api_key: str) -> None:
        self.api_key = api_key
        self.url = "https://api.moonshot.cn/v1/chat/completions"

    async def score_headline(self, headline: str) -> SentimentScore:
        payload = {
            "model": "kimi-latest",
            "messages": [
                {
                    "role": "user",
                    "content": f"Rate sentiment (-1..+1) for headline: {headline}",
                }
            ],
        }
        headers = {"Authorization": f"Bearer {self.api_key}"}
        async with httpx.AsyncClient(timeout=30) as client:
            r = await client.post(self.url, json=payload, headers=headers)
            r.raise_for_status()
            text = r.json()["choices"][0]["message"]["content"]
            # naive parse
            score = float(text.split()[0])
            return SentimentScore(score=score, reasoning=text)