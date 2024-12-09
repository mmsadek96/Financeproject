import requests
import asyncio
import openai
import logging
from flask import Blueprint, render_template, request
from cachetools import TTLCache
import time
import os
from dotenv import load_dotenv


load_dotenv()

market_sentiment_bp = Blueprint(
    "market_sentiment", __name__, template_folder="templates"
)

# Logging configuration
logging.basicConfig(level=logging.DEBUG)

# API Keys
NEWS_API_KEY = "2daec4b767b94998ada443b03b175081"
openai.api_key = os.getenv("OPENAI_API_KEY")


# Cache setup
cache = TTLCache(maxsize=100, ttl=300)

# Helper function for logging
def log_time_and_debug(message, start_time):
    elapsed_time = time.time() - start_time
    logging.debug(f"{message} took {elapsed_time:.2f} seconds")

# Fetch news articles from NewsAPI
def fetch_news(company_name, limit=30):
    logging.debug(f"Fetching news articles for {company_name}")
    url = f"https://newsapi.org/v2/everything?q={company_name}&sortBy=publishedAt&apiKey={NEWS_API_KEY}&pageSize={limit}"
    try:
        response = requests.get(url)
        response.raise_for_status()
        data = response.json()
        if data.get("status") == "ok":
            return data.get("articles", [])
        else:
            logging.error(f"Error fetching news: {data.get('message')}")
            return []
    except Exception as e:
        logging.error(f"Error during news fetching: {e}")
        return []

# Analyze news sentiment using OpenAI
async def analyze_news_with_openai(articles, company_name, semaphore, batch_size=5):
    async with semaphore:
        try:
            if not articles:
                return "No articles found for analysis."

            sentiment_summary = ""
            for i in range(0, len(articles), batch_size):
                article_batch = articles[i : i + batch_size]
                prompt = f"Analyze the sentiment (positive, neutral, negative) of these news articles about {company_name}.\n\n"
                for j, article in enumerate(article_batch, start=1):
                    title = article.get("title", "No Title Available")
                    description = article.get("description", "") or ""
                    content = article.get("content", "") or ""
                    prompt += f"Article {j}:\nTitle: {title}\nContent: {description} {content[:1000]}\n\n"

                response = await openai.ChatCompletion.acreate(
                    model="gpt-4o-mini",
                    messages=[
                        {"role": "system", "content": "You are a financial analyst."},
                        {"role": "user", "content": prompt},
                    ],
                    max_tokens=1500,
                    temperature=0.7,
                )
                batch_summary = response["choices"][0]["message"]["content"].strip()
                sentiment_summary += batch_summary + "\n\n"

            return sentiment_summary.strip()
        except Exception as e:
            logging.error(f"Error analyzing news: {e}")
            return "An error occurred during sentiment analysis."

# Blueprint routes
@market_sentiment_bp.route("/", methods=["GET", "POST"])
async def index():
    if request.method == "POST":
        try:
            company_name = request.form["company_name"]
            ticker = request.form["ticker"]
            semaphore = asyncio.Semaphore(3)

            # Fetch news articles
            news_articles = fetch_news(company_name)
            news_sentiment = await analyze_news_with_openai(
                news_articles, company_name, semaphore
            )

            # Generate actionable insights (placeholder example)
            actionable_insight = f"Insights for {company_name}: {news_sentiment}"

            return render_template(
                "market_sentiment/results.html",
                news_sentiment=news_sentiment,
                actionable_insight=actionable_insight,
            )
        except Exception as e:
            logging.error(f"Error processing request: {e}")
            return "An error occurred while processing your request.", 500

    return render_template("market_sentiment/index.html")
