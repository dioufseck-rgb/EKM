"""
pipeline/stage1_crawl.py — Crawl Wikipedia articles on financial services.

Concurrency: 1 (Wikipedia API rate-limiting). No LLM cost.
"""
import json
import logging
import os
import time
from datetime import datetime

import requests

from pipeline.checkpoint import CheckpointManager
from pipeline.config import settings

logger = logging.getLogger(__name__)

WIKI_API = "https://en.wikipedia.org/w/api.php"
WIKI_HEADERS = {"User-Agent": "FinWikiKG/1.0 (educational research bot; contact research@example.com)"}

# Seed list — 100 Wikipedia financial services articles for the demo run.
SEED_ARTICLES = [
    # Regulation & compliance
    "Basel III", "Basel II", "Basel I",
    "Dodd-Frank Wall Street Reform and Consumer Protection Act",
    "Sarbanes-Oxley Act", "Glass-Steagall legislation",
    "Volcker Rule", "Know your customer", "Anti-money laundering",
    "Foreign Account Tax Compliance Act",
    "Payment Card Industry Data Security Standard",
    "General Data Protection Regulation",
    "Markets in Financial Instruments Directive",
    "Alternative Investment Fund Managers Directive",
    "Solvency II Directive", "Bank Secrecy Act",
    "Consumer Financial Protection Bureau",
    "Financial Industry Regulatory Authority",
    "Office of the Comptroller of the Currency",
    # Capital & risk
    "Capital requirement", "Tier 1 capital", "Tier 2 capital",
    "Risk-weighted asset", "Liquidity coverage ratio",
    "Net stable funding ratio", "Leverage ratio",
    "Credit risk", "Market risk", "Operational risk",
    "Systemic risk", "Counterparty risk", "Liquidity risk",
    "Interest rate risk", "Currency risk", "Concentration risk",
    "Stress test (financial)", "Value at risk", "Expected shortfall",
    "Probability of default",
    # Institutions & bodies
    "Federal Reserve", "European Central Bank",
    "Bank for International Settlements",
    "Financial Stability Board", "Securities and Exchange Commission",
    "Society for Worldwide Interbank Financial Telecommunication",
    "Financial Action Task Force", "International Monetary Fund",
    "World Bank", "Federal Deposit Insurance Corporation",
    # Securities & instruments
    "Asset-backed security", "Mortgage-backed security",
    "Collateralized debt obligation", "Collateralized loan obligation",
    "Credit default swap", "Interest rate swap", "Currency swap",
    "Derivative (finance)", "Option (finance)", "Futures contract",
    "Forward contract", "Bond (finance)", "Equity (finance)",
    "Preferred stock", "Exchange-traded fund",
    "Money market fund", "Hedge fund", "Private equity",
    "Venture capital",
    # Accounting & valuation
    "International Financial Reporting Standards",
    "Generally Accepted Accounting Principles",
    "Fair value accounting", "Mark-to-market accounting",
    "Hedge accounting", "Impairment (financial reporting)",
    "Goodwill (accounting)", "Revenue recognition",
    "Deferred tax", "Earnings per share",
    # Quantitative finance
    "Beta (finance)", "Sharpe ratio", "Capital asset pricing model",
    "Black-Scholes model", "Efficient market hypothesis",
    "Modern portfolio theory", "Arbitrage pricing theory",
    "Monte Carlo methods in finance", "Duration (finance)",
    "Yield curve",
    # Banking & payments
    "Fractional-reserve banking", "Central bank",
    "Commercial bank", "Investment banking",
    "Retail banking", "Correspondent banking",
    "Open banking", "Cryptocurrency", "Blockchain",
    "Central bank digital currency",
    # Corporate finance
    "Weighted average cost of capital", "Net present value",
    "Internal rate of return", "Dividend discount model",
    "Leveraged buyout", "Initial public offering",
    "Mergers and acquisitions", "Financial distress",
    "Bankruptcy", "Credit rating",
]


def fetch_article(title: str) -> dict | None:
    params = {
        "action":      "query",
        "prop":        "extracts|info",
        "titles":      title,
        "explaintext": "1",   # plain text (not HTML)
        "inprop":      "url",
        "format":      "json",
        "redirects":   "1",
    }
    try:
        resp = requests.get(WIKI_API, params=params, headers=WIKI_HEADERS, timeout=30)
        resp.raise_for_status()
        data = resp.json()
    except Exception as e:
        logger.error(f"HTTP error fetching '{title}': {e}")
        return None

    pages = data.get("query", {}).get("pages", {})
    if not pages:
        return None
    page = next(iter(pages.values()))
    if "missing" in page:
        logger.warning(f"Article not found: {title}")
        return None

    return {
        "title":      page.get("title", title),
        "url":        page.get("fullurl", f"https://en.wikipedia.org/wiki/{title.replace(' ', '_')}"),
        "content":    page.get("extract", ""),
        "word_count": len(page.get("extract", "").split()),
        "fetched_at": datetime.utcnow().isoformat(),
        "domain":     _infer_domain(title),
    }


def _infer_domain(title: str) -> str:
    title_lower = title.lower()
    if any(k in title_lower for k in ["capital", "basel", "tier", "liquidity", "risk-weighted"]):
        return "banking"
    if any(k in title_lower for k in ["sec", "securities", "mifid", "fund", "etf"]):
        return "securities"
    if any(k in title_lower for k in ["insurance", "solvency"]):
        return "insurance"
    if any(k in title_lower for k in ["swap", "derivative", "option", "futures"]):
        return "derivatives"
    if any(k in title_lower for k in ["gdpr", "aml", "kyc", "fatca", "pci", "swift"]):
        return "regulatory"
    if any(k in title_lower for k in ["ifrs", "gaap", "accounting", "fair value"]):
        return "accounting"
    return "finance"


def crawl() -> None:
    checkpoint = CheckpointManager("stage1_crawl")
    already_done = checkpoint.get_completed_ids()
    work_queue = [t for t in SEED_ARTICLES if t not in already_done]

    checkpoint.set_total(len(SEED_ARTICLES))
    logger.info(f"Stage 1: {len(work_queue)} articles to crawl ({len(already_done)} already done)")

    os.makedirs(settings.raw_dir, exist_ok=True)

    for title in work_queue:
        article = fetch_article(title)
        if not article:
            checkpoint.mark_failed(title, "not_found_or_http_error")
            continue

        safe_name = title.replace("/", "_").replace(" ", "_").replace("(", "").replace(")", "")
        path = os.path.join(settings.raw_dir, f"{safe_name}.json")
        try:
            with open(path, "w") as f:
                json.dump(article, f, indent=2)
            checkpoint.mark_done(title)
            logger.info(f"Saved: {title} ({article['word_count']} words)")
        except OSError as e:
            checkpoint.mark_failed(title, str(e))

        time.sleep(0.3)  # polite rate-limiting

    checkpoint.complete()


def run() -> None:
    logging.basicConfig(level=getattr(logging, settings.log_level))
    crawl()


if __name__ == "__main__":
    run()
