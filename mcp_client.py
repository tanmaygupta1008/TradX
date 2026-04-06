"""
mcp_client.py  —  Strategic Intelligence MCP Client
=====================================================
Calls mcp_server.py tools to gather data, then sends it to a
HuggingFace LLM (via an Inference Provider) for competitive intelligence.

Flow:
  1. Call MCP tools → get stock data + sentiment for each company
  2. Bundle all data into a structured prompt
  3. Send to HuggingFace via chat_completion (Inference Provider, no URL)
  4. Parse and return: opportunities, threats, trend summary, positioning

Environment variables required in .env:
  HF_TOKEN    — your HuggingFace access token (hf_...)
  HF_MODEL    — HF model repo ID, e.g. meta-llama/Llama-3.1-8B-Instruct
  HF_PROVIDER — inference provider, e.g. together  (default: together)

Usage (called from app.py — not run directly):
    from mcp_client import run_intelligence
    result = asyncio.run(run_intelligence(companies_dict))
"""

import re
import os
import json
import asyncio
from dotenv import load_dotenv
from fastmcp import Client
from huggingface_hub import InferenceClient

load_dotenv()

HF_TOKEN    = os.getenv("HF_TOKEN", "")
HF_MODEL    = os.getenv("HF_MODEL",    "meta-llama/Llama-3.1-8B-Instruct")
HF_PROVIDER = os.getenv("HF_PROVIDER", "together")


# ─────────────────────────────────────────────────────────────────────────────
# Helper — safely extract text from a fastmcp CallToolResult
# Handles both old API (returns list) and new API (returns CallToolResult obj)
# ─────────────────────────────────────────────────────────────────────────────
def _extract_text(result) -> str:
    try:
        if hasattr(result, "content") and result.content:
            return result.content[0].text          # new fastmcp API
        if isinstance(result, list) and result:
            return result[0].text                  # old fastmcp API
    except Exception:
        pass
    return "{}"


# ─────────────────────────────────────────────────────────────────────────────
# Call HuggingFace via SDK — provider-based, no URL hardcoded
# ─────────────────────────────────────────────────────────────────────────────
def call_huggingface(user_prompt: str) -> str:
    if not HF_TOKEN:
        return json.dumps({
            "summary": "HuggingFace token not set. Add HF_TOKEN to your .env file.",
            "opportunities": ["Set up HF_TOKEN in .env to enable AI insights"],
            "threats":       ["Missing API token limits intelligence capabilities"],
            "positioning":   "N/A — token required",
            "recommendation": "Add your HuggingFace token to .env and restart.",
        })

    try:
        client = InferenceClient(
            provider=HF_PROVIDER,
            api_key=HF_TOKEN,      # ← use api_key= not token=
        )

        # Use chat.completions.create — works with ALL providers
        completion = client.chat.completions.create(
            model=HF_MODEL,
            messages=[
                {
                    "role": "system",
                    "content": (
                        "You are a strategic business intelligence analyst. "
                        "Always respond with valid JSON only — no markdown, "
                        "no explanation, no extra text. Output only the JSON object."
                    )
                },
                {
                    "role": "user",
                    "content": user_prompt
                }
            ],
            max_tokens=700,
            temperature=0.4,
        )

        return completion.choices[0].message.content.strip()

    except Exception as e:
        err = f"{type(e).__name__}: {e}"
        return json.dumps({
            "summary": f"HuggingFace inference error: {err}",
            "opportunities": [],
            "threats":       [err],
            "positioning":   "Error",
            "recommendation": (
                f"Check HF_TOKEN, HF_MODEL='{HF_MODEL}', and "
                f"HF_PROVIDER='{HF_PROVIDER}' in .env. "
                "Ensure the model is supported by the chosen provider."
            ),
        })


# ─────────────────────────────────────────────────────────────────────────────
# Build the intelligence prompt
# ─────────────────────────────────────────────────────────────────────────────
def build_prompt(companies_data: list) -> str:
    summaries = []
    for c in companies_data:
        stock = c.get("stock", {})
        sent  = c.get("sentiment", {})

        mc  = stock.get("market_cap",  0) or 0
        rev = stock.get("revenue",     0) or 0
        mc_str  = f"${mc/1e9:.1f}B"  if mc  >= 1e9 else f"${mc/1e6:.0f}M"
        rev_str = f"${rev/1e9:.1f}B" if rev >= 1e9 else f"${rev/1e6:.0f}M"

        summaries.append(
            f"- {stock.get('name', c.get('name'))} ({stock.get('ticker', '')}) | "
            f"Sector: {stock.get('sector', 'N/A')} | "
            f"Price: ${stock.get('price', 'N/A')} ({stock.get('change_pct', 0):+.1f}%) | "
            f"Mkt Cap: {mc_str} | Revenue: {rev_str} | "
            f"P/E: {stock.get('pe_ratio', 'N/A')} | "
            f"Sentiment: {sent.get('label', 'N/A')} (score: {sent.get('score', 0)})"
        )

    companies_block = "\n".join(summaries)

    return f"""Analyze the following real-time company data and provide a competitive intelligence report.

COMPANY DATA:
{companies_block}

Respond ONLY with this exact JSON structure (no extra text, no markdown):
{{
  "summary": "2-3 sentence overall market summary",
  "opportunities": [
    "Specific opportunity 1 based on the data",
    "Specific opportunity 2 based on the data",
    "Specific opportunity 3 based on the data"
  ],
  "threats": [
    "Specific threat 1 based on the data",
    "Specific threat 2 based on the data",
    "Specific threat 3 based on the data"
  ],
  "positioning": "Which company appears strongest and why, in 1-2 sentences",
  "recommendation": "Top strategic recommendation for decision makers in 1-2 sentences"
}}"""


# ─────────────────────────────────────────────────────────────────────────────
# Parse LLM output — extract JSON safely
# ─────────────────────────────────────────────────────────────────────────────
# def parse_llm_output(raw: str) -> dict:
#     try:
#         start = raw.find("{")
#         end   = raw.rfind("}") + 1
#         if start != -1 and end > start:
#             return json.loads(raw[start:end])
#     except json.JSONDecodeError:
#         pass

#     return {
#         "summary":        raw[:300] if raw else "No summary generated.",
#         "opportunities":  ["Analysis generated — see summary"],
#         "threats":        ["Analysis generated — see summary"],
#         "positioning":    "See summary above",
#         "recommendation": "Review summary for insights",
#     }
def parse_llm_output(raw: str) -> dict:
    # Strip markdown code fences if model wraps in ```json ... ```
    cleaned = raw.strip()
    if cleaned.startswith("```"):
        cleaned = re.sub(r"^```[a-z]*\n?", "", cleaned)
        cleaned = re.sub(r"\n?```$", "", cleaned.strip())

    try:
        start = cleaned.find("{")
        end   = cleaned.rfind("}") + 1
        if start != -1 and end > start:
            return json.loads(cleaned[start:end])
    except json.JSONDecodeError:
        pass

    return {
        "summary":        cleaned[:500] if cleaned else "No summary generated.",
        "opportunities":  ["Analysis generated — see summary"],
        "threats":        ["Analysis generated — see summary"],
        "positioning":    "See summary above",
        "recommendation": "Review summary for insights",
    }




# ─────────────────────────────────────────────────────────────────────────────
# Main async function — called from app.py
# ─────────────────────────────────────────────────────────────────────────────
async def run_intelligence(companies: dict) -> dict:
    """
    companies: dict of { display_name: ticker_symbol }
    e.g. {"Apple": "AAPL", "Nike": "NKE"}

    Returns a dict with:
      - per_company: list of raw data per company
      - insights:    AI-generated analysis (summary, opportunities, threats, etc.)
    """
    per_company = []

    # ── Step 1: Call MCP server tools for each company ──────────────────────
    async with Client("mcp_server.py") as client:
        for name, ticker in companies.items():
            try:
                stock_result = await client.call_tool("get_stock_data",   {"ticker_symbol": ticker})
                info_result  = await client.call_tool("get_company_info", {"ticker_symbol": ticker})

                # _extract_text handles both old list API and new CallToolResult API
                stock_data = _extract_text(stock_result)
                stock_dict = json.loads(stock_data) if stock_data and stock_data != "{}" else {}

                # Build short text for sentiment scoring
                sentiment_text = (
                    f"{name} price change {stock_dict.get('change_pct', 0):.1f}% "
                    f"revenue growth market cap {stock_dict.get('market_cap', 0)} "
                    f"sector {stock_dict.get('sector', '')} "
                    f"{'profit gain strong' if (stock_dict.get('change_pct') or 0) > 0 else 'decline loss weak'}"
                )
                sent_result = await client.call_tool("analyze_sentiment", {"text": sentiment_text})

                info_data = _extract_text(info_result)
                sent_data = _extract_text(sent_result)

                per_company.append({
                    "name":      name,
                    "ticker":    ticker,
                    "stock":     json.loads(stock_data) if stock_data else {},
                    "info":      json.loads(info_data)  if info_data  else {},
                    "sentiment": json.loads(sent_data)  if sent_data  else {},
                })

            except Exception as e:
                per_company.append({
                    "name": name, "ticker": ticker,
                    "stock": {}, "info": {}, "sentiment": {},
                    "error": repr(e),
                })

    # ── Step 2: Build prompt and call HuggingFace LLM ───────────────────────
    prompt     = build_prompt(per_company)
    raw_output = call_huggingface(prompt)
    insights   = parse_llm_output(raw_output)

    return {
        "per_company": per_company,
        "insights":    insights,
    }
