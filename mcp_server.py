"""
mcp_server.py  —  Strategic Intelligence MCP Server
=====================================================
Exposes tools via FastMCP:
  1. get_stock_data(ticker)       → live financials from yfinance
  2. get_company_info(ticker)     → company profile from yfinance
  3. get_price_history(ticker)    → historical price data
  4. analyze_sentiment(headlines) → rule-based sentiment scoring

No SerpAPI needed — everything uses yfinance.

To run standalone:
    python mcp_server.py
"""

import os
import json
import yfinance as yf
from fastmcp import FastMCP

# ── Create the MCP server ─────────────────────────────────────────────────────
mcp = FastMCP("Strategic Intelligence Server")


# ─────────────────────────────────────────────────────────────────────────────
# TOOL 1 — Live stock price + key financial metrics
# ─────────────────────────────────────────────────────────────────────────────
@mcp.tool()
def get_stock_data(ticker_symbol: str) -> dict:
    """
    Fetches live stock price and key financial metrics for a ticker.
    Returns price, change%, market cap, revenue, P/E, EPS, beta, 52w high/low.
    """
    try:
        tk   = yf.Ticker(ticker_symbol)
        info = tk.info

        price      = info.get("currentPrice") or info.get("regularMarketPrice") or 0
        prev_close = info.get("previousClose") or info.get("regularMarketPreviousClose") or price
        change     = price - prev_close
        change_pct = (change / prev_close * 100) if prev_close else 0

        return {
            "ticker":       ticker_symbol.upper(),
            "name":         info.get("longName", ticker_symbol),
            "price":        round(price, 2),
            "change":       round(change, 2),
            "change_pct":   round(change_pct, 2),
            "market_cap":   info.get("marketCap"),
            "revenue":      info.get("totalRevenue"),
            "net_income":   info.get("netIncomeToCommon"),
            "pe_ratio":     info.get("trailingPE"),
            "eps":          info.get("trailingEps"),
            "beta":         info.get("beta"),
            "week52_high":  info.get("fiftyTwoWeekHigh"),
            "week52_low":   info.get("fiftyTwoWeekLow"),
            "div_yield":    info.get("dividendYield"),
            "sector":       info.get("sector", "N/A"),
            "industry":     info.get("industry", "N/A"),
        }
    except Exception as e:
        return {"ticker": ticker_symbol, "error": str(e)}


# ─────────────────────────────────────────────────────────────────────────────
# TOOL 2 — Company profile / background info
# ─────────────────────────────────────────────────────────────────────────────
@mcp.tool()
def get_company_info(ticker_symbol: str) -> dict:
    """
    Fetches company profile: description, headquarters, employees, website.
    """
    try:
        tk   = yf.Ticker(ticker_symbol)
        info = tk.info
        return {
            "ticker":      ticker_symbol.upper(),
            "name":        info.get("longName", ticker_symbol),
            "description": info.get("longBusinessSummary", "No description available.")[:600],
            "sector":      info.get("sector", "N/A"),
            "industry":    info.get("industry", "N/A"),
            "hq":          f"{info.get('city','')}, {info.get('country','')}".strip(", "),
            "employees":   info.get("fullTimeEmployees"),
            "website":     info.get("website", "#"),
        }
    except Exception as e:
        return {"ticker": ticker_symbol, "error": str(e)}


# ─────────────────────────────────────────────────────────────────────────────
# TOOL 3 — Historical closing prices
# ─────────────────────────────────────────────────────────────────────────────
@mcp.tool()
def get_price_history(ticker_symbol: str, period: str = "3mo") -> dict:
    """
    Returns closing price history for a given period.
    period options: 1mo, 3mo, 6mo, 1y, 2y
    """
    try:
        tk   = yf.Ticker(ticker_symbol)
        hist = tk.history(period=period)
        if hist.empty:
            return {"ticker": ticker_symbol, "dates": [], "prices": []}
        return {
            "ticker":  ticker_symbol.upper(),
            "period":  period,
            "dates":   [str(d.date()) for d in hist.index],
            "prices":  [round(p, 2) for p in hist["Close"].tolist()],
        }
    except Exception as e:
        return {"ticker": ticker_symbol, "error": str(e)}


# ─────────────────────────────────────────────────────────────────────────────
# TOOL 4 — Rule-based sentiment analysis on text
# ─────────────────────────────────────────────────────────────────────────────
@mcp.tool()
def analyze_sentiment(text: str) -> dict:
    """
    Performs simple rule-based sentiment scoring on any text input.
    Returns: score (-1 to +1), label (Positive/Neutral/Negative), confidence.
    """
    positive_words = [
        "growth", "profit", "record", "beat", "surge", "rise", "gain",
        "strong", "bullish", "expand", "innovation", "lead", "success",
        "revenue", "upgrade", "outperform", "opportunity", "launch",
        "partnership", "acquisition", "milestone", "breakthrough"
    ]
    negative_words = [
        "loss", "decline", "fall", "drop", "miss", "weak", "bearish",
        "lawsuit", "fine", "recall", "layoff", "cut", "risk", "threat",
        "concern", "struggle", "debt", "downgrade", "underperform",
        "investigation", "penalty", "crisis", "volatility", "warning"
    ]

    text_lower = text.lower()
    pos_hits = sum(1 for w in positive_words if w in text_lower)
    neg_hits = sum(1 for w in negative_words if w in text_lower)
    total    = pos_hits + neg_hits

    if total == 0:
        return {"score": 0.0, "label": "Neutral", "confidence": 0.5,
                "positive_signals": 0, "negative_signals": 0}

    score      = (pos_hits - neg_hits) / total
    confidence = round(min(total / 10, 1.0), 2)

    if score > 0.1:
        label = "Positive"
    elif score < -0.1:
        label = "Negative"
    else:
        label = "Neutral"

    return {
        "score":             round(score, 2),
        "label":             label,
        "confidence":        confidence,
        "positive_signals":  pos_hits,
        "negative_signals":  neg_hits,
    }


# ─────────────────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    mcp.run()