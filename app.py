"""
app.py  —  Real-Time Industry Insight & Strategic Intelligence System
======================================================================
Streamlit dashboard with two tabs:
  Tab 1 — 📊 Market Dashboard  : live stock data, charts, financials
  Tab 2 — 🧠 AI Intelligence   : MCP-powered competitive analysis via HuggingFace

Run:
    streamlit run app.py
"""

import asyncio
import threading
import json
import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st
import yfinance as yf

from mcp_client import run_intelligence

def run_async(coro):
    """Safely run async coroutine in environments with running event loops (like HuggingFace Spaces)."""
    try:
        loop = asyncio.get_running_loop()
    except RuntimeError:
        loop = None

    if loop and loop.is_running():
        result = []
        error = []
        def _runner():
            try:
                result.append(asyncio.run(coro))
            except Exception as e:
                error.append(e)
        t = threading.Thread(target=_runner)
        t.start()
        t.join()
        if error:
            raise error[0]
        return result[0]
    else:
        return asyncio.run(coro)

# ── PAGE CONFIG ───────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="Strategic Intelligence System",
    page_icon="🧠",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ── CUSTOM CSS ────────────────────────────────────────────────────────────────
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Syne:wght@400;700;800&family=DM+Mono:wght@400;500&display=swap');

html, body, [class*="css"] { font-family: 'DM Mono', monospace; }
.stApp { background-color: #09090f; }

[data-testid="metric-container"], [data-testid="stMetric"] {
    background: #16161f; border: 1px solid #1e1e2e;
    border-radius: 12px; padding: 1rem 1.2rem;
}
[data-testid="metric-container"] label, [data-testid="stMetricLabel"] * {
    color: #a8a8b8 !important; font-size: 0.68rem !important;
    text-transform: uppercase; letter-spacing: 0.1em;
}
[data-testid="metric-container"] [data-testid="stMetricValue"], [data-testid="stMetricValue"], [data-testid="stMetricValue"] * {
    font-family: 'Syne', sans-serif !important; font-size: 1.6rem !important;
    font-weight: 700 !important; color: #ffffff !important;
}
[data-testid="stSidebar"] {
    background: #111118 !important; border-right: 1px solid #1e1e2e;
}
h1, h2, h3 { font-family: 'Syne', sans-serif !important; color: #e8e8f0 !important; }
hr { border-color: #1e1e2e; }

/* Tab styling */
[data-testid="stTabs"] button {
    font-family: 'DM Mono', monospace !important;
    font-size: 0.82rem !important;
    color: #a8a8b8 !important;
}
[data-testid="stTabs"] button[aria-selected="true"] {
    color: #a397ff !important;
    border-bottom: 2px solid #a397ff !important;
}

.section-label {
    font-size: 0.68rem; text-transform: uppercase; letter-spacing: 0.14em;
    color: #a397ff; margin-bottom: 0.5rem; font-weight: 600;
}

/* Company profile card */
.co-card {
    background: #16161f; border: 1px solid #1e1e2e;
    border-radius: 14px; padding: 1.2rem; margin-bottom: 0.8rem;
    transition: border-color 0.2s;
}
.co-card:hover { border-color: rgba(124,106,255,0.4); }
.co-card-name  { font-family:'Syne',sans-serif; font-size:1rem; font-weight:700; color:#ffffff; }
.co-card-type  { font-size:0.66rem; color:#a8a8b8; margin-bottom:0.5rem; }
.co-card-price { font-family:'Syne',sans-serif; font-size:1.35rem; font-weight:700; color:#ffffff; }
.co-card-up      { color:#4fffb0; font-size:0.73rem; }
.co-card-down    { color:#ff4f7b; font-size:0.73rem; }
.co-card-neutral { color:#a8a8b8; font-size:0.73rem; }
.co-meta { font-size:0.68rem; color:#a8a8b8; margin-top:0.55rem; line-height:1.9; }
.co-meta b { color:#ffffff; }

/* Intelligence cards */
.intel-card {
    border-radius: 14px; padding: 1.2rem; margin-bottom: 0.7rem;
}
.intel-opportunity {
    background: rgba(79,255,176,0.06); border: 1px solid rgba(79,255,176,0.2);
}
.intel-threat {
    background: rgba(255,79,123,0.06); border: 1px solid rgba(255,79,123,0.2);
}
.intel-card-title {
    font-family:'Syne',sans-serif; font-size:0.78rem; font-weight:700;
    text-transform:uppercase; letter-spacing:0.1em; margin-bottom:0.5rem;
}
.intel-opportunity .intel-card-title { color:#4fffb0; }
.intel-threat      .intel-card-title { color:#ff4f7b; }
.intel-card-text { font-size:0.78rem; color:#e0e0e0; line-height:1.8; }

/* Sentiment badge */
.badge-positive { background:rgba(79,255,176,0.15); color:#4fffb0; border:1px solid rgba(79,255,176,0.3); padding:3px 10px; border-radius:999px; font-size:0.65rem; }
.badge-negative { background:rgba(255,79,123,0.15); color:#ff4f7b; border:1px solid rgba(255,79,123,0.3); padding:3px 10px; border-radius:999px; font-size:0.65rem; }
.badge-neutral  { background:rgba(107,107,133,0.25); color:#a8a8b8; border:1px solid rgba(168,168,184,0.4); padding:3px 10px; border-radius:999px; font-size:0.65rem; }

/* Summary box */
.summary-box {
    background: rgba(124,106,255,0.07); border: 1px solid rgba(124,106,255,0.25);
    border-radius: 14px; padding: 1.3rem 1.5rem; margin-bottom: 1.2rem;
    font-size: 0.82rem; color: #e0e0e0; line-height: 1.9;
}
.summary-box-title {
    font-family:'Syne',sans-serif; font-size:0.78rem; font-weight:700;
    color:#a397ff; text-transform:uppercase; letter-spacing:0.1em; margin-bottom:0.5rem;
}

/* Recommendation box */
.rec-box {
    background: rgba(255,192,106,0.07); border: 1px solid rgba(255,192,106,0.25);
    border-radius: 14px; padding: 1.3rem 1.5rem; margin-top: 1rem;
    font-size: 0.82rem; color: #c8c8d8; line-height: 1.9;
}
.rec-box-title {
    font-family:'Syne',sans-serif; font-size:0.78rem; font-weight:700;
    color:#ffc06a; text-transform:uppercase; letter-spacing:0.1em; margin-bottom:0.5rem;
}
</style>
""", unsafe_allow_html=True)

# ── CONSTANTS ─────────────────────────────────────────────────────────────────
COLORS = ["#7C6AFF","#FF6A9E","#6AFFCA","#FFC06A","#6AB4FF",
          "#FF9F6A","#6AFFFF","#FF6AFF","#AFFF6A","#FF6A6A"]

ALL_COMPANIES = {
    "Apple":     "AAPL",
    "Microsoft": "MSFT",
    "Google":    "GOOGL",
    "Amazon":    "AMZN",
    "Tesla":     "TSLA",
    "Nike":      "NKE",
    "Puma":      "PUM.DE",
    "Adidas":    "ADDYY",
    "Reebok":    "ADDYY",   # Reebok is owned by Authentic Brands (private); using Adidas proxy
}

PERIOD_MAP = {
    "1 Month":  "1mo",
    "3 Months": "3mo",
    "6 Months": "6mo",
    "1 Year":   "1y",
    "2 Years":  "2y",
}

# ── HELPERS ───────────────────────────────────────────────────────────────────
def fmt_big(n):
    if n is None or (isinstance(n, float) and np.isnan(n)): return "N/A"
    if n >= 1e12: return f"${n/1e12:.2f}T"
    if n >= 1e9:  return f"${n/1e9:.2f}B"
    if n >= 1e6:  return f"${n/1e6:.2f}M"
    return f"${n:,.0f}"

def fmt_num(n):
    if n is None or (isinstance(n, float) and np.isnan(n)): return "N/A"
    return f"{int(n):,}"

def pct_class(val):
    if val is None: return "neutral"
    return "up" if val >= 0 else "down"

def arrow(val):
    if val is None: return "–"
    return "▲" if val >= 0 else "▼"

# ── YFINANCE FETCH ────────────────────────────────────────────────────────────
@st.cache_data(ttl=900, show_spinner=False)
def fetch_ticker(symbol: str, period: str):
    tk   = yf.Ticker(symbol)
    info = tk.info
    hist = tk.history(period=period)
    return info, hist

def load_company(name: str, symbol: str, period: str):
    info, hist = fetch_ticker(symbol, period)
    price      = info.get("currentPrice") or info.get("regularMarketPrice") or 0
    prev_close = info.get("previousClose") or info.get("regularMarketPreviousClose") or price
    change     = price - prev_close
    change_pct = (change / prev_close * 100) if prev_close else 0
    return {
        "name":        info.get("longName", name),
        "symbol":      symbol,
        "sector":      info.get("sector", "N/A"),
        "industry":    info.get("industry", "N/A"),
        "description": info.get("longBusinessSummary", "No description.")[:500],
        "hq":          f"{info.get('city','')}, {info.get('country','')}".strip(", "),
        "employees":   info.get("fullTimeEmployees"),
        "market_cap":  info.get("marketCap"),
        "revenue":     info.get("totalRevenue"),
        "net_income":  info.get("netIncomeToCommon"),
        "pe_ratio":    info.get("trailingPE"),
        "eps":         info.get("trailingEps"),
        "div_yield":   info.get("dividendYield"),
        "week52_high": info.get("fiftyTwoWeekHigh"),
        "week52_low":  info.get("fiftyTwoWeekLow"),
        "beta":        info.get("beta"),
        "website":     info.get("website", "#"),
        "price":       price,
        "change":      change,
        "change_pct":  change_pct,
        "history":     hist,
    }

# ── SIDEBAR ───────────────────────────────────────────────────────────────────
with st.sidebar:
    st.markdown("## 🧠 Intel System")
    st.markdown("<div class='section-label'>Companies</div>", unsafe_allow_html=True)

    selected_names = st.multiselect(
        "Track companies",
        options=list(ALL_COMPANIES.keys()),
        default=list(ALL_COMPANIES.keys()),
    )

    st.markdown("<div class='section-label' style='margin-top:1rem'>Add Custom</div>",
                unsafe_allow_html=True)
    c1, c2 = st.columns(2)
    custom_sym  = c1.text_input("Ticker", placeholder="NVDA").upper().strip()
    custom_name = c2.text_input("Name",   placeholder="Nvidia").strip()
    if custom_sym and custom_name:
        ALL_COMPANIES[custom_name] = custom_sym
        if custom_name not in selected_names:
            selected_names.append(custom_name)

    st.markdown("<div class='section-label' style='margin-top:1rem'>Chart Period</div>",
                unsafe_allow_html=True)
    period_label = st.selectbox("History window", list(PERIOD_MAP.keys()), index=2)
    period = PERIOD_MAP[period_label]

    st.markdown("---")
    if st.button("🔄  Refresh Market Data", use_container_width=True):
        st.cache_data.clear()
        st.rerun()

    st.markdown(
        "<div style='font-size:0.65rem;color:#a8a8b8;line-height:1.9;margin-top:0.5rem'>"
        "Market data: <b style='color:#ffffff'>yfinance</b><br>"
        "AI insights: <b style='color:#ffffff'>HuggingFace</b><br>"
        "MCP tools: <b style='color:#ffffff'>FastMCP</b><br>"
        "Cache TTL: 15 minutes"
        "</div>", unsafe_allow_html=True,
    )

# ── HEADER ────────────────────────────────────────────────────────────────────
st.markdown(
    "<h1 style='font-family:Syne,sans-serif;font-size:1.9rem;margin-bottom:0.15rem'>"
    "🧠 Real-Time Industry Insight & Strategic Intelligence</h1>"
    "<div style='color:#a8a8b8;font-size:0.73rem;margin-bottom:1.2rem'>"
    "Live market data · AI-powered competitive analysis · MCP tool architecture"
    "</div>",
    unsafe_allow_html=True,
)

if not selected_names:
    st.warning("Select at least one company from the sidebar.")
    st.stop()

# ── FETCH MARKET DATA ─────────────────────────────────────────────────────────
companies_data = {}
bar = st.progress(0, text="Loading market data…")
for i, name in enumerate(selected_names):
    sym = ALL_COMPANIES.get(name, name)
    try:
        companies_data[name] = load_company(name, sym, period)
    except Exception as e:
        st.warning(f"Could not load **{name}**: {e}")
    bar.progress((i + 1) / len(selected_names), text=f"Fetching {name}…")
bar.empty()

data = list(companies_data.values())
if not data:
    st.error("No market data loaded.")
    st.stop()

# ── TABS ──────────────────────────────────────────────────────────────────────
tab1, tab2 = st.tabs(["📊  Market Dashboard", "🧠  AI Intelligence"])


# ══════════════════════════════════════════════════════════════════════════════
# TAB 1 — MARKET DASHBOARD
# ══════════════════════════════════════════════════════════════════════════════
with tab1:

    # KPI Strip
    st.markdown("### Overview")
    k1, k2, k3, k4, k5 = st.columns(5)
    k1.metric("Companies",        len(data))
    k2.metric("Total Mkt Cap",    fmt_big(sum(d["market_cap"] or 0 for d in data)))
    k3.metric("Total Revenue",    fmt_big(sum(d["revenue"]    or 0 for d in data)))
    k4.metric("Total Employees",  fmt_num(sum(d["employees"]  or 0 for d in data)))
    k5.metric("Gainers Today",    f"{sum(1 for d in data if d['change_pct'] > 0)}/{len(data)}")
    st.markdown("---")

    # Company Cards
    st.markdown("### Company Profiles")
    num_cols = min(len(data), 3)
    cols = st.columns(num_cols)
    for i, d in enumerate(data):
        cls       = pct_class(d["change_pct"])
        price_str = f"${d['price']:.2f}" if d["price"] else "N/A"
        chg_str   = (
            f"{arrow(d['change'])} ${abs(d['change']):.2f} ({abs(d['change_pct']):.2f}%)"
            if d["change"] is not None else "N/A"
        )
        with cols[i % num_cols]:
            st.markdown(f"""
            <div class='co-card'>
              <div class='co-card-name'>{d['name']}
                <span style='color:#6b6b85;font-size:.75rem'> ({d['symbol']})</span>
              </div>
              <div class='co-card-type'>{d['sector']} · {d['industry']}</div>
              <div class='co-card-price'>{price_str}</div>
              <div class='co-card-{cls}'>{chg_str}</div>
              <div class='co-meta'>
                <b>HQ:</b> {d['hq']}<br>
                <b>Employees:</b> {fmt_num(d['employees'])}<br>
                <b>Market Cap:</b> {fmt_big(d['market_cap'])}<br>
                <b>P/E:</b> {f"{d['pe_ratio']:.1f}" if d['pe_ratio'] else 'N/A'} &nbsp;
                <b>Beta:</b> {f"{d['beta']:.2f}" if d['beta'] else 'N/A'}
              </div>
            </div>""", unsafe_allow_html=True)

    st.markdown("---")

    # Price History
    st.markdown("### Stock Price History")
    st.markdown(f"<div class='section-label'>Closing prices — {period_label}</div>",
                unsafe_allow_html=True)
    fig_hist = go.Figure()
    for i, d in enumerate(data):
        hist = d["history"]
        if hist is not None and not hist.empty:
            fig_hist.add_trace(go.Scatter(
                x=hist.index, y=hist["Close"],
                name=d["symbol"], mode="lines",
                line=dict(color=COLORS[i % len(COLORS)], width=2),
                hovertemplate=f"<b>{d['name']}</b><br>%{{x|%b %d, %Y}}<br>${{y:.2f}}<extra></extra>",
            ))
    fig_hist.update_layout(
        paper_bgcolor="#16161f", plot_bgcolor="#16161f", font_color="#6b6b85",
        hovermode="x unified",
        legend=dict(orientation="h", y=-0.18, font=dict(size=11)),
        margin=dict(t=10, b=30, l=10, r=10), height=380,
        xaxis=dict(showgrid=False, color="#6b6b85"),
        yaxis=dict(showgrid=True, gridcolor="#1e1e2e", tickprefix="$", color="#6b6b85"),
    )
    st.plotly_chart(fig_hist, use_container_width=True)

    # Financials row
    st.markdown("### Financials")
    names       = [d["name"].split()[0] for d in data]
    mktcaps     = [(d["market_cap"]  or 0)/1e9 for d in data]
    revenues    = [(d["revenue"]     or 0)/1e9 for d in data]
    net_incomes = [(d["net_income"]  or 0)/1e9 for d in data]
    employees   = [(d["employees"]   or 0)/1e3 for d in data]
    pe_vals     = [ d["pe_ratio"]    or 0      for d in data]

    f1, f2 = st.columns(2)
    with f1:
        st.markdown("<div class='section-label'>Market Cap (B USD)</div>", unsafe_allow_html=True)
        fig_mc = px.bar(x=names, y=mktcaps, color=names,
                        color_discrete_sequence=COLORS,
                        text=[f"${v:.1f}B" for v in mktcaps])
        fig_mc.update_traces(textposition="outside", marker_line_width=0)
        fig_mc.update_layout(paper_bgcolor="#16161f", plot_bgcolor="#16161f",
                             font_color="#6b6b85", showlegend=False, height=300,
                             margin=dict(t=20,b=10,l=10,r=10),
                             xaxis=dict(showgrid=False, tickangle=-30),
                             yaxis=dict(showgrid=True, gridcolor="#1e1e2e", ticksuffix="B"))
        st.plotly_chart(fig_mc, use_container_width=True)

    with f2:
        st.markdown("<div class='section-label'>Revenue vs Net Income (B USD)</div>",
                    unsafe_allow_html=True)
        fig_fn = go.Figure([
            go.Bar(name="Revenue",    x=names, y=revenues,
                   marker_color=COLORS[0], text=[f"${v:.1f}B" for v in revenues],
                   textposition="outside"),
            go.Bar(name="Net Income", x=names, y=net_incomes,
                   marker_color=COLORS[1], text=[f"${v:.1f}B" for v in net_incomes],
                   textposition="outside"),
        ])
        fig_fn.update_layout(barmode="group", paper_bgcolor="#16161f", plot_bgcolor="#16161f",
                             font_color="#6b6b85", height=300,
                             legend=dict(orientation="h", y=-0.3, font=dict(size=11)),
                             margin=dict(t=20,b=30,l=10,r=10),
                             xaxis=dict(showgrid=False, tickangle=-30),
                             yaxis=dict(showgrid=True, gridcolor="#1e1e2e", ticksuffix="B"))
        st.plotly_chart(fig_fn, use_container_width=True)

    f3, f4 = st.columns(2)
    with f3:
        st.markdown("<div class='section-label'>Workforce (Thousands)</div>",
                    unsafe_allow_html=True)
        fig_emp = px.bar(y=names, x=employees, orientation="h",
                         color=names, color_discrete_sequence=COLORS[::-1],
                         text=[f"{v:.0f}K" for v in employees])
        fig_emp.update_traces(textposition="outside", marker_line_width=0)
        fig_emp.update_layout(paper_bgcolor="#16161f", plot_bgcolor="#16161f",
                              font_color="#6b6b85", showlegend=False, height=300,
                              margin=dict(t=10,b=10,l=10,r=10),
                              xaxis=dict(showgrid=True, gridcolor="#1e1e2e", ticksuffix="K"),
                              yaxis=dict(showgrid=False))
        st.plotly_chart(fig_emp, use_container_width=True)

    with f4:
        st.markdown("<div class='section-label'>P/E Ratio</div>", unsafe_allow_html=True)
        fig_pe = px.bar(x=names, y=pe_vals, color=names,
                        color_discrete_sequence=COLORS,
                        text=[f"{v:.1f}x" if v else "N/A" for v in pe_vals])
        fig_pe.update_traces(textposition="outside", marker_line_width=0)
        fig_pe.update_layout(paper_bgcolor="#16161f", plot_bgcolor="#16161f",
                             font_color="#6b6b85", showlegend=False, height=300,
                             margin=dict(t=20,b=10,l=10,r=10),
                             xaxis=dict(showgrid=False, tickangle=-30),
                             yaxis=dict(showgrid=True, gridcolor="#1e1e2e", ticksuffix="x"))
        st.plotly_chart(fig_pe, use_container_width=True)

    # Scatter + Donut
    sc_col, do_col = st.columns(2)
    with sc_col:
        st.markdown("<div class='section-label'>Market Cap vs Revenue</div>",
                    unsafe_allow_html=True)
        fig_sc = px.scatter(x=revenues, y=mktcaps, text=names,
                            size=[max(v, 0.1) for v in mktcaps],
                            color=names, color_discrete_sequence=COLORS,
                            labels={"x":"Revenue (B USD)","y":"Market Cap (B USD)"})
        fig_sc.update_traces(textposition="top center", marker_line_width=0)
        fig_sc.update_layout(paper_bgcolor="#16161f", plot_bgcolor="#16161f",
                             font_color="#6b6b85", showlegend=False, height=320,
                             margin=dict(t=10,b=10,l=10,r=10),
                             xaxis=dict(showgrid=True, gridcolor="#1e1e2e", ticksuffix="B"),
                             yaxis=dict(showgrid=True, gridcolor="#1e1e2e", ticksuffix="B"))
        st.plotly_chart(fig_sc, use_container_width=True)

    with do_col:
        st.markdown("<div class='section-label'>Sector Distribution</div>",
                    unsafe_allow_html=True)
        sc = {}
        for d in data:
            sc[d["sector"]] = sc.get(d["sector"], 0) + 1
        fig_dn = px.pie(names=list(sc.keys()), values=list(sc.values()),
                        hole=0.6, color_discrete_sequence=COLORS)
        fig_dn.update_traces(textinfo="label+percent", showlegend=False)
        fig_dn.update_layout(paper_bgcolor="#16161f", font_color="#6b6b85",
                             margin=dict(t=10,b=10,l=10,r=10), height=320)
        st.plotly_chart(fig_dn, use_container_width=True)

    st.markdown("---")

    # Data Table
    st.markdown("### Full Data Table")
    df = pd.DataFrame([{
        "Company":    d["name"],
        "Symbol":     d["symbol"],
        "Sector":     d["sector"],
        "Price":      f"${d['price']:.2f}"        if d["price"]      else "N/A",
        "Change %":   f"{d['change_pct']:+.2f}%"  if d["change_pct"] else "N/A",
        "Market Cap": fmt_big(d["market_cap"]),
        "Revenue":    fmt_big(d["revenue"]),
        "Net Income": fmt_big(d["net_income"]),
        "Employees":  fmt_num(d["employees"]),
        "P/E":        f"{d['pe_ratio']:.1f}"      if d["pe_ratio"]   else "N/A",
        "Beta":       f"{d['beta']:.2f}"          if d["beta"]       else "N/A",
        "52W High":   f"${d['week52_high']:.2f}"  if d["week52_high"] else "N/A",
        "52W Low":    f"${d['week52_low']:.2f}"   if d["week52_low"]  else "N/A",
        "Website":    d["website"],
    } for d in data])
    st.dataframe(df, use_container_width=True, hide_index=True,
                 column_config={
                     "Website": st.column_config.LinkColumn("Website"),
                     "Company": st.column_config.TextColumn("Company", width="medium"),
                 })

    # Descriptions
    st.markdown("---")
    st.markdown("### Company Descriptions")
    for d in data:
        with st.expander(f"📄  {d['name']} ({d['symbol']})"):
            st.markdown(
                f"<div style='color:#e0e0e0;font-size:0.8rem;line-height:1.9'>"
                f"{d['description']}</div>", unsafe_allow_html=True,
            )
            cm = st.columns(4)
            cm[0].metric("HQ",         d["hq"])
            cm[1].metric("Industry",   d["industry"])
            cm[2].metric("Div. Yield", f"{d['div_yield']*100:.2f}%" if d["div_yield"] else "N/A")
            cm[3].metric("EPS (TTM)",  f"${d['eps']:.2f}"           if d["eps"]       else "N/A")


# ══════════════════════════════════════════════════════════════════════════════
# TAB 2 — AI INTELLIGENCE
# ══════════════════════════════════════════════════════════════════════════════
with tab2:
    st.markdown("### 🧠 AI-Powered Competitive Intelligence")
    st.markdown(
        "<div style='color:#a8a8b8;font-size:0.75rem;margin-bottom:1.2rem'>"
        "Uses FastMCP tools to gather data, then sends it to a HuggingFace LLM "
        "to generate strategic opportunities, threats, and recommendations."
        "</div>", unsafe_allow_html=True,
    )

    # ── Analyze button ────────────────────────────────────────────────────────
    col_btn, col_info = st.columns([1, 3])
    with col_btn:
        run_btn = st.button("⚡  Run Intelligence Analysis",
                            use_container_width=True, type="primary")
    with col_info:
        st.markdown(
            "<div style='padding-top:0.6rem;font-size:0.72rem;color:#a8a8b8'>"
            f"Will analyze: <b style='color:#ffffff'>"
            f"{', '.join(selected_names[:5])}{'...' if len(selected_names)>5 else ''}"
            f"</b> via MCP → HuggingFace pipeline"
            "</div>", unsafe_allow_html=True,
        )

    st.markdown("---")

    # ── Run analysis ──────────────────────────────────────────────────────────
    if run_btn:
        with st.spinner("🔧 MCP tools fetching data… then sending to HuggingFace LLM…"):
            try:
                companies_for_mcp = {
                    name: ALL_COMPANIES.get(name, name)
                    for name in selected_names
                }
                result = run_async(run_intelligence(companies_for_mcp))
                st.session_state["intel_result"] = result
            except Exception as e:
                st.error(f"Analysis failed: {e}")
                st.session_state["intel_result"] = None

    # ── Display results ───────────────────────────────────────────────────────
    result = st.session_state.get("intel_result")

    if result is None:
        st.markdown(
            "<div style='text-align:center;padding:3rem;color:#a8a8b8;font-size:0.82rem'>"
            "Click <b style='color:#a397ff'>⚡ Run Intelligence Analysis</b> above "
            "to generate AI-powered insights for the selected companies."
            "</div>", unsafe_allow_html=True,
        )
    else:
        insights    = result.get("insights", {})
        per_company = result.get("per_company", [])

        # ── Summary box ──────────────────────────────────────────────────────
        summary = insights.get("summary", "")
        if summary:
            st.markdown(f"""
            <div class='summary-box'>
                <div class='summary-box-title'>📋 Market Summary</div>
                {summary}
            </div>""", unsafe_allow_html=True)

        # ── Opportunities + Threats side by side ─────────────────────────────
        opp_col, thr_col = st.columns(2)

        with opp_col:
            st.markdown(
                "<div class='section-label' style='color:#4fffb0'>✅ Opportunities</div>",
                unsafe_allow_html=True,
            )
            opportunities = insights.get("opportunities", [])
            if opportunities:
                for opp in opportunities:
                    st.markdown(f"""
                    <div class='intel-card intel-opportunity'>
                        <div class='intel-card-title'>📈 Opportunity</div>
                        <div class='intel-card-text'>{opp}</div>
                    </div>""", unsafe_allow_html=True)
            else:
                st.info("No opportunities identified.")

        with thr_col:
            st.markdown(
                "<div class='section-label' style='color:#ff4f7b'>⚠️ Threats</div>",
                unsafe_allow_html=True,
            )
            threats = insights.get("threats", [])
            if threats:
                for thr in threats:
                    st.markdown(f"""
                    <div class='intel-card intel-threat'>
                        <div class='intel-card-title'>🚨 Threat</div>
                        <div class='intel-card-text'>{thr}</div>
                    </div>""", unsafe_allow_html=True)
            else:
                st.info("No threats identified.")

        # ── Positioning + Recommendation ─────────────────────────────────────
        pos_col, rec_col = st.columns(2)

        with pos_col:
            positioning = insights.get("positioning", "")
            if positioning:
                st.markdown(f"""
                <div class='summary-box' style='margin-top:1rem'>
                    <div class='summary-box-title'>🏆 Competitive Positioning</div>
                    {positioning}
                </div>""", unsafe_allow_html=True)

        with rec_col:
            recommendation = insights.get("recommendation", "")
            if recommendation:
                st.markdown(f"""
                <div class='rec-box' style='margin-top:1rem'>
                    <div class='rec-box-title'>💡 Strategic Recommendation</div>
                    {recommendation}
                </div>""", unsafe_allow_html=True)

        st.markdown("---")

        # ── Per-Company Sentiment Analysis ────────────────────────────────────
        st.markdown("### Per-Company Sentiment & Data Quality")

        valid = [c for c in per_company if not c.get("error")]
        if valid:
            sent_cols = st.columns(min(len(valid), 4))
            for i, co in enumerate(valid):
                sent  = co.get("sentiment", {})
                stock = co.get("stock", {})
                label = sent.get("label", "Neutral")
                score = sent.get("score", 0)
                
                if score >= 0.6:
                    outlook = "Strongly Bullish 🚀"
                elif score >= 0.2:
                    outlook = "Slightly Optimistic 📈"
                elif score > -0.2:
                    outlook = "Mixed / Neutral ⚖️"
                elif score > -0.6:
                    outlook = "Slightly Cautious 📉"
                else:
                    outlook = "Strongly Bearish ⚠️"

                badge_cls = f"badge-{label.lower()}"
                color_map = {"Positive": "#4fffb0", "Negative": "#ff4f7b", "Neutral": "#6b6b85"}
                bar_color = color_map.get(label, "#6b6b85")
                bar_pct   = int((score + 1) / 2 * 100)

                with sent_cols[i % 4]:
                    st.markdown(f"""
                    <div class='co-card' style='margin-bottom:0.6rem'>
                        <div class='co-card-name' style='font-size:0.88rem'>{co['name']}</div>
                        <div style='margin:0.4rem 0'>
                            <span class='{badge_cls}'>{label}</span>
                        </div>
                        <div style='font-size:0.75rem;color:#e8e8f0;margin-top:0.6rem;font-weight:600'>
                            Outlook: <span style='color:{bar_color}'>{outlook}</span>
                        </div>
                        <div style='background:#1e1e2e;border-radius:999px;height:5px;margin-top:0.5rem;overflow:hidden'>
                            <div style='background:{bar_color};width:{bar_pct}%;height:100%;border-radius:999px'></div>
                        </div>
                        <div style='font-size:0.63rem;color:#6b6b85;margin-top:0.4rem'>
                            Detected <b>{sent.get('positive_signals',0)}</b> growth indicators vs <b>{sent.get('negative_signals',0)}</b> risk factors.
                        </div>
                    </div>""", unsafe_allow_html=True)

        # ── Strategic Matrix ──────────────────────────────────────────────────
        if valid:
            st.markdown("---")
            st.markdown("<div class='section-label'>Strategic Matrix: AI Sentiment vs Market Momentum</div>",
                        unsafe_allow_html=True)

            scatter_names = [c["name"].split()[0] for c in valid]
            scatter_x = [c.get("sentiment", {}).get("score", 0) for c in valid]
            scatter_y = [c.get("stock", {}).get("change_pct", 0) or 0 for c in valid]
            
            # Bubble size based on market cap (minimum size 10)
            scatter_size = [max((c.get("stock", {}).get("market_cap", 0) or 0) / 1e9, 10) for c in valid]

            fig_matrix = px.scatter(
                x=scatter_x, 
                y=scatter_y, 
                text=scatter_names,
                size=scatter_size,
                color=scatter_names,
                color_discrete_sequence=COLORS,
            )

            fig_matrix.update_traces(
                textposition="top center", 
                marker_line_width=1, 
                marker_line_color="#1e1e2e",
                hovertemplate="<b>%{text}</b><br>Sentiment: %{x:+.2f}<br>Momentum: %{y:+.2f}%<extra></extra>"
            )
            
            # Quadrant lines
            fig_matrix.add_hline(y=0, line_dash="dash", line_color="#3a3a50", opacity=0.5)
            fig_matrix.add_vline(x=0, line_dash="dash", line_color="#3a3a50", opacity=0.5)

            # Quadrant Annotations
            fig_matrix.add_annotation(x=0.98, y=0.98, xref="paper", yref="paper", text="Momentum Leaders 🚀", showarrow=False, font=dict(color="#4fffb0", size=11), xanchor="right", yanchor="top", opacity=0.7)
            fig_matrix.add_annotation(x=0.98, y=0.04, xref="paper", yref="paper", text="Value / Recovery 💎", showarrow=False, font=dict(color="#6ab4ff", size=11), xanchor="right", yanchor="bottom", opacity=0.7)
            fig_matrix.add_annotation(x=0.02, y=0.98, xref="paper", yref="paper", text="High Risk / Overvalued ⚠️", showarrow=False, font=dict(color="#ffc06a", size=11), xanchor="left", yanchor="top", opacity=0.7)
            fig_matrix.add_annotation(x=0.02, y=0.04, xref="paper", yref="paper", text="Underperforming 📉", showarrow=False, font=dict(color="#ff4f7b", size=11), xanchor="left", yanchor="bottom", opacity=0.7)

            # Dynamically adjust y-axis range to ensure bubbles fit
            max_abs_y = max([abs(y) for y in scatter_y] + [0.5]) * 1.5 

            fig_matrix.update_layout(
                paper_bgcolor="#16161f", plot_bgcolor="#16161f",
                font_color="#6b6b85", showlegend=False,
                margin=dict(t=30, b=30, l=10, r=10), height=420,
                xaxis=dict(showgrid=False, gridcolor="#1e1e2e", range=[-1.2, 1.2], zeroline=False, title="⬅️ Bearish AI Sentiment &nbsp; &nbsp; | &nbsp; &nbsp; Bullish AI Sentiment ➡️"),
                yaxis=dict(showgrid=True, gridcolor="#1e1e2e", zeroline=False, ticksuffix="%", range=[-max_abs_y, max_abs_y], title="Today's Market Momentum"),
            )
            st.plotly_chart(fig_matrix, use_container_width=True)

        # ── Raw JSON expander ─────────────────────────────────────────────────
        with st.expander("🔍 View Raw MCP + LLM Output"):
            st.json(result)

# ── FOOTER ────────────────────────────────────────────────────────────────────
st.markdown("---")
st.markdown(
    "<div style='text-align:center;color:#a8a8b8;font-size:0.67rem;padding:0.8rem 0'>"
    "Strategic Intelligence System · yfinance · HuggingFace · FastMCP · Streamlit"
    "</div>", unsafe_allow_html=True,
)





















# """
# app.py  –  Company Intelligence Dashboard
# ==========================================
# Streamlit app that fetches live company & stock data via yfinance
# (completely free, no API key required).

# Usage:
#     pip install -r requirements.txt
#     streamlit run app.py
# """

# import numpy as np
# import pandas as pd
# import plotly.express as px
# import plotly.graph_objects as go
# import streamlit as st
# import yfinance as yf

# # ── PAGE CONFIG ───────────────────────────────────────────────────────────────
# st.set_page_config(
#     page_title="Market Intel Dashboard",
#     page_icon="📡",
#     layout="wide",
#     initial_sidebar_state="expanded",
# )

# # ── CUSTOM CSS ────────────────────────────────────────────────────────────────
# st.markdown("""
# <style>
# @import url('https://fonts.googleapis.com/css2?family=Syne:wght@400;700;800&family=DM+Mono:wght@400;500&display=swap');

# html, body, [class*="css"] { font-family: 'DM Mono', monospace; }
# .stApp { background-color: #09090f; }

# [data-testid="metric-container"] {
#     background: #16161f; border: 1px solid #1e1e2e;
#     border-radius: 12px; padding: 1rem 1.2rem;
# }
# [data-testid="metric-container"] label {
#     color: #6b6b85 !important; font-size: 0.68rem !important;
#     text-transform: uppercase; letter-spacing: 0.1em;
# }
# [data-testid="metric-container"] [data-testid="stMetricValue"] {
#     font-family: 'Syne', sans-serif !important; font-size: 1.8rem !important;
#     font-weight: 700 !important; color: #e8e8f0 !important;
# }
# [data-testid="stSidebar"] { background: #111118 !important; border-right: 1px solid #1e1e2e; }
# h1, h2, h3 { font-family: 'Syne', sans-serif !important; color: #e8e8f0 !important; }
# hr { border-color: #1e1e2e; }

# .section-label {
#     font-size: 0.68rem; text-transform: uppercase; letter-spacing: 0.14em;
#     color: #7c6aff; margin-bottom: 0.5rem; font-weight: 600;
# }
# .co-card {
#     background: #16161f; border: 1px solid #1e1e2e;
#     border-radius: 14px; padding: 1.2rem; margin-bottom: 0.8rem;
# }
# .co-card-name  { font-family:'Syne',sans-serif; font-size:1.05rem; font-weight:700; color:#e8e8f0; }
# .co-card-type  { font-size:0.68rem; color:#6b6b85; margin-bottom:0.6rem; }
# .co-card-price { font-family:'Syne',sans-serif; font-size:1.4rem; font-weight:700; color:#e8e8f0; }
# .co-card-up      { color:#4fffb0; font-size:0.75rem; }
# .co-card-down    { color:#ff4f7b; font-size:0.75rem; }
# .co-card-neutral { color:#6b6b85; font-size:0.75rem; }
# .co-meta { font-size:0.7rem; color:#6b6b85; margin-top:0.6rem; line-height:1.9; }
# .co-meta b { color:#c8c8d8; }
# </style>
# """, unsafe_allow_html=True)

# # ── CONSTANTS ─────────────────────────────────────────────────────────────────
# COLORS = ["#7C6AFF","#FF6A9E","#6AFFCA","#FFC06A","#6AB4FF",
#           "#FF9F6A","#6AFFFF","#FF6AFF","#AFFF6A","#FF6A6A"]

# DEFAULT_TICKERS = {
#     "Apple":     "AAPL",
#     "Microsoft": "MSFT",
#     "Google":    "GOOGL",
#     "Amazon":    "AMZN",
#     "Tesla":     "TSLA",
# }

# PERIOD_MAP = {
#     "1 Month":  "1mo",
#     "3 Months": "3mo",
#     "6 Months": "6mo",
#     "1 Year":   "1y",
#     "2 Years":  "2y",
# }

# # ── HELPERS ───────────────────────────────────────────────────────────────────
# def fmt_big(n):
#     if n is None or (isinstance(n, float) and np.isnan(n)):
#         return "N/A"
#     if n >= 1e12: return f"${n/1e12:.2f}T"
#     if n >= 1e9:  return f"${n/1e9:.2f}B"
#     if n >= 1e6:  return f"${n/1e6:.2f}M"
#     return f"${n:,.0f}"

# def fmt_num(n):
#     if n is None or (isinstance(n, float) and np.isnan(n)):
#         return "N/A"
#     return f"{int(n):,}"

# def pct_class(val):
#     if val is None: return "neutral"
#     return "up" if val >= 0 else "down"

# def arrow(val):
#     if val is None: return "–"
#     return "▲" if val >= 0 else "▼"

# # ── FETCH ─────────────────────────────────────────────────────────────────────
# @st.cache_data(ttl=900, show_spinner=False)
# def fetch_ticker(symbol: str, period: str):
#     tk   = yf.Ticker(symbol)
#     info = tk.info
#     hist = tk.history(period=period)
#     return info, hist

# def load_company(name: str, symbol: str, period: str):
#     info, hist = fetch_ticker(symbol, period)
#     price      = info.get("currentPrice") or info.get("regularMarketPrice")
#     prev_close = info.get("previousClose") or info.get("regularMarketPreviousClose")
#     change     = (price - prev_close) if (price and prev_close) else None
#     change_pct = (change / prev_close * 100) if (change and prev_close) else None

#     return {
#         "name":        info.get("longName", name),
#         "symbol":      symbol,
#         "sector":      info.get("sector", "Technology"),
#         "industry":    info.get("industry", "N/A"),
#         "description": info.get("longBusinessSummary", "No description available."),
#         "hq":          f"{info.get('city','')}, {info.get('country','')}".strip(", "),
#         "employees":   info.get("fullTimeEmployees"),
#         "market_cap":  info.get("marketCap"),
#         "revenue":     info.get("totalRevenue"),
#         "net_income":  info.get("netIncomeToCommon"),
#         "pe_ratio":    info.get("trailingPE"),
#         "eps":         info.get("trailingEps"),
#         "div_yield":   info.get("dividendYield"),
#         "week52_high": info.get("fiftyTwoWeekHigh"),
#         "week52_low":  info.get("fiftyTwoWeekLow"),
#         "beta":        info.get("beta"),
#         "website":     info.get("website", "#"),
#         "price":       price,
#         "change":      change,
#         "change_pct":  change_pct,
#         "history":     hist,
#     }

# # ── SIDEBAR ───────────────────────────────────────────────────────────────────
# with st.sidebar:
#     st.markdown("## 📡 Market Intel")
#     st.markdown("<div class='section-label'>Companies</div>", unsafe_allow_html=True)

#     selected = st.multiselect(
#         "Select companies",
#         options=list(DEFAULT_TICKERS.keys()),
#         default=list(DEFAULT_TICKERS.keys()),
#     )

#     st.markdown("<div class='section-label' style='margin-top:1rem'>Add Custom Ticker</div>",
#                 unsafe_allow_html=True)
#     c1, c2 = st.columns(2)
#     custom_sym  = c1.text_input("Symbol",  placeholder="NVDA").upper().strip()
#     custom_name = c2.text_input("Name",    placeholder="Nvidia").strip()
#     if custom_sym and custom_name:
#         DEFAULT_TICKERS[custom_name] = custom_sym
#         if custom_name not in selected:
#             selected.append(custom_name)

#     st.markdown("<div class='section-label' style='margin-top:1rem'>Chart Period</div>",
#                 unsafe_allow_html=True)
#     period_label = st.selectbox("Price history window", list(PERIOD_MAP.keys()), index=2)
#     period = PERIOD_MAP[period_label]

#     st.markdown("---")
#     if st.button("🔄  Refresh Data", use_container_width=True):
#         st.cache_data.clear()
#         st.rerun()

#     st.markdown(
#         "<div style='font-size:0.65rem;color:#6b6b85;line-height:1.9;margin-top:1rem'>"
#         "Data via <b style='color:#c8c8d8'>yfinance</b> (Yahoo Finance)<br>"
#         "✅ Free · No API key needed<br>"
#         "Cache TTL: 15 minutes"
#         "</div>", unsafe_allow_html=True,
#     )

# # ── HEADER ────────────────────────────────────────────────────────────────────
# st.markdown(
#     "<h1 style='font-family:Syne,sans-serif;font-size:2rem;margin-bottom:0.2rem'>"
#     "📡 Market Intelligence Dashboard</h1>"
#     "<div style='color:#6b6b85;font-size:0.75rem;margin-bottom:1.5rem'>"
#     "🟢 Live data · Powered by Yahoo Finance · No API key required</div>",
#     unsafe_allow_html=True,
# )

# if not selected:
#     st.warning("Select at least one company from the sidebar.")
#     st.stop()

# # ── FETCH ALL ─────────────────────────────────────────────────────────────────
# companies = {}
# bar = st.progress(0, text="Loading market data…")

# for i, name in enumerate(selected):
#     sym = DEFAULT_TICKERS.get(name, name)
#     try:
#         companies[name] = load_company(name, sym, period)
#     except Exception as e:
#         st.warning(f"Could not load **{name}** ({sym}): {e}")
#     bar.progress((i + 1) / len(selected), text=f"Fetching {name}…")

# bar.empty()
# data = list(companies.values())

# if not data:
#     st.error("No data loaded. Check your ticker symbols.")
#     st.stop()

# # ── KPI STRIP ─────────────────────────────────────────────────────────────────
# st.markdown("### Overview")
# k1, k2, k3, k4, k5 = st.columns(5)
# k1.metric("Companies",       len(data))
# k2.metric("Total Mkt Cap",   fmt_big(sum(d["market_cap"] or 0 for d in data)))
# k3.metric("Total Revenue",   fmt_big(sum(d["revenue"]    or 0 for d in data)))
# k4.metric("Total Employees", fmt_num(sum(d["employees"]  or 0 for d in data)))
# k5.metric("Gainers Today",   f"{sum(1 for d in data if d['change_pct'] and d['change_pct']>0)}/{len(data)}")

# st.markdown("---")

# # ── COMPANY CARDS ─────────────────────────────────────────────────────────────
# st.markdown("### Company Profiles")
# num_cols = min(len(data), 3)
# cols = st.columns(num_cols)

# for i, d in enumerate(data):
#     cls       = pct_class(d["change_pct"])
#     price_str = f"${d['price']:.2f}" if d["price"] else "N/A"
#     chg_str   = (
#         f"{arrow(d['change'])} ${abs(d['change']):.2f} ({abs(d['change_pct']):.2f}%)"
#         if d["change"] is not None else "N/A"
#     )
#     with cols[i % num_cols]:
#         st.markdown(f"""
#         <div class='co-card'>
#           <div class='co-card-name'>{d['name']} <span style='color:#6b6b85;font-size:.8rem'>({d['symbol']})</span></div>
#           <div class='co-card-type'>{d['sector']} · {d['industry']}</div>
#           <div class='co-card-price'>{price_str}</div>
#           <div class='co-card-{cls}'>{chg_str}</div>
#           <div class='co-meta'>
#             <b>HQ:</b> {d['hq']}<br>
#             <b>Employees:</b> {fmt_num(d['employees'])}<br>
#             <b>Market Cap:</b> {fmt_big(d['market_cap'])}<br>
#             <b>P/E Ratio:</b> {f"{d['pe_ratio']:.1f}" if d['pe_ratio'] else 'N/A'}<br>
#             <b>52W High:</b> {f"${d['week52_high']:.2f}" if d['week52_high'] else 'N/A'}
#             &nbsp;<b>Low:</b> {f"${d['week52_low']:.2f}" if d['week52_low'] else 'N/A'}
#           </div>
#         </div>""", unsafe_allow_html=True)

# st.markdown("---")

# # ── STOCK PRICE HISTORY ───────────────────────────────────────────────────────
# st.markdown("### Stock Price History")
# st.markdown(f"<div class='section-label'>Closing Price — {period_label}</div>", unsafe_allow_html=True)

# fig_hist = go.Figure()
# for i, d in enumerate(data):
#     hist = d["history"]
#     if hist is not None and not hist.empty:
#         fig_hist.add_trace(go.Scatter(
#             x=hist.index, y=hist["Close"],
#             name=d["symbol"], mode="lines",
#             line=dict(color=COLORS[i % len(COLORS)], width=2.2),
#             hovertemplate=f"<b>{d['name']}</b><br>%{{x|%b %d, %Y}}<br>${{y:.2f}}<extra></extra>",
#         ))

# fig_hist.update_layout(
#     paper_bgcolor="#16161f", plot_bgcolor="#16161f", font_color="#6b6b85",
#     hovermode="x unified",
#     legend=dict(orientation="h", y=-0.15, font=dict(size=11)),
#     margin=dict(t=10, b=30, l=10, r=10),
#     xaxis=dict(showgrid=False, color="#6b6b85"),
#     yaxis=dict(showgrid=True, gridcolor="#1e1e2e", tickprefix="$", color="#6b6b85"),
#     height=400,
# )
# st.plotly_chart(fig_hist, use_container_width=True)

# # ── FINANCIALS ────────────────────────────────────────────────────────────────
# st.markdown("### Financials")
# names      = [d["name"].split()[0] for d in data]
# mktcaps    = [(d["market_cap"]  or 0)/1e9 for d in data]
# revenues   = [(d["revenue"]     or 0)/1e9 for d in data]
# net_incomes= [(d["net_income"]  or 0)/1e9 for d in data]
# employees  = [(d["employees"]   or 0)/1e3 for d in data]
# pe_vals    = [ d["pe_ratio"]    or 0      for d in data]

# f1, f2 = st.columns(2)

# with f1:
#     st.markdown("<div class='section-label'>Market Cap (B USD)</div>", unsafe_allow_html=True)
#     fig_mc = px.bar(x=names, y=mktcaps, color=names,
#                     color_discrete_sequence=COLORS,
#                     text=[f"${v:.1f}B" for v in mktcaps])
#     fig_mc.update_traces(textposition="outside", marker_line_width=0)
#     fig_mc.update_layout(paper_bgcolor="#16161f", plot_bgcolor="#16161f",
#                          font_color="#6b6b85", showlegend=False, height=300,
#                          margin=dict(t=20,b=10,l=10,r=10),
#                          xaxis=dict(showgrid=False),
#                          yaxis=dict(showgrid=True, gridcolor="#1e1e2e", ticksuffix="B"))
#     st.plotly_chart(fig_mc, use_container_width=True)

# with f2:
#     st.markdown("<div class='section-label'>Revenue vs Net Income (B USD)</div>", unsafe_allow_html=True)
#     fig_fn = go.Figure([
#         go.Bar(name="Revenue",    x=names, y=revenues,
#                marker_color=COLORS[0], text=[f"${v:.1f}B" for v in revenues],
#                textposition="outside"),
#         go.Bar(name="Net Income", x=names, y=net_incomes,
#                marker_color=COLORS[1], text=[f"${v:.1f}B" for v in net_incomes],
#                textposition="outside"),
#     ])
#     fig_fn.update_layout(barmode="group", paper_bgcolor="#16161f", plot_bgcolor="#16161f",
#                          font_color="#6b6b85", height=300,
#                          legend=dict(orientation="h", y=-0.25, font=dict(size=11)),
#                          margin=dict(t=20,b=30,l=10,r=10),
#                          xaxis=dict(showgrid=False),
#                          yaxis=dict(showgrid=True, gridcolor="#1e1e2e", ticksuffix="B"))
#     st.plotly_chart(fig_fn, use_container_width=True)

# f3, f4 = st.columns(2)

# with f3:
#     st.markdown("<div class='section-label'>Workforce (Thousands)</div>", unsafe_allow_html=True)
#     fig_emp = px.bar(y=names, x=employees, orientation="h",
#                      color=names, color_discrete_sequence=COLORS[::-1],
#                      text=[f"{v:.0f}K" for v in employees])
#     fig_emp.update_traces(textposition="outside", marker_line_width=0)
#     fig_emp.update_layout(paper_bgcolor="#16161f", plot_bgcolor="#16161f",
#                           font_color="#6b6b85", showlegend=False, height=300,
#                           margin=dict(t=10,b=10,l=10,r=10),
#                           xaxis=dict(showgrid=True, gridcolor="#1e1e2e", ticksuffix="K"),
#                           yaxis=dict(showgrid=False))
#     st.plotly_chart(fig_emp, use_container_width=True)

# with f4:
#     st.markdown("<div class='section-label'>P/E Ratio</div>", unsafe_allow_html=True)
#     fig_pe = px.bar(x=names, y=pe_vals, color=names,
#                     color_discrete_sequence=COLORS,
#                     text=[f"{v:.1f}x" if v else "N/A" for v in pe_vals])
#     fig_pe.update_traces(textposition="outside", marker_line_width=0)
#     fig_pe.update_layout(paper_bgcolor="#16161f", plot_bgcolor="#16161f",
#                          font_color="#6b6b85", showlegend=False, height=300,
#                          margin=dict(t=20,b=10,l=10,r=10),
#                          xaxis=dict(showgrid=False),
#                          yaxis=dict(showgrid=True, gridcolor="#1e1e2e", ticksuffix="x"))
#     st.plotly_chart(fig_pe, use_container_width=True)

# # ── SCATTER + DONUT ───────────────────────────────────────────────────────────
# sc_col, do_col = st.columns(2)

# with sc_col:
#     st.markdown("<div class='section-label'>Market Cap vs Revenue</div>", unsafe_allow_html=True)
#     fig_sc = px.scatter(x=revenues, y=mktcaps, text=names,
#                         size=[max(v, 1) for v in mktcaps],
#                         color=names, color_discrete_sequence=COLORS,
#                         labels={"x":"Revenue (B USD)","y":"Market Cap (B USD)"})
#     fig_sc.update_traces(textposition="top center", marker_line_width=0)
#     fig_sc.update_layout(paper_bgcolor="#16161f", plot_bgcolor="#16161f",
#                          font_color="#6b6b85", showlegend=False, height=320,
#                          margin=dict(t=10,b=10,l=10,r=10),
#                          xaxis=dict(showgrid=True, gridcolor="#1e1e2e", ticksuffix="B"),
#                          yaxis=dict(showgrid=True, gridcolor="#1e1e2e", ticksuffix="B"))
#     st.plotly_chart(fig_sc, use_container_width=True)

# with do_col:
#     st.markdown("<div class='section-label'>Sector Distribution</div>", unsafe_allow_html=True)
#     sc = {}
#     for d in data:
#         sc[d["sector"]] = sc.get(d["sector"], 0) + 1
#     fig_dn = px.pie(names=list(sc.keys()), values=list(sc.values()),
#                     hole=0.6, color_discrete_sequence=COLORS)
#     fig_dn.update_traces(textinfo="label+percent", showlegend=False)
#     fig_dn.update_layout(paper_bgcolor="#16161f", font_color="#6b6b85",
#                          margin=dict(t=10,b=10,l=10,r=10), height=320)
#     st.plotly_chart(fig_dn, use_container_width=True)

# st.markdown("---")

# # ── DATA TABLE ────────────────────────────────────────────────────────────────
# st.markdown("### Full Data Table")
# df = pd.DataFrame([{
#     "Company":    d["name"],
#     "Symbol":     d["symbol"],
#     "Sector":     d["sector"],
#     "Price":      f"${d['price']:.2f}"        if d["price"]      else "N/A",
#     "Change %":   f"{d['change_pct']:+.2f}%"  if d["change_pct"] else "N/A",
#     "Market Cap": fmt_big(d["market_cap"]),
#     "Revenue":    fmt_big(d["revenue"]),
#     "Net Income": fmt_big(d["net_income"]),
#     "Employees":  fmt_num(d["employees"]),
#     "P/E":        f"{d['pe_ratio']:.1f}"      if d["pe_ratio"]   else "N/A",
#     "Beta":       f"{d['beta']:.2f}"          if d["beta"]       else "N/A",
#     "52W High":   f"${d['week52_high']:.2f}"  if d["week52_high"] else "N/A",
#     "52W Low":    f"${d['week52_low']:.2f}"   if d["week52_low"]  else "N/A",
#     "Website":    d["website"],
# } for d in data])

# st.dataframe(df, use_container_width=True, hide_index=True,
#              column_config={
#                  "Website": st.column_config.LinkColumn("Website"),
#                  "Company": st.column_config.TextColumn("Company", width="medium"),
#              })

# # ── DESCRIPTIONS ──────────────────────────────────────────────────────────────
# st.markdown("---")
# st.markdown("### Company Descriptions")
# for d in data:
#     with st.expander(f"📄  {d['name']} ({d['symbol']})"):
#         st.markdown(
#             f"<div style='color:#c8c8d8;font-size:0.82rem;line-height:1.9'>{d['description']}</div>",
#             unsafe_allow_html=True,
#         )
#         cm = st.columns(4)
#         cm[0].metric("HQ",         d["hq"])
#         cm[1].metric("Industry",   d["industry"])
#         cm[2].metric("Div. Yield", f"{d['div_yield']*100:.2f}%" if d["div_yield"] else "N/A")
#         cm[3].metric("EPS (TTM)",  f"${d['eps']:.2f}"           if d["eps"]       else "N/A")

# # ── FOOTER ────────────────────────────────────────────────────────────────────
# st.markdown("---")
# st.markdown(
#     "<div style='text-align:center;color:#6b6b85;font-size:0.68rem;padding:1rem 0'>"
#     "Market Intel · Data via Yahoo Finance (yfinance) · For educational use only"
#     "</div>", unsafe_allow_html=True,
# )