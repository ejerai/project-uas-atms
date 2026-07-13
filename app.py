import os
import re
import json
import math
from collections import Counter

import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st

# ─────────────────────────────────────────────
# PAGE CONFIG
# ─────────────────────────────────────────────
st.set_page_config(
    page_title="Bansos Insight · Dashboard Analisis",
    page_icon="🔥",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ─────────────────────────────────────────────
# DESIGN TOKENS — persis dari bansos_insight_v3.html
# ─────────────────────────────────────────────
INK        = "#0A0A0A"
INK2       = "#0F0F0F"
INK3       = "#161616"
INK4       = "#1C1C1C"
INK5       = "#242424"
FIRE       = "#F55A00"
FIRE2      = "#FF7A30"
FIRE_DIM   = "rgba(245,90,0,0.10)"
FIRE_GLOW  = "rgba(245,90,0,0.05)"
LINE       = "rgba(255,255,255,0.05)"
LINE2      = "rgba(255,255,255,0.08)"
LINE_HOVER = "rgba(255,255,255,0.12)"
T1 = "#F0F0F0"
T2 = "#888888"
T3 = "#444444"
GREEN     = "#3FB950"
RED_SOFT  = "#F85149"

SENTIMENT_COLORS = {
    "positif":  "#666666",
    "negatif":  FIRE,
    "netral":   T3,
    "positive": "#666666",
    "negative": FIRE,
    "neutral":  T3,
}
CLUSTER_PALETTE = [FIRE, "#666666", "#2E2E2E", "#FF7A30", "#4A4A4A", "#C9622E"]

# ─────────────────────────────────────────────
# SVG ICONS
# ─────────────────────────────────────────────
def svg(name, size=16, color="currentColor"):
    s = size; c = color
    icons = {
        "tweet":    f'<svg width="{s}" height="{s}" viewBox="0 0 24 24" fill="none" stroke="{c}" stroke-width="1.8" stroke-linecap="round" stroke-linejoin="round"><path d="M23 3a10.9 10.9 0 01-3.14 1.53 4.48 4.48 0 00-7.86 3v1A10.66 10.66 0 013 4s-4 9 5 13a11.64 11.64 0 01-7 2c9 5 20 0 20-11.5a4.5 4.5 0 00-.08-.83A7.72 7.72 0 0023 3z"/></svg>',
        "cluster":  f'<svg width="{s}" height="{s}" viewBox="0 0 24 24" fill="none" stroke="{c}" stroke-width="1.8" stroke-linecap="round" stroke-linejoin="round"><circle cx="18" cy="5" r="3"/><circle cx="6" cy="12" r="3"/><circle cx="18" cy="19" r="3"/><line x1="8.59" y1="13.51" x2="15.42" y2="17.49"/><line x1="15.41" y1="6.51" x2="8.59" y2="10.49"/></svg>',
        "retweet":  f'<svg width="{s}" height="{s}" viewBox="0 0 24 24" fill="none" stroke="{c}" stroke-width="1.8" stroke-linecap="round" stroke-linejoin="round"><polyline points="17 1 21 5 17 9"/><path d="M3 11V9a4 4 0 014-4h14"/><polyline points="7 23 3 19 7 15"/><path d="M21 13v2a4 4 0 01-4 4H3"/></svg>',
        "heart":    f'<svg width="{s}" height="{s}" viewBox="0 0 24 24" fill="none" stroke="{c}" stroke-width="1.8" stroke-linecap="round" stroke-linejoin="round"><path d="M20.84 4.61a5.5 5.5 0 00-7.78 0L12 5.67l-1.06-1.06a5.5 5.5 0 00-7.78 7.78l1.06 1.06L12 21.23l7.78-7.78 1.06-1.06a5.5 5.5 0 000-7.78z"/></svg>',
        "chart":    f'<svg width="{s}" height="{s}" viewBox="0 0 24 24" fill="none" stroke="{c}" stroke-width="1.8" stroke-linecap="round" stroke-linejoin="round"><line x1="18" y1="20" x2="18" y2="10"/><line x1="12" y1="20" x2="12" y2="4"/><line x1="6" y1="20" x2="6" y2="14"/></svg>',
        "network":  f'<svg width="{s}" height="{s}" viewBox="0 0 24 24" fill="none" stroke="{c}" stroke-width="1.8" stroke-linecap="round" stroke-linejoin="round"><circle cx="12" cy="4" r="2"/><circle cx="4" cy="20" r="2"/><circle cx="20" cy="20" r="2"/><line x1="12" y1="6" x2="4" y2="18"/><line x1="12" y1="6" x2="20" y2="18"/><line x1="6" y1="20" x2="18" y2="20"/></svg>',
        "search":   f'<svg width="{s}" height="{s}" viewBox="0 0 24 24" fill="none" stroke="{c}" stroke-width="1.8" stroke-linecap="round" stroke-linejoin="round"><circle cx="11" cy="11" r="8"/><line x1="21" y1="21" x2="16.65" y2="16.65"/></svg>',
        "filter":   f'<svg width="{s}" height="{s}" viewBox="0 0 24 24" fill="none" stroke="{c}" stroke-width="1.8" stroke-linecap="round" stroke-linejoin="round"><polygon points="22 3 2 3 10 12.46 10 19 14 21 14 12.46 22 3"/></svg>',
        "calendar": f'<svg width="{s}" height="{s}" viewBox="0 0 24 24" fill="none" stroke="{c}" stroke-width="1.8" stroke-linecap="round" stroke-linejoin="round"><rect x="3" y="4" width="18" height="18" rx="2" ry="2"/><line x1="16" y1="2" x2="16" y2="6"/><line x1="8" y1="2" x2="8" y2="6"/><line x1="3" y1="10" x2="21" y2="10"/></svg>',
        "tag":      f'<svg width="{s}" height="{s}" viewBox="0 0 24 24" fill="none" stroke="{c}" stroke-width="1.8" stroke-linecap="round" stroke-linejoin="round"><path d="M20.59 13.41l-7.17 7.17a2 2 0 01-2.83 0L2 12V2h10l8.59 8.59a2 2 0 010 2.82z"/><line x1="7" y1="7" x2="7.01" y2="7"/></svg>',
        "trending": f'<svg width="{s}" height="{s}" viewBox="0 0 24 24" fill="none" stroke="{c}" stroke-width="1.8" stroke-linecap="round" stroke-linejoin="round"><polyline points="23 6 13.5 15.5 8.5 10.5 1 18"/><polyline points="17 6 23 6 23 12"/></svg>',
        "pipeline": f'<svg width="{s}" height="{s}" viewBox="0 0 24 24" fill="none" stroke="{c}" stroke-width="1.8" stroke-linecap="round" stroke-linejoin="round"><polyline points="22 12 18 12 15 21 9 3 6 12 2 12"/></svg>',
        "smile":    f'<svg width="{s}" height="{s}" viewBox="0 0 24 24" fill="none" stroke="{c}" stroke-width="1.8" stroke-linecap="round" stroke-linejoin="round"><circle cx="12" cy="12" r="10"/><path d="M8 14s1.5 2 4 2 4-2 4-2"/><line x1="9" y1="9" x2="9.01" y2="9"/><line x1="15" y1="9" x2="15.01" y2="9"/></svg>',
        "overview": f'<svg width="{s}" height="{s}" viewBox="0 0 24 24" fill="none" stroke="{c}" stroke-width="1.8" stroke-linecap="round" stroke-linejoin="round"><rect x="3" y="3" width="7" height="7" rx="1"/><rect x="14" y="3" width="7" height="7" rx="1"/><rect x="3" y="14" width="7" height="7" rx="1"/><rect x="14" y="14" width="7" height="7" rx="1"/></svg>',
        "download": f'<svg width="{s}" height="{s}" viewBox="0 0 24 24" fill="none" stroke="{c}" stroke-width="1.8" stroke-linecap="round" stroke-linejoin="round"><path d="M21 15v4a2 2 0 01-2 2H5a2 2 0 01-2-2v-4"/><polyline points="7 10 12 15 17 10"/><line x1="12" y1="15" x2="12" y2="3"/></svg>',
        "warning":  f'<svg width="{s}" height="{s}" viewBox="0 0 24 24" fill="none" stroke="{c}" stroke-width="1.8" stroke-linecap="round" stroke-linejoin="round"><path d="M10.29 3.86L1.82 18a2 2 0 001.71 3h16.94a2 2 0 001.71-3L13.71 3.86a2 2 0 00-3.42 0z"/><line x1="12" y1="9" x2="12" y2="13"/><line x1="12" y1="17" x2="12.01" y2="17"/></svg>',
        "check":    f'<svg width="{s}" height="{s}" viewBox="0 0 24 24" fill="none" stroke="{c}" stroke-width="2.2" stroke-linecap="round" stroke-linejoin="round"><polyline points="20 6 9 17 4 12"/></svg>',
        "chevron-left": f'<svg width="{s}" height="{s}" viewBox="0 0 24 24" fill="none" stroke="{c}" stroke-width="1.8" stroke-linecap="round" stroke-linejoin="round"><polyline points="15 18 9 12 15 6"/></svg>',
    }
    return icons.get(name, icons["overview"])

# ─────────────────────────────────────────────
# GLOBAL CSS — EXACT MATCH bansos_insight_v3.html
# ─────────────────────────────────────────────
st.markdown(f"""
<style>
@import url('https://fonts.googleapis.com/css2?family=Poppins:wght@300;400;500;600;700;800&display=swap');

*,*::before,*::after{{box-sizing:border-box;margin:0;padding:0;}}
html,body,[class*="css"]{{
  font-family:'Poppins',-apple-system,sans-serif;
  color:{T1}; font-size:13px; line-height:1.55;
  -webkit-font-smoothing:antialiased;
}}
::-webkit-scrollbar{{width:3px;}}
::-webkit-scrollbar-track{{background:transparent;}}
::-webkit-scrollbar-thumb{{background:{INK5};border-radius:3px;}}

/* ── KILL ALL BLUE LINK COLORS ── */
a, a:visited, a:hover, a:active, a:focus {{
  color:inherit !important;
  text-decoration:none !important;
  outline:none !important;
}}

/* ── HIDE STREAMLIT CHROME ── */
#MainMenu,footer,.stDeployButton{{visibility:hidden;}}
header[data-testid="stHeader"]{{background:transparent;height:0;}}
.stApp{{background:{INK};}}
.block-container{{padding:0!important;max-width:100%!important;}}

/* ── SIDEBAR RESET ── */
section[data-testid="stSidebar"]{{
  width:216px!important;min-width:216px!important;max-width:216px!important;
  background:{INK}!important;border-right:none!important;
  padding:0!important;position:relative!important;overflow:hidden!important;
}}
section[data-testid="stSidebar"]::before{{
  content:'';position:fixed;top:0;left:0;width:216px;height:2px;
  background:linear-gradient(90deg,{FIRE} 0%,rgba(245,90,0,0.2) 60%,transparent 100%);
  z-index:999;pointer-events:none;
}}
section[data-testid="stSidebar"] > div{{
  width:216px!important;background:{INK}!important;padding:0!important;
}}
[data-testid="stSidebarContent"]{{background:{INK}!important;padding:0!important;}}
[data-testid="stSidebarUserContent"]{{padding:0!important;}}
[data-testid="collapsedControl"]{{display:none!important;}}
button[kind="header"]{{display:none!important;}}
[data-testid="stSidebarCollapseButton"]{{display:none!important;}}
.st-emotion-cache-dvne4q{{display:none!important;}}

/* ── SIDEBAR STRUCTURE ── */
.sb-top{{padding:18px 14px 16px 18px;display:flex;align-items:center;justify-content:space-between;gap:8px;}}
.sb-logo{{display:flex;align-items:center;gap:10px;min-width:0;overflow:hidden;}}
.sb-icon{{
  width:32px;height:32px;border-radius:8px;
  background:{FIRE_DIM};border:1px solid rgba(245,90,0,0.25);
  display:flex;align-items:center;justify-content:center;
  position:relative;flex-shrink:0;
}}
.sb-icon::after{{
  content:'';position:absolute;top:-4px;right:-4px;
  width:8px;height:8px;border-radius:2px;
  background:{FIRE_DIM};border:1px solid rgba(245,90,0,0.25);
  transform:rotate(30deg);
}}
.sb-logo-text{{min-width:0;overflow:hidden;white-space:nowrap;}}
.sb-title{{font-family:'Poppins',sans-serif;font-size:14px;font-weight:700;color:{T1};letter-spacing:-.2px;white-space:nowrap;}}
.sb-sub{{font-size:9.5px;color:{T3};letter-spacing:.3px;margin-top:1px;white-space:nowrap;}}
.sb-divider{{height:1px;background:{LINE};margin:0 18px;}}
.sb-nav-scroll{{padding:14px 10px;}}
.sb-section{{font-size:9px;font-weight:600;color:{T3};text-transform:uppercase;letter-spacing:1.8px;margin:14px 0 5px 8px;}}
.sb-section:first-of-type{{margin-top:0;}}

/* NAV BUTTONS — Streamlit st.button() in sidebar, styled as nav-btn */
/* Base styles */
section[data-testid="stSidebar"] div[class*="st-key-nav_"] > div > button {{
  width:100%!important;display:flex!important;align-items:center!important;gap:8px!important;
  padding:7px 10px 7px 32px!important;border-radius:7px!important;
  font-family:'Poppins',sans-serif!important;font-size:11.5px!important;font-weight:500!important;
  cursor:pointer!important;text-align:left!important;justify-content:flex-start!important;
  transition:all .18s!important;margin-bottom:2px!important;position:relative!important;
  box-shadow:none!important;min-height:34px!important;
}}
/* Inactive */
section[data-testid="stSidebar"] div[class*="st-key-nav_"] > div > button[kind="secondary"] {{
  background:transparent!important;border:1px solid transparent!important;color:{T3}!important;
}}
section[data-testid="stSidebar"] div[class*="st-key-nav_"] > div > button[kind="secondary"]:hover {{
  background:{INK3}!important;color:{T1}!important;border-color:{LINE_HOVER}!important;
}}
/* Active */
section[data-testid="stSidebar"] div[class*="st-key-nav_"] > div > button[kind="primary"] {{
  background:{INK3}!important;border:1px solid {LINE}!important;border-left:2px solid {FIRE}!important;
  color:{T1}!important;box-shadow:none!important;
}}
section[data-testid="stSidebar"] div[class*="st-key-nav_"] > div > button:focus:not(:active) {{
  box-shadow:none!important;
}}
/* SVG icons via ::before pseudo-element — base64 encoded SVGs */
/* overview icon */
section[data-testid="stSidebar"] div[class*="st-key-nav_overview"] > div > button::before {{
  content:'';display:block;position:absolute;left:10px;top:50%;transform:translateY(-50%);
  width:13px;height:13px;flex-shrink:0;
  background:currentColor;mask-size:contain;mask-repeat:no-repeat;mask-position:center;
  -webkit-mask-size:contain;-webkit-mask-repeat:no-repeat;-webkit-mask-position:center;
  -webkit-mask-image:url("data:image/svg+xml,%3Csvg xmlns='http://www.w3.org/2000/svg' viewBox='0 0 24 24' fill='none' stroke='currentColor' stroke-width='1.8'%3E%3Crect x='3' y='3' width='7' height='7' rx='1'/%3E%3Crect x='14' y='3' width='7' height='7' rx='1'/%3E%3Crect x='3' y='14' width='7' height='7' rx='1'/%3E%3Crect x='14' y='14' width='7' height='7' rx='1'/%3E%3C/svg%3E");
  mask-image:url("data:image/svg+xml,%3Csvg xmlns='http://www.w3.org/2000/svg' viewBox='0 0 24 24' fill='none' stroke='currentColor' stroke-width='1.8'%3E%3Crect x='3' y='3' width='7' height='7' rx='1'/%3E%3Crect x='14' y='3' width='7' height='7' rx='1'/%3E%3Crect x='3' y='14' width='7' height='7' rx='1'/%3E%3Crect x='14' y='14' width='7' height='7' rx='1'/%3E%3C/svg%3E");
}}
/* sentiment icon */
section[data-testid="stSidebar"] div[class*="st-key-nav_sentiment"] > div > button::before {{
  content:'';display:block;position:absolute;left:10px;top:50%;transform:translateY(-50%);
  width:13px;height:13px;background:currentColor;mask-size:contain;mask-repeat:no-repeat;mask-position:center;
  -webkit-mask-size:contain;-webkit-mask-repeat:no-repeat;-webkit-mask-position:center;
  -webkit-mask-image:url("data:image/svg+xml,%3Csvg xmlns='http://www.w3.org/2000/svg' viewBox='0 0 24 24' fill='none' stroke='currentColor' stroke-width='1.8'%3E%3Ccircle cx='12' cy='12' r='10'/%3E%3Cpath d='M8 14s1.5 2 4 2 4-2 4-2'/%3E%3Cline x1='9' y1='9' x2='9.01' y2='9'/%3E%3Cline x1='15' y1='9' x2='15.01' y2='9'/%3E%3C/svg%3E");
  mask-image:url("data:image/svg+xml,%3Csvg xmlns='http://www.w3.org/2000/svg' viewBox='0 0 24 24' fill='none' stroke='currentColor' stroke-width='1.8'%3E%3Ccircle cx='12' cy='12' r='10'/%3E%3Cpath d='M8 14s1.5 2 4 2 4-2 4-2'/%3E%3Cline x1='9' y1='9' x2='9.01' y2='9'/%3E%3Cline x1='15' y1='9' x2='15.01' y2='9'/%3E%3C/svg%3E");
}}
/* trend icon */
section[data-testid="stSidebar"] div[class*="st-key-nav_trend"] > div > button::before {{
  content:'';display:block;position:absolute;left:10px;top:50%;transform:translateY(-50%);
  width:13px;height:13px;background:currentColor;mask-size:contain;mask-repeat:no-repeat;mask-position:center;
  -webkit-mask-size:contain;-webkit-mask-repeat:no-repeat;-webkit-mask-position:center;
  -webkit-mask-image:url("data:image/svg+xml,%3Csvg xmlns='http://www.w3.org/2000/svg' viewBox='0 0 24 24' fill='none' stroke='currentColor' stroke-width='1.8'%3E%3Cpolyline points='23 6 13.5 15.5 8.5 10.5 1 18'/%3E%3Cpolyline points='17 6 23 6 23 12'/%3E%3C/svg%3E");
  mask-image:url("data:image/svg+xml,%3Csvg xmlns='http://www.w3.org/2000/svg' viewBox='0 0 24 24' fill='none' stroke='currentColor' stroke-width='1.8'%3E%3Cpolyline points='23 6 13.5 15.5 8.5 10.5 1 18'/%3E%3Cpolyline points='17 6 23 6 23 12'/%3E%3C/svg%3E");
}}
/* sna icon */
section[data-testid="stSidebar"] div[class*="st-key-nav_sna"] > div > button::before {{
  content:'';display:block;position:absolute;left:10px;top:50%;transform:translateY(-50%);
  width:13px;height:13px;background:currentColor;mask-size:contain;mask-repeat:no-repeat;mask-position:center;
  -webkit-mask-size:contain;-webkit-mask-repeat:no-repeat;-webkit-mask-position:center;
  -webkit-mask-image:url("data:image/svg+xml,%3Csvg xmlns='http://www.w3.org/2000/svg' viewBox='0 0 24 24' fill='none' stroke='currentColor' stroke-width='1.8'%3E%3Ccircle cx='18' cy='5' r='3'/%3E%3Ccircle cx='6' cy='12' r='3'/%3E%3Ccircle cx='18' cy='19' r='3'/%3E%3Cline x1='8.59' y1='13.51' x2='15.42' y2='17.49'/%3E%3Cline x1='15.41' y1='6.51' x2='8.59' y2='10.49'/%3E%3C/svg%3E");
  mask-image:url("data:image/svg+xml,%3Csvg xmlns='http://www.w3.org/2000/svg' viewBox='0 0 24 24' fill='none' stroke='currentColor' stroke-width='1.8'%3E%3Ccircle cx='18' cy='5' r='3'/%3E%3Ccircle cx='6' cy='12' r='3'/%3E%3Ccircle cx='18' cy='19' r='3'/%3E%3Cline x1='8.59' y1='13.51' x2='15.42' y2='17.49'/%3E%3Cline x1='15.41' y1='6.51' x2='8.59' y2='10.49'/%3E%3C/svg%3E");
}}
/* keywords icon */
section[data-testid="stSidebar"] div[class*="st-key-nav_keywords"] > div > button::before {{
  content:'';display:block;position:absolute;left:10px;top:50%;transform:translateY(-50%);
  width:13px;height:13px;background:currentColor;mask-size:contain;mask-repeat:no-repeat;mask-position:center;
  -webkit-mask-size:contain;-webkit-mask-repeat:no-repeat;-webkit-mask-position:center;
  -webkit-mask-image:url("data:image/svg+xml,%3Csvg xmlns='http://www.w3.org/2000/svg' viewBox='0 0 24 24' fill='none' stroke='currentColor' stroke-width='1.8'%3E%3Cpath d='M20.59 13.41l-7.17 7.17a2 2 0 01-2.83 0L2 12V2h10l8.59 8.59a2 2 0 010 2.82z'/%3E%3Cline x1='7' y1='7' x2='7.01' y2='7'/%3E%3C/svg%3E");
  mask-image:url("data:image/svg+xml,%3Csvg xmlns='http://www.w3.org/2000/svg' viewBox='0 0 24 24' fill='none' stroke='currentColor' stroke-width='1.8'%3E%3Cpath d='M20.59 13.41l-7.17 7.17a2 2 0 01-2.83 0L2 12V2h10l8.59 8.59a2 2 0 010 2.82z'/%3E%3Cline x1='7' y1='7' x2='7.01' y2='7'/%3E%3C/svg%3E");
}}
/* pipeline icon */
section[data-testid="stSidebar"] div[class*="st-key-nav_pipeline"] > div > button::before {{
  content:'';display:block;position:absolute;left:10px;top:50%;transform:translateY(-50%);
  width:13px;height:13px;background:currentColor;mask-size:contain;mask-repeat:no-repeat;mask-position:center;
  -webkit-mask-size:contain;-webkit-mask-repeat:no-repeat;-webkit-mask-position:center;
  -webkit-mask-image:url("data:image/svg+xml,%3Csvg xmlns='http://www.w3.org/2000/svg' viewBox='0 0 24 24' fill='none' stroke='currentColor' stroke-width='1.8'%3E%3Cpolyline points='22 12 18 12 15 21 9 3 6 12 2 12'/%3E%3C/svg%3E");
  mask-image:url("data:image/svg+xml,%3Csvg xmlns='http://www.w3.org/2000/svg' viewBox='0 0 24 24' fill='none' stroke='currentColor' stroke-width='1.8'%3E%3Cpolyline points='22 12 18 12 15 21 9 3 6 12 2 12'/%3E%3C/svg%3E");
}}
/* tweets icon */
section[data-testid="stSidebar"] div[class*="st-key-nav_tweets"] > div > button::before {{
  content:'';display:block;position:absolute;left:10px;top:50%;transform:translateY(-50%);
  width:13px;height:13px;background:currentColor;mask-size:contain;mask-repeat:no-repeat;mask-position:center;
  -webkit-mask-size:contain;-webkit-mask-repeat:no-repeat;-webkit-mask-position:center;
  -webkit-mask-image:url("data:image/svg+xml,%3Csvg xmlns='http://www.w3.org/2000/svg' viewBox='0 0 24 24' fill='none' stroke='currentColor' stroke-width='1.8'%3E%3Cpath d='M23 3a10.9 10.9 0 01-3.14 1.53 4.48 4.48 0 00-7.86 3v1A10.66 10.66 0 013 4s-4 9 5 13a11.64 11.64 0 01-7 2c9 5 20 0 20-11.5'/%3E%3C/svg%3E");
  mask-image:url("data:image/svg+xml,%3Csvg xmlns='http://www.w3.org/2000/svg' viewBox='0 0 24 24' fill='none' stroke='currentColor' stroke-width='1.8'%3E%3Cpath d='M23 3a10.9 10.9 0 01-3.14 1.53 4.48 4.48 0 00-7.86 3v1A10.66 10.66 0 013 4s-4 9 5 13a11.64 11.64 0 01-7 2c9 5 20 0 20-11.5'/%3E%3C/svg%3E");
}}
.btn-label{{white-space:nowrap;}}

/* SIDEBAR FOOTER */
.sb-footer{{padding:14px 10px 18px;}}
.sb-footer-divider{{height:1px;background:{LINE};margin-bottom:12px;}}
.sb-stats{{display:grid;grid-template-columns:1fr 1fr;gap:6px;}}
.sb-stat{{background:{INK3};border-radius:7px;padding:8px 10px;border:1px solid transparent;transition:border-color .18s;}}
.sb-stat:hover{{border-color:{LINE};}}
.sb-stat-v{{font-family:'Poppins',sans-serif;font-size:15px;font-weight:700;color:{T1};line-height:1;}}
.sb-stat-l{{font-size:9px;color:{T3};margin-top:2px;}}
.sb-stat.wide{{grid-column:1/-1;}}
.sb-stat.wide .sb-stat-v{{font-size:11.5px;color:{T2};font-weight:500;}}

/* SIDEBAR FILTER WIDGETS STYLING */
section[data-testid="stSidebar"] .stMultiSelect [data-baseweb="select"]>div{{
  background:{INK3}!important;border-color:{LINE2}!important;color:{T1}!important;border-radius:7px!important;font-size:11.5px!important;
}}
section[data-testid="stSidebar"] .stTextInput input{{
  background:{INK3}!important;border:1px solid {LINE2}!important;color:{T1}!important;border-radius:7px!important;font-size:11.5px!important;
}}
section[data-testid="stSidebar"] label{{
  color:{T3}!important;font-size:9px!important;font-weight:600!important;
  text-transform:uppercase!important;letter-spacing:1.4px!important;
}}
section[data-testid="stSidebar"] .stSlider{{padding:0!important;}}
.sb-filter-section{{
  padding:8px 10px 0;font-size:9px;font-weight:600;color:{T3};
  text-transform:uppercase;letter-spacing:1.8px;margin:4px 0 8px 8px;
}}

/* ── MAIN CONTENT ── */
section[data-testid="stMain"], [data-testid="stMain"]{{border-left:1px solid {LINE}!important;}}

/* ── TOPBAR — Streamlit native buttons styled as tab pills ── */

/* Sticky topbar wrapper */
div[data-testid="stMain"] > div[data-testid="stVerticalBlock"] > div[data-testid="stVerticalBlock"] > div[data-testid="stHorizontalBlock"]:first-of-type {{
  position:sticky!important;top:0!important;z-index:100!important;
  background:{INK}!important;border-bottom:1px solid {LINE}!important;
  padding:3px 6px!important;margin:0!important;
}}

/* All tab pill buttons */
div[class*="st-key-topbtn_"] > div > button {{
  padding:4px 10px 4px 26px!important;border-radius:20px!important;
  font-family:'Poppins',sans-serif!important;font-size:11px!important;font-weight:500!important;
  cursor:pointer!important;white-space:nowrap!important;transition:all .18s!important;
  box-shadow:none!important;line-height:1.5!important;position:relative!important;
  min-height:28px!important;
}}
/* Inactive */
div[class*="st-key-topbtn_"] > div > button[kind="secondary"] {{
  background:transparent!important;border:1px solid transparent!important;color:{T3}!important;
}}
div[class*="st-key-topbtn_"] > div > button[kind="secondary"]:hover {{
  color:{T2}!important;border-color:{LINE}!important;background:transparent!important;box-shadow:none!important;
}}
/* Active */
div[class*="st-key-topbtn_"] > div > button[kind="primary"] {{
  background:{INK3}!important;border:1px solid {LINE2}!important;
  color:{T1}!important;box-shadow:none!important;
}}
div[class*="st-key-topbtn_"] > div > button[kind="primary"]:hover {{
  background:{INK3}!important;border-color:{LINE_HOVER}!important;box-shadow:none!important;
}}
div[class*="st-key-topbtn_"] > div > button:focus:not(:active) {{box-shadow:none!important;outline:none!important;}}

/* SVG icons for each topbar tab via CSS mask */
div[class*="st-key-topbtn_overview"] > div > button::before {{
  content:'';display:block;position:absolute;left:9px;top:50%;transform:translateY(-50%);
  width:10px;height:10px;background:currentColor;mask-size:contain;mask-repeat:no-repeat;mask-position:center;
  -webkit-mask-size:contain;-webkit-mask-repeat:no-repeat;-webkit-mask-position:center;
  -webkit-mask-image:url("data:image/svg+xml,%3Csvg xmlns='http://www.w3.org/2000/svg' viewBox='0 0 24 24' fill='none' stroke='currentColor' stroke-width='2'%3E%3Crect x='3' y='3' width='7' height='7' rx='1'/%3E%3Crect x='14' y='3' width='7' height='7' rx='1'/%3E%3Crect x='3' y='14' width='7' height='7' rx='1'/%3E%3Crect x='14' y='14' width='7' height='7' rx='1'/%3E%3C/svg%3E");
  mask-image:url("data:image/svg+xml,%3Csvg xmlns='http://www.w3.org/2000/svg' viewBox='0 0 24 24' fill='none' stroke='currentColor' stroke-width='2'%3E%3Crect x='3' y='3' width='7' height='7' rx='1'/%3E%3Crect x='14' y='3' width='7' height='7' rx='1'/%3E%3Crect x='3' y='14' width='7' height='7' rx='1'/%3E%3Crect x='14' y='14' width='7' height='7' rx='1'/%3E%3C/svg%3E");
}}
div[class*="st-key-topbtn_sentiment"] > div > button::before {{
  content:'';display:block;position:absolute;left:9px;top:50%;transform:translateY(-50%);
  width:10px;height:10px;background:currentColor;mask-size:contain;mask-repeat:no-repeat;mask-position:center;
  -webkit-mask-size:contain;-webkit-mask-repeat:no-repeat;-webkit-mask-position:center;
  -webkit-mask-image:url("data:image/svg+xml,%3Csvg xmlns='http://www.w3.org/2000/svg' viewBox='0 0 24 24' fill='none' stroke='currentColor' stroke-width='2'%3E%3Ccircle cx='12' cy='12' r='10'/%3E%3C/svg%3E");
  mask-image:url("data:image/svg+xml,%3Csvg xmlns='http://www.w3.org/2000/svg' viewBox='0 0 24 24' fill='none' stroke='currentColor' stroke-width='2'%3E%3Ccircle cx='12' cy='12' r='10'/%3E%3C/svg%3E");
}}
div[class*="st-key-topbtn_trend"] > div > button::before {{
  content:'';display:block;position:absolute;left:9px;top:50%;transform:translateY(-50%);
  width:10px;height:10px;background:currentColor;mask-size:contain;mask-repeat:no-repeat;mask-position:center;
  -webkit-mask-size:contain;-webkit-mask-repeat:no-repeat;-webkit-mask-position:center;
  -webkit-mask-image:url("data:image/svg+xml,%3Csvg xmlns='http://www.w3.org/2000/svg' viewBox='0 0 24 24' fill='none' stroke='currentColor' stroke-width='2'%3E%3Cpolyline points='23 6 13.5 15.5 8.5 10.5 1 18'/%3E%3Cpolyline points='17 6 23 6 23 12'/%3E%3C/svg%3E");
  mask-image:url("data:image/svg+xml,%3Csvg xmlns='http://www.w3.org/2000/svg' viewBox='0 0 24 24' fill='none' stroke='currentColor' stroke-width='2'%3E%3Cpolyline points='23 6 13.5 15.5 8.5 10.5 1 18'/%3E%3Cpolyline points='17 6 23 6 23 12'/%3E%3C/svg%3E");
}}
div[class*="st-key-topbtn_sna"] > div > button::before {{
  content:'';display:block;position:absolute;left:9px;top:50%;transform:translateY(-50%);
  width:10px;height:10px;background:currentColor;mask-size:contain;mask-repeat:no-repeat;mask-position:center;
  -webkit-mask-size:contain;-webkit-mask-repeat:no-repeat;-webkit-mask-position:center;
  -webkit-mask-image:url("data:image/svg+xml,%3Csvg xmlns='http://www.w3.org/2000/svg' viewBox='0 0 24 24' fill='none' stroke='currentColor' stroke-width='2'%3E%3Ccircle cx='6' cy='12' r='3'/%3E%3Ccircle cx='18' cy='5' r='3'/%3E%3Ccircle cx='18' cy='19' r='3'/%3E%3Cline x1='8.59' y1='13.51' x2='15.42' y2='17.49'/%3E%3Cline x1='15.41' y1='6.51' x2='8.59' y2='10.49'/%3E%3C/svg%3E");
  mask-image:url("data:image/svg+xml,%3Csvg xmlns='http://www.w3.org/2000/svg' viewBox='0 0 24 24' fill='none' stroke='currentColor' stroke-width='2'%3E%3Ccircle cx='6' cy='12' r='3'/%3E%3Ccircle cx='18' cy='5' r='3'/%3E%3Ccircle cx='18' cy='19' r='3'/%3E%3Cline x1='8.59' y1='13.51' x2='15.42' y2='17.49'/%3E%3Cline x1='15.41' y1='6.51' x2='8.59' y2='10.49'/%3E%3C/svg%3E");
}}
div[class*="st-key-topbtn_keywords"] > div > button::before {{
  content:'';display:block;position:absolute;left:9px;top:50%;transform:translateY(-50%);
  width:10px;height:10px;background:currentColor;mask-size:contain;mask-repeat:no-repeat;mask-position:center;
  -webkit-mask-size:contain;-webkit-mask-repeat:no-repeat;-webkit-mask-position:center;
  -webkit-mask-image:url("data:image/svg+xml,%3Csvg xmlns='http://www.w3.org/2000/svg' viewBox='0 0 24 24' fill='none' stroke='currentColor' stroke-width='2'%3E%3Cpath d='M20.59 13.41l-7.17 7.17a2 2 0 01-2.83 0L2 12V2h10l8.59 8.59a2 2 0 010 2.82z'/%3E%3Cline x1='7' y1='7' x2='7.01' y2='7'/%3E%3C/svg%3E");
  mask-image:url("data:image/svg+xml,%3Csvg xmlns='http://www.w3.org/2000/svg' viewBox='0 0 24 24' fill='none' stroke='currentColor' stroke-width='2'%3E%3Cpath d='M20.59 13.41l-7.17 7.17a2 2 0 01-2.83 0L2 12V2h10l8.59 8.59a2 2 0 010 2.82z'/%3E%3Cline x1='7' y1='7' x2='7.01' y2='7'/%3E%3C/svg%3E");
}}
div[class*="st-key-topbtn_pipeline"] > div > button::before {{
  content:'';display:block;position:absolute;left:9px;top:50%;transform:translateY(-50%);
  width:10px;height:10px;background:currentColor;mask-size:contain;mask-repeat:no-repeat;mask-position:center;
  -webkit-mask-size:contain;-webkit-mask-repeat:no-repeat;-webkit-mask-position:center;
  -webkit-mask-image:url("data:image/svg+xml,%3Csvg xmlns='http://www.w3.org/2000/svg' viewBox='0 0 24 24' fill='none' stroke='currentColor' stroke-width='2'%3E%3Cpolyline points='22 12 18 12 15 21 9 3 6 12 2 12'/%3E%3C/svg%3E");
  mask-image:url("data:image/svg+xml,%3Csvg xmlns='http://www.w3.org/2000/svg' viewBox='0 0 24 24' fill='none' stroke='currentColor' stroke-width='2'%3E%3Cpolyline points='22 12 18 12 15 21 9 3 6 12 2 12'/%3E%3C/svg%3E");
}}
div[class*="st-key-topbtn_tweets"] > div > button::before {{
  content:'';display:block;position:absolute;left:9px;top:50%;transform:translateY(-50%);
  width:10px;height:10px;background:currentColor;mask-size:contain;mask-repeat:no-repeat;mask-position:center;
  -webkit-mask-size:contain;-webkit-mask-repeat:no-repeat;-webkit-mask-position:center;
  -webkit-mask-image:url("data:image/svg+xml,%3Csvg xmlns='http://www.w3.org/2000/svg' viewBox='0 0 24 24' fill='none' stroke='currentColor' stroke-width='2'%3E%3Cpath d='M23 3a10.9 10.9 0 01-3.14 1.53 4.48 4.48 0 00-7.86 3v1A10.66 10.66 0 013 4s-4 9 5 13a11.64 11.64 0 01-7 2c9 5 20 0 20-11.5'/%3E%3C/svg%3E");
  mask-image:url("data:image/svg+xml,%3Csvg xmlns='http://www.w3.org/2000/svg' viewBox='0 0 24 24' fill='none' stroke='currentColor' stroke-width='2'%3E%3Cpath d='M23 3a10.9 10.9 0 01-3.14 1.53 4.48 4.48 0 00-7.86 3v1A10.66 10.66 0 013 4s-4 9 5 13a11.64 11.64 0 01-7 2c9 5 20 0 20-11.5'/%3E%3C/svg%3E");
}}

/* Live Preview badge */
.badge-live{{
  display:inline-flex;align-items:center;gap:5px;padding:4px 10px;border-radius:20px;
  background:rgba(245,90,0,0.08);border:1px solid rgba(245,90,0,0.2);
  font-size:10px;font-weight:600;color:{FIRE2};letter-spacing:.3px;white-space:nowrap;
}}
.dot-live{{width:5px;height:5px;border-radius:50%;background:{FIRE};animation:pulse 2s infinite;display:inline-block;}}
@keyframes pulse{{0%,100%{{opacity:1;}}50%{{opacity:.3;}}}}

/* ── TOPBAR column gap fix ── */
div[data-testid="stMain"] > div[data-testid="stVerticalBlock"] > div[data-testid="stVerticalBlock"] > div[data-testid="stHorizontalBlock"]:first-of-type [data-testid="stColumn"] {{
  padding:0 1px!important;
}}
div[data-testid="stMain"] > div[data-testid="stVerticalBlock"] > div[data-testid="stVerticalBlock"] > div[data-testid="stHorizontalBlock"]:first-of-type [data-testid="stHorizontalBlock"] {{
  gap:2px!important;
}}
/* Remove Streamlit button p tag extra spacing */
div[class*="st-key-topbtn_"] > div > button p,
section[data-testid="stSidebar"] div[class*="st-key-nav_"] > div > button p {{
  margin:0!important;padding:0!important;line-height:inherit!important;
}}

/* ── CONTENT ── */
.content-wrap{{padding:24px 26px 40px;}}
div[class*="st-key-content_wrap"]{{padding:24px 26px 40px;}}

/* ── ORANGE SHAPE SYSTEM ── */
.shape{{position:absolute;pointer-events:none;border-radius:3px;}}
.shape-sq{{width:26px;height:26px;background:{FIRE_DIM};border:1px solid rgba(245,90,0,0.2);transform:rotate(22deg);}}
.shape-sq-sm{{width:12px;height:12px;background:{FIRE_DIM};border:1px solid rgba(245,90,0,0.18);transform:rotate(45deg);}}
.shape-sq-lg{{width:42px;height:42px;background:{FIRE_GLOW};border:1px solid rgba(245,90,0,0.12);transform:rotate(-15deg);}}
.shape-dot{{width:5px;height:5px;border-radius:50%;background:{FIRE};opacity:0.45;}}
.shape-ring{{width:38px;height:38px;border-radius:50%;border:1.5px solid rgba(245,90,0,0.18);background:transparent;}}
.shape-bar{{width:2px;height:26px;background:linear-gradient(180deg,{FIRE},transparent);border-radius:2px;opacity:.6;}}
.shape-cross{{position:absolute;width:16px;height:16px;pointer-events:none;}}
.shape-cross::before,.shape-cross::after{{content:'';position:absolute;background:rgba(245,90,0,0.3);}}
.shape-cross::before{{width:16px;height:1.5px;top:7px;left:0;}}
.shape-cross::after{{width:1.5px;height:16px;top:0;left:7px;}}
.shape-tri{{width:0;height:0;border-left:9px solid transparent;border-right:9px solid transparent;border-bottom:16px solid rgba(245,90,0,0.15);pointer-events:none;position:absolute;}}

/* ── HERO ── */
.hero{{
  position:relative;overflow:hidden;
  background:{INK3};border:1px solid {LINE};border-radius:12px;
  padding:26px 30px;margin-bottom:18px;
}}
.hero-deco{{position:absolute;top:0;right:0;width:260px;height:100%;pointer-events:none;overflow:hidden;}}
.hero-deco::before{{
  content:'';position:absolute;top:-60px;right:-80px;width:200px;height:200px;
  background:radial-gradient(ellipse at center,rgba(245,90,0,0.08) 0%,transparent 65%);border-radius:50%;
}}
.hero-tag{{
  display:inline-flex;align-items:center;gap:6px;padding:3px 9px;border-radius:4px;
  background:{FIRE_DIM};border:1px solid rgba(245,90,0,0.2);
  font-size:9px;font-weight:600;color:{FIRE2};text-transform:uppercase;letter-spacing:1px;margin-bottom:12px;
}}
.hero-tag::before{{content:'';width:4px;height:4px;border-radius:50%;background:{FIRE};}}
.hero-h{{font-family:'Poppins',sans-serif;font-size:26px;font-weight:800;color:{T1};letter-spacing:-.5px;line-height:1.1;margin-bottom:8px;}}
.hero-h em{{font-style:normal;color:{FIRE};}}
.hero-p{{font-size:12.5px;color:{T2};max-width:480px;line-height:1.7;font-weight:300;}}

/* ── KPI ROW ── */
.kpi-row{{display:grid;grid-template-columns:repeat(4,1fr);gap:8px;margin-bottom:18px;}}
.kpi{{
  background:{INK2};border:1px solid {LINE};border-radius:10px;
  padding:14px 16px;position:relative;overflow:hidden;transition:border-color .2s;
}}
.kpi:hover{{border-color:{LINE_HOVER};}}
.kpi::after{{
  content:'';position:absolute;top:0;left:0;right:0;height:1.5px;border-radius:10px 10px 0 0;
  background:linear-gradient(90deg,{FIRE},transparent);opacity:0;transition:opacity .2s;
}}
.kpi:hover::after{{opacity:1;}}
.kpi-icon{{width:30px;height:30px;border-radius:7px;background:{FIRE_DIM};display:flex;align-items:center;justify-content:center;margin-bottom:10px;}}
.kpi-icon svg{{width:13px;height:13px;stroke:{FIRE};fill:none;stroke-width:1.8;stroke-linecap:round;stroke-linejoin:round;}}
.kpi-label{{font-size:9.5px;color:{T3};font-weight:500;text-transform:uppercase;letter-spacing:.8px;margin-bottom:3px;}}
.kpi-num{{font-family:'Poppins',sans-serif;font-size:26px;font-weight:700;color:{T1};line-height:1;letter-spacing:-.4px;}}
.kpi-sub{{font-size:10px;color:{T3};margin-top:3px;}}

/* ── GRID ── */
.g2{{display:grid;grid-template-columns:1fr 1fr;gap:10px;margin-bottom:10px;}}
.g3{{display:grid;grid-template-columns:1fr 1fr 1fr;gap:8px;margin-bottom:10px;}}
.g4{{display:grid;grid-template-columns:repeat(4,1fr);gap:8px;margin-bottom:12px;}}

/* ── CARD ── */
.card{{
  background:{INK2};border:1px solid {LINE};border-radius:10px;
  padding:16px 18px;position:relative;overflow:hidden;
  transition:border-color .2s;margin-bottom:10px;
}}
.card:hover{{border-color:{LINE_HOVER};}}
div[class*="st-key-card_"]{{background:{INK2};border:1px solid {LINE};border-radius:10px;padding:16px 18px;margin-bottom:10px;}}
div[class*="st-key-card_"]:hover{{border-color:{LINE_HOVER};}}
div[class*="st-key-card_"] [data-testid="stVerticalBlock"]{{gap:0!important;}}
.card-label{{font-size:9px;font-weight:600;color:{FIRE};text-transform:uppercase;letter-spacing:1.2px;margin-bottom:3px;}}
.card-title{{font-family:'Poppins',sans-serif;font-size:12.5px;font-weight:600;color:{T1};margin-bottom:12px;}}

/* ── INSIGHT STRIP ── */
.insight{{
  background:{INK3};border-left:2px solid {FIRE};border-radius:0 7px 7px 0;
  padding:10px 14px;font-size:12px;color:{T2};margin-bottom:14px;line-height:1.7;font-weight:400;
}}
.insight strong{{color:{T1};}}
.insight.muted{{border-left-color:{T3};background:{INK2};}}
.insight.red{{border-left-color:{RED_SOFT};background:rgba(248,81,73,0.04);}}

/* ── SECTION HEADER ── */
.sec-eye{{font-size:9px;font-weight:600;color:{FIRE};text-transform:uppercase;letter-spacing:1.2px;margin-bottom:3px;}}
.sec-ttl{{font-family:'Poppins',sans-serif;font-size:12.5px;font-weight:600;color:{T1};margin-bottom:10px;}}

/* ── CSS BAR ROW ── */
.bar-row{{display:flex;align-items:center;gap:8px;margin-bottom:6px;}}
.bar-label{{font-size:11px;color:{T2};width:92px;text-align:right;flex-shrink:0;letter-spacing:-.1px;}}
.bar-track{{flex:1;height:4px;background:{INK4};border-radius:3px;overflow:hidden;}}
.bar-fill{{height:100%;border-radius:3px;transition:width .6s;}}
.bar-val{{font-size:11px;color:{T3};width:44px;font-variant-numeric:tabular-nums;}}

/* ── METRIC ── */
.met{{background:{INK2};border:1px solid {LINE};border-radius:9px;padding:12px;text-align:center;transition:border-color .2s;}}
.met:hover{{border-color:{LINE_HOVER};}}
.met-v{{font-family:'Poppins',sans-serif;font-size:20px;font-weight:700;line-height:1;margin-bottom:3px;}}
.met-l{{font-size:9px;color:{T3};font-weight:500;text-transform:uppercase;letter-spacing:.8px;}}
.met-cap{{font-size:9px;color:{T3};margin-top:3px;}}

/* ── PIPELINE ── */
.pipe{{
  background:{INK2};border:1px solid {LINE};border-radius:10px;
  padding:14px 16px;margin-bottom:7px;display:flex;gap:14px;transition:border-color .2s;
}}
.pipe:hover{{border-color:{LINE_HOVER};}}
.pipe-num{{font-family:'Poppins',sans-serif;font-size:10.5px;font-weight:700;color:{FIRE};min-width:24px;padding-top:2px;}}
.pipe-ttl{{font-family:'Poppins',sans-serif;font-size:12.5px;font-weight:700;color:{T1};margin-bottom:4px;letter-spacing:-.1px;}}
.pipe-desc{{font-size:11.5px;color:{T2};line-height:1.65;margin-bottom:6px;font-weight:300;}}
.pipe-code{{
  font-size:10.5px;color:{T3};background:{INK};border-radius:5px;
  padding:6px 10px;font-family:'Poppins',monospace;line-height:1.5;border:1px solid {LINE};
}}

/* ── TWEET CARD ── */
.tweet{{
  background:{INK3};border:1px solid {LINE};border-left:2px solid {T3};border-radius:0 8px 8px 0;
  padding:12px 14px;margin-bottom:7px;font-size:12px;color:{T1};line-height:1.6;transition:border-color .2s;
}}
.tweet:hover{{border-color:{LINE_HOVER};border-left-color:{FIRE};}}
.tweet.fire{{border-left-color:{FIRE};}}
.tweet-meta{{margin-top:6px;display:flex;flex-wrap:wrap;gap:8px;font-size:10px;color:{T3};}}

/* ── BADGE ── */
.bdg{{display:inline-flex;align-items:center;gap:3px;border-radius:3px;padding:2px 6px;font-size:9px;font-weight:600;letter-spacing:.3px;}}
.bdg-pos{{background:rgba(245,90,0,0.12);color:{FIRE};}}
.bdg-neg{{background:rgba(255,255,255,0.06);color:{T2};}}
.bdg-neu{{background:rgba(255,255,255,0.04);color:{T3};}}

/* ── TABLE ── */
.tbl{{width:100%;border-collapse:collapse;font-size:11.5px;}}
.tbl th{{color:{T3};font-weight:500;text-transform:uppercase;font-size:9px;letter-spacing:.8px;padding:7px 10px;border-bottom:1px solid {LINE};text-align:left;}}
.tbl td{{padding:7px 10px;border-bottom:1px solid {LINE};color:{T2};}}
.tbl tr:last-child td{{border-bottom:none;}}
.tbl tr:hover td{{background:{INK3};}}
.tbl td:first-child{{color:{T3};}}

/* ── KEYWORD SEARCH ── */
.kw-search-wrap{{margin:6px 0 16px;display:flex;flex-direction:column;align-items:center;}}
.kw-search-label{{font-size:9px;font-weight:600;color:{T3};text-transform:uppercase;letter-spacing:1.4px;margin-bottom:12px;}}
.kw-search-label span{{color:{FIRE};margin:0 4px;}}
.kw-search-box{{
  position:relative;width:100%;max-width:560px;
  background:linear-gradient(180deg,{INK3},{INK2});
  border:1px solid {LINE2};border-radius:18px;padding:14px;
  box-shadow:0 0 0 1px rgba(0,0,0,.4),0 24px 60px -20px rgba(0,0,0,.8);
  transition:border-color .35s,box-shadow .35s;
}}
.kw-search-field{{
  display:flex;align-items:center;gap:10px;
  background:{INK};border:1px solid {LINE};border-radius:12px;padding:14px 16px;
}}
.kw-search-field input{{
  flex:1;background:none;border:none;outline:none;
  color:{T1};font-family:'Poppins',sans-serif;font-size:14px;font-weight:400;
}}
.kw-search-field input::placeholder{{color:{T3};font-weight:300;}}
.kw-chip{{
  display:inline-flex;align-items:center;gap:6px;font-size:10px;color:{T3};font-weight:400;
  background:{INK};border:1px solid {LINE};padding:5px 11px;border-radius:20px;
}}
.kw-search-foot{{
  display:flex;align-items:center;justify-content:space-between;
  margin-top:12px;padding:0 2px;
}}
.kw-search-btn{{
  font-family:'Poppins',sans-serif;font-size:11.5px;font-weight:600;color:{T1};cursor:pointer;
  background:{INK4};border:1px solid {LINE2};padding:7px 18px;border-radius:9px;
  transition:all .25s;
}}
.kw-search-btn:hover{{background:{FIRE};border-color:{FIRE};color:#fff;box-shadow:0 0 24px -6px rgba(245,90,0,.6);}}

/* ── DIVIDER ── */
.divider{{height:1px;background:{LINE};margin:16px 0;}}

/* ── WORDCLOUD PLACEHOLDER ── */
.wc-ph{{border:1px dashed {LINE2};border-radius:8px;padding:36px 20px;text-align:center;color:{T3};font-size:11.5px;line-height:1.7;}}
.wc-ph strong{{color:{T2};}}

/* ── DONUT WRAP ── */
.donut-wrap{{position:relative;display:flex;align-items:center;justify-content:center;}}
.donut-center{{position:absolute;text-align:center;pointer-events:none;}}
.donut-center .dn{{font-family:'Poppins',sans-serif;font-size:20px;font-weight:700;color:{T1};line-height:1;}}
.donut-center .ds{{font-size:9px;color:{T3};margin-top:2px;}}

/* ── LEGEND ── */
.legend{{display:flex;flex-wrap:wrap;gap:8px;margin-top:10px;}}
.leg-i{{display:flex;align-items:center;gap:5px;font-size:10.5px;color:{T2};}}
.leg-sq{{width:6px;height:6px;border-radius:1px;flex-shrink:0;}}

/* ── PLOTLY CONTAINER ── */
.js-plotly-plot .plotly,.plot-container{{background:transparent!important;}}

/* ── STREAMLIT OVERRIDES ── */
.stDataFrame{{border-radius:10px;overflow:hidden;}}
[data-testid="stDataFrameContainer"]{{border:1px solid {LINE};border-radius:10px;}}
.stDataFrame table{{font-size:11.5px!important;}}
[data-baseweb="select"]>div{{background:{INK3}!important;border-color:{LINE2}!important;color:{T1}!important;border-radius:7px!important;}}
[data-baseweb="tag"]{{background:{FIRE_DIM}!important;color:{FIRE2}!important;}}
.stAlert{{background:{INK2}!important;border:1px solid {LINE}!important;color:{T2}!important;border-radius:8px!important;}}
.stDownloadButton button{{
  background:{FIRE_DIM}!important;border:1px solid rgba(245,90,0,0.3)!important;
  color:{FIRE2}!important;border-radius:8px!important;font-weight:600!important;
  font-size:12px!important;padding:7px 16px!important;
}}
.stDownloadButton button:hover{{background:rgba(245,90,0,0.2)!important;}}
.streamlit-expanderHeader{{
  background:{INK2}!important;border:1px solid {LINE}!important;
  border-radius:8px!important;color:{T1}!important;font-weight:600!important;font-size:12px!important;
}}
.streamlit-expanderContent{{
  background:{INK2}!important;border:1px solid {LINE}!important;
  border-top:none!important;border-radius:0 0 8px 8px!important;padding:12px!important;
}}
input[data-testid]{{background:{INK3}!important;border:1px solid {LINE2}!important;color:{T1}!important;border-radius:7px!important;}}
label{{color:{T3}!important;font-size:9px!important;font-weight:600!important;text-transform:uppercase!important;letter-spacing:1.4px!important;}}

/* ── KILL STRAY BLUE ON ALL INTERACTIVE ELEMENTS ── */
button, [role="button"] {{color:inherit!important;}}

/* Download button */
.stDownloadButton button {{
  color:{FIRE2}!important;background:{FIRE_DIM}!important;
  border:1px solid rgba(245,90,0,0.3)!important;border-radius:8px!important;
  font-family:'Poppins',sans-serif!important;font-weight:600!important;font-size:12px!important;
  padding:7px 16px!important;
}}
.stDownloadButton button:hover {{background:rgba(245,90,0,0.2)!important;}}

/* Keyword chip buttons (in main content, not sidebar/topbar) */
div[class*="st-key-chip_"] button {{
  color:{T3}!important;background:{INK}!important;
  border:1px solid {LINE}!important;border-radius:20px!important;
  font-size:10px!important;padding:4px 10px!important;
  font-family:'Poppins',sans-serif!important;
}}
div[class*="st-key-chip_"] button:hover {{
  border-color:rgba(245,90,0,0.4)!important;color:{FIRE}!important;
}}
</style>
""", unsafe_allow_html=True)

# ─────────────────────────────────────────────
# PLOTLY DARK THEME
# ─────────────────────────────────────────────
PLOTLY_LAYOUT = dict(
    paper_bgcolor="rgba(0,0,0,0)",
    plot_bgcolor="rgba(0,0,0,0)",
    font=dict(family="Poppins, sans-serif", size=11, color=T2),
    margin=dict(l=14, r=14, t=14, b=14),  # t=14: no space for undefined title
    legend=dict(bgcolor="rgba(0,0,0,0)", borderwidth=0, font=dict(color=T2)),
    xaxis=dict(showgrid=True, gridcolor=LINE, linecolor=LINE, zerolinecolor=LINE, tickfont=dict(color=T3)),
    yaxis=dict(showgrid=True, gridcolor=LINE, linecolor=LINE, zerolinecolor=LINE, tickfont=dict(color=T3)),
    title=dict(text="", font=dict(family="Poppins, sans-serif", size=12.5, color=T1)),  # empty title = no "undefined"
    title_font=dict(family="Poppins, sans-serif", size=12.5, color=T1),
)

def themed(fig, h=320):
    fig.update_layout(**PLOTLY_LAYOUT, height=h)
    return fig

# ─────────────────────────────────────────────
# HELPERS
# ─────────────────────────────────────────────
_POS_MINI = {
    "baik","bagus","terima","kasih","bantu","membantu","senang","alhamdulillah",
    "syukur","bersyukur","mantap","semoga","dukung","mendukung","manfaat","cair",
    "lancar","adil","jujur","amanah","sukses","berhasil","tepat","peduli","sejahtera",
    "harap","berharap","apresiasi","hebat","setuju","aman","nyaman","cepat","mudah",
}
_NEG_MINI = {
    "korupsi","koruptor","tidak","gak","ga","susah","sulit","miskin","marah","kecewa",
    "buruk","jelek","gagal","lambat","telat","bohong","tipu","palsu","salah","protes",
    "demo","tolak","menolak","keluhan","mengeluh","curang","sengsara","menderita",
    "parah","bingung","ribet","mahal","kurang","nunggak","tunggak","janji","omong",
    "pungli","potong","dipotong","hilang","macet","sindir","capek",
}

def _mini_sent(text):
    s = 0
    for w in str(text).lower().split():
        if w in _POS_MINI: s += 1
        elif w in _NEG_MINI: s -= 1
    return s

# ─────────────────────────────────────────────
# STOPWORDS — replika notebook (cell 15) + Indonesia umum
# Dipakai untuk preprocessing TF-IDF agar konsisten dgn Colab.
# ─────────────────────────────────────────────
STOP_PROC = {
    "yg","ya","ga","gak","nggak","nya","sih","kok","deh","aja","kayak","dgn","utk","dll",
    "krn","sm","tp","jg","dr","sdh","udah","buat","apa","siapa","kapan","dimana","kenapa",
    "gimana","itu","ini","dan","atau","tapi","karena","jadi","ada","dari","ke","di","yang",
    "dengan","untuk","pada","adalah","mereka","kita","aku","kamu","dia","gw","gue","lu",
    "loe","gua","in","bgt","banget","kalo","shi","ku","bikin","amp","b","yaa","pengen","tuh",
    "a","an","orang","beneran","with","jin","si","kek","rt","nih","dah","ni","tu","klo","gitu",
    "gini","udh","blm","dpt","jdi","trus","terus","td","tdk","tak","bs","bisa","mau",
    # tambahan stopwords Indonesia umum (pengganti nltk.stopwords saat offline)
    "akan","juga","sudah","saya","anda","kami","kalian","tersebut","yaitu","sebagai","oleh",
    "lebih","tidak","saat","kalau","masih","setelah","sampai","hingga","agar","supaya","ketika",
    "telah","dapat","harus","seperti","hanya","sangat","sini","situ","sana","semua","setiap",
    "namun","sehingga","tetapi","maka","bahwa","atas","bawah","dalam","luar","antara","tanpa",
}

def _clean_proper(text):
    """Replika clean_text_proper() notebook cell 16."""
    text = str(text).lower()
    text = re.sub(r"http\S+|www\S+|https\S+", "", text, flags=re.MULTILINE)
    text = re.sub(r"\@\w+|\#\w+", "", text)
    text = re.sub(r"[^\w\s]", "", text)
    text = re.sub(r"\d+", "", text)
    words = text.split()
    return " ".join(w for w in words if w not in STOP_PROC and len(w) > 2)

def _load_inset_lexicon():
    """
    Load kamus InSet secara robust.
    Masalah sebelumnya: file InSet_pos.tsv/InSet_neg.tsv pada Colab biasanya tidak punya header,
    sedangkan dashboard lama membaca seolah-olah ada kolom `word` dan `weight`. Akibatnya
    kamus gagal terbaca, dashboard memakai fallback kecil, lalu banyak tweet menjadi Netral.
    """
    candidates_pos = [
        "InSet_pos.tsv", "klaster/InSet_pos.tsv", "data/InSet_pos.tsv", "/content/InSet_pos.tsv",
        "inset/InSet_pos.tsv", "/content/inset/InSet_pos.tsv"
    ]
    candidates_neg = [
        "InSet_neg.tsv", "klaster/InSet_neg.tsv", "data/InSet_neg.tsv", "/content/InSet_neg.tsv",
        "inset/InSet_neg.tsv", "/content/inset/InSet_neg.tsv"
    ]
    pos_path = next((path for path in candidates_pos if os.path.exists(path)), None)
    neg_path = next((path for path in candidates_neg if os.path.exists(path)), None)
    if not pos_path or not neg_path:
        return None, None

    def _read_lex(path):
        try:
            lex = pd.read_csv(path, sep="\t", header=None, engine="python")
            if lex.shape[1] < 2:
                lex = pd.read_csv(path, sep=",", header=None, engine="python")
            lex = lex.iloc[:, :2].copy()
            lex.columns = ["word", "weight"]
            lex["word"] = lex["word"].astype(str).str.lower().str.strip()
            lex["weight"] = pd.to_numeric(lex["weight"], errors="coerce")
            lex = lex.dropna(subset=["word", "weight"])
            lex = lex[(lex["word"] != "") & (lex["word"] != "word")]
            return dict(zip(lex["word"], lex["weight"]))
        except Exception:
            return {}

    pos = _read_lex(pos_path)
    neg = _read_lex(neg_path)
    if not pos or not neg:
        return None, None
    return pos, neg

def _norm_sentiment_value(value):
    """Samakan label InSet: Positive/Negative/Neutral atau Positif/Negatif/Netral."""
    if pd.isna(value):
        return np.nan
    s = str(value).strip().lower()
    if s in {"positive", "positif", "pos", "p", "1", "+1"} or s.startswith("pos"):
        return "positif"
    if s in {"negative", "negatif", "neg", "n", "-1"} or s.startswith("neg"):
        return "negatif"
    if s in {"neutral", "netral", "neu", "0"} or s.startswith("net") or s.startswith("neu"):
        return "netral"
    return s

def _label_from_score(score):
    try:
        score = float(score)
    except Exception:
        return "netral"
    if score > 0:
        return "positif"
    if score < 0:
        return "negatif"
    return "netral"

def _standardize_sentiment_columns(df):
    """Cari kolom sentiment dari hasil notebook dan ubah ke sentiment_label/sentiment_score."""
    df = df.copy()

    label_candidates = [
        "inset_sentiment", "sentiment_label", "sentimen", "Sentimen", "sentiment",
        "label_sentimen", "label_sentiment", "sentiment_inset", "kategori_sentimen"
    ]
    score_candidates = [
        "inset_score", "sentiment_score", "score_sentiment", "sentimen_score",
        "skor_sentimen", "score", "inset"
    ]

    for col in label_candidates:
        if col in df.columns and df[col].notna().any():
            df["sentiment_label"] = df[col].map(_norm_sentiment_value)
            break

    for col in score_candidates:
        if col in df.columns and df[col].notna().any():
            df["sentiment_score"] = pd.to_numeric(df[col], errors="coerce")
            break

    # Bila skor tersedia, skor lebih aman untuk menentukan label karena berasal langsung dari InSet.
    if "sentiment_score" in df.columns and df["sentiment_score"].notna().any():
        if (pd.to_numeric(df["sentiment_score"], errors="coerce").fillna(0) != 0).any():
            df["sentiment_label"] = df["sentiment_score"].apply(_label_from_score)

    if "sentiment_label" in df.columns:
        df["sentiment_label"] = df["sentiment_label"].map(_norm_sentiment_value).fillna("netral")

    return df

def _merge_inset_labeled_file(df):
    """
    Ambil hasil sentiment asli dari notebook: bansos_inset_labeled_final.csv.
    Ini menjaga dashboard sama dengan grafik InSet/Lexicon yang dibuat di Colab.
    """
    candidates = [
        "bansos_inset_labeled_final.csv",
        "/content/bansos_inset_labeled_final.csv",
        "klaster/bansos_inset_labeled_final.csv",
        "data/bansos_inset_labeled_final.csv",
        "bansos_inset_labeled.csv",
        "/content/bansos_inset_labeled.csv",
    ]
    path = next((path for path in candidates if os.path.exists(path)), None)
    if not path:
        return df

    try:
        sent = pd.read_csv(path)
        sent = _standardize_sentiment_columns(sent)
        keep = [c for c in ["id_str", "full_text", "clean_text", "sentiment_label", "sentiment_score", "matched_words"] if c in sent.columns]
        if "sentiment_label" not in keep and "sentiment_score" not in keep:
            return df
        sent = sent[keep].copy()

        out = df.copy()
        # Prioritas merge berdasarkan id_str, karena paling stabil.
        if "id_str" in out.columns and "id_str" in sent.columns:
            out["id_str"] = out["id_str"].astype(str)
            sent["id_str"] = sent["id_str"].astype(str)
            sent_small = sent.drop_duplicates("id_str")
            cols = ["id_str"] + [c for c in ["sentiment_label", "sentiment_score", "matched_words"] if c in sent_small.columns]
            out = out.merge(sent_small[cols], on="id_str", how="left", suffixes=("", "_inset_src"))
            for col in ["sentiment_label", "sentiment_score", "matched_words"]:
                src = f"{col}_inset_src"
                if src in out.columns:
                    out[col] = out[src].combine_first(out[col]) if col in out.columns else out[src]
                    out.drop(columns=[src], inplace=True)
            return _standardize_sentiment_columns(out)

        # Jika jumlah baris sama, data biasanya berasal dari dataframe yang sama di notebook.
        if len(out) == len(sent):
            if "sentiment_label" in sent.columns:
                out["sentiment_label"] = sent["sentiment_label"].values
            if "sentiment_score" in sent.columns:
                out["sentiment_score"] = sent["sentiment_score"].values
            if "matched_words" in sent.columns:
                out["matched_words"] = sent["matched_words"].values
            return _standardize_sentiment_columns(out)

        # Cadangan: merge berdasarkan clean_text bila tersedia.
        if "clean_text" in out.columns and "clean_text" in sent.columns:
            sent_small = sent.drop_duplicates("clean_text")
            cols = ["clean_text"] + [c for c in ["sentiment_label", "sentiment_score", "matched_words"] if c in sent_small.columns]
            out = out.merge(sent_small[cols], on="clean_text", how="left", suffixes=("", "_inset_src"))
            for col in ["sentiment_label", "sentiment_score", "matched_words"]:
                src = f"{col}_inset_src"
                if src in out.columns:
                    out[col] = out[src].combine_first(out[col]) if col in out.columns else out[src]
                    out.drop(columns=[src], inplace=True)
            return _standardize_sentiment_columns(out)

        return out
    except Exception:
        return df

def _recompute_sentiment_inset(df):
    """Hitung ulang sentiment dari clean_text/full_text bila CSV klaster belum membawa hasil InSet."""
    df = df.copy()
    text_source = "clean_text" if "clean_text" in df.columns else ("full_text" if "full_text" in df.columns else None)
    if not text_source:
        df["sentiment_score"] = 0
        df["sentiment_label"] = "netral"
        return df

    text_series = df[text_source].fillna("").astype(str)
    if text_source != "clean_text":
        text_series = text_series.apply(_clean_proper)

    pos_words, neg_words = _load_inset_lexicon()
    if pos_words is not None and neg_words is not None:
        lexicon = {}
        lexicon.update(pos_words)
        lexicon.update(neg_words)
        def _score(text):
            total = 0.0
            for token in str(text).lower().split():
                total += float(lexicon.get(token, 0))
            return total
        df["sentiment_score"] = text_series.apply(_score)
    else:
        df["sentiment_score"] = text_series.apply(_mini_sent)

    df["sentiment_label"] = df["sentiment_score"].apply(_label_from_score)
    return df

def _sentiment_needs_fix(df):
    if "sentiment_label" not in df.columns or not df["sentiment_label"].notna().any():
        return True
    labels = df["sentiment_label"].dropna().map(_norm_sentiment_value)
    if labels.empty:
        return True
    # Kasus yang terjadi di dashboard: semua/nyaris semua menjadi netral.
    vc = labels.value_counts(normalize=True)
    return (vc.index[0] == "netral" and vc.iloc[0] >= 0.98)

def _apply_best_sentiment(df):
    """
    Urutan perbaikan sentiment:
    1) pakai kolom InSet yang sudah ada,
    2) merge dari bansos_inset_labeled_final.csv,
    3) hitung ulang dengan InSet_pos.tsv/InSet_neg.tsv,
    4) fallback mini lexicon.
    """
    df = _standardize_sentiment_columns(df)
    if _sentiment_needs_fix(df):
        df = _merge_inset_labeled_file(df)
    if _sentiment_needs_fix(df):
        df = _recompute_sentiment_inset(df)
    df = _standardize_sentiment_columns(df)
    return df

def _process_raw(df):
    """
    Proses dataframe MENTAH (bansos.csv) menjadi siap-pakai, mereplikasi
    notebook: clean_text -> sentiment InSet -> TF-IDF+SVD+KMeans -> cluster_label
    -> tanggal. Dipakai saat hasil olahan (klaster/...csv) belum ada.
    """
    df = df.copy()

    # 1) Tanggal dari created_at
    if "created_at" in df.columns:
        dt = pd.to_datetime(df["created_at"], format="%a %b %d %H:%M:%S %z %Y", errors="coerce")
        if dt.isna().all():
            dt = pd.to_datetime(df["created_at"], errors="coerce")
        try:
            if getattr(dt.dt, "tz", None) is not None:
                dt = dt.dt.tz_localize(None)
        except Exception:
            pass
        df["tanggal"] = dt
    else:
        df["tanggal"] = pd.NaT

    # 2) Preprocessing teks
    text_src = "full_text" if "full_text" in df.columns else df.columns[0]
    df["clean_text"] = df[text_src].apply(_clean_proper)
    df = df[df["clean_text"].str.strip().astype(bool)].reset_index(drop=True)

    # 3) Numeric metrics
    for col in ("favorite_count","retweet_count","reply_count","quote_count"):
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce").fillna(0).astype(int)

    # 4) Sentimen — InSet lexicon kalau ada, fallback mini-lexicon
    pos_words, neg_words = _load_inset_lexicon()
    if pos_words is not None:
        def _inset(text):
            score = 0
            for w in str(text).split():
                if w in pos_words:   score += pos_words[w]
                elif w in neg_words: score += neg_words[w]
            return score
        df["sentiment_score"] = df["clean_text"].apply(_inset)
    else:
        df["sentiment_score"] = df["clean_text"].apply(_mini_sent)
    df["sentiment_label"] = df["sentiment_score"].apply(
        lambda s: "positif" if s > 0 else ("negatif" if s < 0 else "netral"))

    # 5) Klasterisasi TF-IDF + SVD + KMeans (replika notebook cell 18-20)
    cluster_done = False
    eval_metrics = {}
    try:
        from sklearn.feature_extraction.text import TfidfVectorizer
        from sklearn.decomposition import TruncatedSVD
        from sklearn.cluster import KMeans

        vectorizer = TfidfVectorizer(max_features=1000, min_df=2)
        X_tfidf = vectorizer.fit_transform(df["clean_text"])
        n_comp = min(2, X_tfidf.shape[1]-1) if X_tfidf.shape[1] > 1 else 1
        svd = TruncatedSVD(n_components=max(n_comp,1), random_state=42)
        X_2d = svd.fit_transform(X_tfidf)

        optimal_k = 3 if len(df) >= 3 else max(1, len(df))
        kmeans = KMeans(n_clusters=optimal_k, random_state=42, n_init=10)
        df["cluster"] = kmeans.fit_predict(X_2d)
        cluster_done = True

        # Simpan metrik untuk panel evaluasi
        try:
            from sklearn.metrics import silhouette_score, davies_bouldin_score, calinski_harabasz_score
            if df["cluster"].nunique() > 1:
                eval_metrics["silhouette"]        = float(silhouette_score(X_2d, df["cluster"]))
                eval_metrics["davies_bouldin"]    = float(davies_bouldin_score(X_2d, df["cluster"]))
                eval_metrics["calinski_harabasz"] = float(calinski_harabasz_score(X_2d, df["cluster"]))
            krange = list(range(2, min(8, len(df))))
            sils, wcss = [], []
            for k in krange:
                km = KMeans(n_clusters=k, random_state=42, n_init=10).fit(X_2d)
                sils.append(float(silhouette_score(X_2d, km.labels_)))
                wcss.append(float(km.inertia_))
            eval_metrics["k_range"]          = krange
            eval_metrics["silhouette_per_k"] = sils
            eval_metrics["wcss_per_k"]       = wcss
            eval_metrics["optimal_k"]        = optimal_k
        except Exception:
            pass
    except Exception:
        # sklearn tidak tersedia → cluster heuristik berbasis kata kunci
        cluster_done = False

    # 6) Penamaan klaster (replika notebook cell — cluster_label_map)
    # Notebook: {0: "Opini dan Keluhan Publik", 1: "Diskusi/Kebijakan Resmi Bansos", 2: "Informasi Program dan Penerima Bansos"}
    # cluster id KMeans tidak stabil, jadi nama ditentukan dari ISI tiap klaster via heuristik.
    FORMAL_WORDS = {
        "bantuan","program","penyaluran","kementerian","kemensos","pemerintah","resmi",
        "data","penerima","bpnt","pkh","kebijakan","anggaran","apbn","subsidi","dana",
        "sembako","kartu","verifikasi","terdaftar","pendataan","distribusi","menteri",
    }
    PROGRAM_WORDS = {
        "penerima","bpnt","pkh","bansos","sembako","kartu","prakerja","kks","cair",
        "terdaftar","pendataan","verifikasi","dtks","pip","kip","blt","cek","daftar",
        "penyaluran","bulog","inpres","kemensos","kemenkes","beras","sembako","keluarga",
    }
    if cluster_done:
        label_map = {}
        for c in sorted(df["cluster"].unique()):
            sub = df[df["cluster"] == c]
            words = Counter(w for t in sub["clean_text"] for w in t.split() if len(w) > 3)
            top = {w for w, _ in words.most_common(20)}
            formal_hits  = len(top & FORMAL_WORDS)
            program_hits = len(top & PROGRAM_WORDS)
            if formal_hits >= 3:
                label_map[c] = "Diskusi/Kebijakan Resmi Bansos"
            elif program_hits >= 3:
                label_map[c] = "Informasi Program dan Penerima Bansos"
            else:
                label_map[c] = "Opini dan Keluhan Publik"
        # Pastikan 3 label berbeda kalau ada 3 klaster
        unique_labels = list(dict.fromkeys(label_map.values()))
        ALL_LABELS = [
            "Opini dan Keluhan Publik",
            "Diskusi/Kebijakan Resmi Bansos",
            "Informasi Program dan Penerima Bansos",
        ]
        if len(set(label_map.values())) < len(label_map):
            # Ada duplikat — paksa setiap cluster ambil label berbeda berdasarkan ukuran
            sizes = df["cluster"].value_counts().sort_values(ascending=False)
            for rank, c in enumerate(sizes.index):
                label_map[c] = ALL_LABELS[rank % len(ALL_LABELS)]
        df["cluster_label"] = df["cluster"].map(label_map)
    else:
        # Tanpa sklearn: klasterkan via skor kata formal vs informal/keluhan.
        INFORMAL_WORDS = {
            "keluhan","mengeluh","susah","sulit","kecewa","marah","tolong","capek","capai",
            "nunggak","belum","tunggu","menunggu","janji","bohong","korupsi","koruptor","pungli",
            "potong","dipotong","sindir","protes","demo","parah","ribet","kapan","kok","mana",
        }
        def _score(text):
            toks = str(text).split()
            f   = sum(1 for w in toks if w in FORMAL_WORDS)
            p   = sum(1 for w in toks if w in PROGRAM_WORDS)
            inf = sum(1 for w in toks if w in INFORMAL_WORDS)
            return f - inf, p
        scores = df["clean_text"].apply(_score)
        def _label(s_pair):
            s, p = s_pair
            if s >= 2: return "Diskusi/Kebijakan Resmi Bansos"
            if p >= 2: return "Informasi Program dan Penerima Bansos"
            return "Opini dan Keluhan Publik"
        df["cluster_label"] = scores.apply(_label)
        df["cluster"] = df["cluster_label"].map({
            "Opini dan Keluhan Publik": 0,
            "Diskusi/Kebijakan Resmi Bansos": 1,
            "Informasi Program dan Penerima Bansos": 2,
        }).fillna(0).astype(int)

    # Simpan metrik ke atribut df (diambil load_metrics nanti)
    df.attrs["eval_metrics"] = eval_metrics
    return df

@st.cache_data
def load_data():
    """
    Prioritas:
    1. klaster/hasil_klasterisasi_bansos.csv (hasil olahan notebook) — pakai langsung
    2. bansos.csv (mentah) — proses internal lewat _process_raw()
    3. None — tidak ada data
    """
    processed_path = "klaster/hasil_klasterisasi_bansos.csv"
    raw_candidates = ["bansos.csv","data/bansos.csv","/content/bansos.csv","klaster/bansos.csv"]

    # 1) Hasil olahan
    if os.path.exists(processed_path):
        df = pd.read_csv(processed_path)
        # Pastikan tanggal terparse
        if "tanggal" in df.columns:
            df["tanggal"] = pd.to_datetime(df["tanggal"], errors="coerce")
        if ("tanggal" not in df.columns or df["tanggal"].isna().all()) and "created_at" in df.columns:
            df["tanggal"] = pd.to_datetime(df["created_at"], format="%a %b %d %H:%M:%S %z %Y", errors="coerce")
            if df["tanggal"].isna().all():
                df["tanggal"] = pd.to_datetime(df["created_at"], errors="coerce")
        if "tanggal" not in df.columns:
            df["tanggal"] = pd.NaT
        try:
            if getattr(df["tanggal"].dt, "tz", None) is not None:
                df["tanggal"] = df["tanggal"].dt.tz_localize(None)
        except Exception:
            pass
        for col in ("favorite_count","retweet_count","reply_count","quote_count"):
            if col in df.columns:
                df[col] = pd.to_numeric(df[col], errors="coerce").fillna(0).astype(int)
        # Normalisasi/perbaiki sentimen agar hasil dashboard sama dengan hasil InSet/Lexicon di notebook.
        # Dashboard akan membaca bansos_inset_labeled_final.csv jika ada, atau menghitung ulang memakai InSet_pos/InSet_neg.
        df = _apply_best_sentiment(df)

        # SNA: coba ambil in_reply_to_screen_name dari bansos.csv mentah
        # (tidak diexport di klaster CSV, tapi dibutuhkan panel SNA)
        if "in_reply_to_screen_name" not in df.columns:
            for rp in ["bansos.csv","data/bansos.csv","/content/bansos.csv","klaster/bansos.csv"]:
                if os.path.exists(rp):
                    try:
                        raw_sna = pd.read_csv(rp, usecols=["id_str","in_reply_to_screen_name","user_id_str","username"])
                        raw_sna["id_str"] = raw_sna["id_str"].astype(str)
                        df["id_str"] = df["id_str"].astype(str)
                        df = df.merge(raw_sna, on="id_str", how="left", suffixes=("","_raw"))
                    except Exception:
                        pass
                    break

        # Pastikan cluster_label tidak ada NaN
        if "cluster_label" in df.columns:
            df["cluster_label"] = df["cluster_label"].fillna("Tidak Masuk Klaster").astype(str)

        return df

    # 2) Data mentah → proses internal
    raw_path = next((p for p in raw_candidates if os.path.exists(p)), None)
    if raw_path:
        df_raw_in = pd.read_csv(raw_path)
        return _process_raw(df_raw_in)

    # 3) Tidak ada data
    return None

def load_metrics(df=None):
    path = "klaster/metrics_klaster.json"
    default = {
        "optimal_k": 3, "silhouette": 0.5276, "davies_bouldin": 0.5455,
        "calinski_harabasz": 586.78, "k_range": [2,3,4,5,6,7],
        "silhouette_per_k": [0.4760,0.5276,0.4934,0.4630,0.4876,0.5045],
        "wcss_per_k": None, "cluster_sizes": {"0":154,"1":353,"2":9}, "from_file": False,
    }
    if os.path.exists(path):
        try:
            with open(path, "r", encoding="utf-8") as f:
                loaded = json.load(f)
            default.update({k:v for k,v in loaded.items() if v is not None})
            default["from_file"] = True
        except Exception:
            pass
    # Override dgn metrik live hasil _process_raw() bila tersedia
    if df is not None:
        em = df.attrs.get("eval_metrics", {}) if hasattr(df, "attrs") else {}
        for k, v in em.items():
            if v is not None:
                default[k] = v
        # cluster_sizes aktual
        if "cluster" in df.columns:
            cs = df["cluster"].value_counts().sort_index()
            default["cluster_sizes"] = {str(k): int(v) for k, v in cs.items()}
    return default

def cluster_col(df):
    # Prefer cluster_label only if it actually has values; else fall back to numeric cluster
    if "cluster_label" in df.columns and df["cluster_label"].notna().any():
        # Isi NaN agar tidak muncul "undefined" di chart
        df["cluster_label"] = df["cluster_label"].fillna("Tidak Masuk Klaster").astype(str)
        return "cluster_label"
    if "cluster" in df.columns:
        df["cluster"] = df["cluster"].fillna(-1)
        return "cluster"
    return df.columns[0]

def text_col(df):
    for c in ("clean_text","full_text"):
        if c in df.columns: return c
    return df.columns[0]

STOPS = {
    "yang","dan","atau","dari","untuk","dengan","pada","ini","itu","ada","ke","di",
    "nya","yg","ga","gak","rt","amp","https","co","the","is","in","of","to","a","an",
    "be","are","was","were","will","can","aja","ya","juga","tapi","kalau","kalo",
    "jadi","bisa","pun","nih","lah","dong","sih","udah","sudah","buat","gitu","sama",
}

def top_words(series, n=20):
    words = []
    for text in series.dropna().astype(str):
        text = re.sub(r"[^a-zA-Z\s]", " ", text.lower())
        words.extend([w for w in text.split() if len(w) > 3 and w not in STOPS])
    return pd.DataFrame(Counter(words).most_common(n), columns=["Kata","Frekuensi"])

def fmt_n(n):
    if n >= 1_000_000: return f"{n/1_000_000:.1f}J"
    if n >= 1_000: return f"{n/1_000:.1f}K"
    return str(int(n))

def ins(text, variant=""):
    cls = f"insight{' '+variant if variant else ''}"
    return f'<div class="{cls}">{text}</div>'

def sec(eye, title, mt=0):
    return (f'<div class="sec-eye" style="margin-top:{mt}px;">{eye}</div>'
            f'<div class="sec-ttl">{title}</div>')

# ─────────────────────────────────────────────
# LOAD DATA
# ─────────────────────────────────────────────
df_raw = load_data()

if df_raw is None:
    st.markdown(ins(
        f"<strong>Data tidak ditemukan.</strong> Letakkan salah satu file berikut di folder yang sama dengan <code>app.py</code>:<br>"
        "• <code>klaster/hasil_klasterisasi_bansos.csv</code> (hasil olahan notebook), atau<br>"
        "• <code>bansos.csv</code> (data mentah hasil crawl) — akan diproses otomatis oleh dashboard.",
        "red"
    ), unsafe_allow_html=True)
    st.stop()

C_COL = cluster_col(df_raw)
T_COL = text_col(df_raw)
cm    = load_metrics(df_raw)
has_rt  = "retweet_count" in df_raw.columns
has_fav = "favorite_count" in df_raw.columns

# Label klaster yang bukan klaster utama K-Means — disembunyikan dari filter & tampilan
EXCLUDED_CLUSTER_LABELS = {"Tidak Masuk Klaster", "Lainnya", "Data tidak relevan", "-1", "-1.0"}

# ─────────────────────────────────────────────
# PANEL STATE — read directly from URL query param (?panel=X)
# Navigation links set ?panel=X with target="_self", so the
# active panel always reflects the URL. Falls back to overview.
# ─────────────────────────────────────────────
VALID_PANELS = {"overview","sentiment","trend","sna","keywords","pipeline","tweets"}
panel = st.query_params.get("panel", "overview")
if panel not in VALID_PANELS:
    panel = "overview"

# ─────────────────────────────────────────────
# DATE RANGE
# ─────────────────────────────────────────────
has_dates = df_raw["tanggal"].notna().any()
date_str = "—"
if has_dates:
    mn = df_raw["tanggal"].min(); mx = df_raw["tanggal"].max()
    if pd.notna(mn) and pd.notna(mx):
        date_str = f"{mn.strftime('%b')} – {mx.strftime('%b %Y')}"

# ─────────────────────────────────────────────
# PANELS CONFIG
# ─────────────────────────────────────────────
PANELS = [
    ("overview",  "overview", "Ringkasan"),
    ("sentiment", "smile",    "Sentimen"),
    ("trend",     "trending", "Tren"),
    ("sna",       "cluster",  "SNA"),
    ("keywords",  "tag",      "Kata Kunci"),
    ("pipeline",  "pipeline", "Metodologi"),
    ("tweets",    "tweet",    "Tweet"),
]

def _active(p): return " active" if panel == p else ""

# ─────────────────────────────────────────────
# NAV ITEMS
# ─────────────────────────────────────────────
nav_items_analisis = [
    ("overview",  "overview", "Ringkasan Klaster"),
    ("sentiment", "smile",    "Analisis Sentimen"),
    ("trend",     "trending", "Tren & Waktu"),
    ("sna",       "cluster",  "Social Network"),
    ("keywords",  "tag",      "Kata Kunci"),
]
nav_items_dokumen = [
    ("pipeline",  "pipeline", "Metodologi"),
    ("tweets",    "tweet",    "Data Tweet"),
]

# ─────────────────────────────────────────────
# SIDEBAR — matches HTML sidebar exactly
# Navigation uses <a href="?panel=X" target="_self">.
# target="_self" is the KEY: it navigates in the SAME tab
# (no new tab), exactly like the original HTML buttons.
# Streamlit re-runs on the URL change and reads ?panel= below.
# ─────────────────────────────────────────────
with st.sidebar:
    # ── Brand (sb-top) ──
    st.markdown(f"""
    <div class="sb-top">
      <div class="sb-logo">
        <div class="sb-icon">{svg("network",15,FIRE)}</div>
        <div class="sb-logo-text">
          <div class="sb-title">Bansos Insight</div>
          <div class="sb-sub">Dashboard Analisis</div>
        </div>
      </div>
    </div>
    <div class="sb-divider"></div>
    """, unsafe_allow_html=True)

    # ── Nav — real st.button() styled as nav-btn via CSS (HTML anchors unclickable) ──
    st.markdown('<div class="sb-section" style="margin-top:14px;">Analisis</div>', unsafe_allow_html=True)
    for _p, _ic, _lbl in nav_items_analisis:
        if st.button(_lbl, key=f"nav_{_p}", use_container_width=True,
                     type="primary" if panel==_p else "secondary"):
            st.query_params["panel"] = _p
            st.rerun()
    st.markdown('<div class="sb-section">Dokumen</div>', unsafe_allow_html=True)
    for _p, _ic, _lbl in nav_items_dokumen:
        if st.button(_lbl, key=f"nav_{_p}", use_container_width=True,
                     type="primary" if panel==_p else "secondary"):
            st.query_params["panel"] = _p
            st.rerun()
    # ── Filter Section ──
    st.markdown(f'<div style="height:1px;background:{LINE};margin:4px 18px 0;"></div>', unsafe_allow_html=True)
    st.markdown(f'<div class="sb-filter-section">{svg("filter",9,T3)}&nbsp; Filter</div>', unsafe_allow_html=True)

    all_cls = sorted([
        c for c in df_raw[C_COL].dropna().astype(str).unique().tolist()
        if c not in EXCLUDED_CLUSTER_LABELS
    ])
    sel_cls = st.multiselect("Klaster", options=all_cls, default=all_cls)

    sel_sent = []
    if "sentiment_label" in df_raw.columns and df_raw["sentiment_label"].notna().any():
        sent_order = ["positif", "negatif", "netral"]
        all_sent = [s for s in sent_order if s in set(df_raw["sentiment_label"].dropna().astype(str).str.lower())]
        all_sent += [s for s in sorted(df_raw["sentiment_label"].dropna().astype(str).str.lower().unique()) if s not in all_sent]
        sel_sent = st.multiselect("Sentimen", options=all_sent, default=all_sent)

    keyword_filter = st.text_input("Kata Kunci", placeholder="bansos, bantuan...", key="sidebar_keyword")
    date_range = None
    if has_dates:
        min_d = df_raw["tanggal"].min().date()
        max_d = df_raw["tanggal"].max().date()
        date_range = st.date_input("Periode", value=(min_d, max_d), min_value=min_d, max_value=max_d)
    min_rt = min_fav = 0
    if has_rt:
        max_rt = int(df_raw["retweet_count"].max())
        min_rt = st.slider("Min. Retweet", 0, max(max_rt,1), 0)
    if has_fav:
        max_fav = int(df_raw["favorite_count"].max())
        min_fav = st.slider("Min. Like", 0, max(max_fav,1), 0)

    # ── Footer Stats ──
    total_tw_all = len(df_raw[~df_raw[C_COL].astype(str).isin(EXCLUDED_CLUSTER_LABELS)])
    n_cls_all    = len(all_cls)
    st.markdown(f"""
    <div class="sb-footer">
      <div class="sb-footer-divider"></div>
      <div class="sb-stats">
        <div class="sb-stat">
          <div class="sb-stat-v">{fmt_n(total_tw_all)}</div>
          <div class="sb-stat-l">Total Tweet</div>
        </div>
        <div class="sb-stat">
          <div class="sb-stat-v" style="color:{FIRE};">{n_cls_all}</div>
          <div class="sb-stat-l">Klaster</div>
        </div>
        <div class="sb-stat wide">
          <div class="sb-stat-v">{date_str}</div>
          <div class="sb-stat-l">Periode data</div>
        </div>
      </div>
    </div>
    """, unsafe_allow_html=True)

# ─────────────────────────────────────────────
# APPLY FILTERS
# ─────────────────────────────────────────────
# Selalu exclude cluster bukan utama (cluster=-1, "Tidak Masuk Klaster", dll)
fdf = df_raw[~df_raw[C_COL].astype(str).isin(EXCLUDED_CLUSTER_LABELS)].copy()
if sel_cls:
    fdf = fdf[fdf[C_COL].astype(str).isin(sel_cls)]
if "sentiment_label" in fdf.columns and sel_sent:
    fdf = fdf[fdf["sentiment_label"].astype(str).str.lower().isin(sel_sent)]
if keyword_filter:
    fdf = fdf[fdf[T_COL].astype(str).str.contains(keyword_filter, case=False, na=False)]
if date_range and isinstance(date_range, tuple) and len(date_range) == 2:
    s_d, e_d = date_range
    fdf = fdf[(fdf["tanggal"].dt.date >= s_d) & (fdf["tanggal"].dt.date <= e_d)]
if has_rt:
    fdf = fdf[fdf["retweet_count"] >= min_rt]
if has_fav:
    fdf = fdf[fdf["favorite_count"] >= min_fav]

# ─────────────────────────────────────────────
# TOPBAR — Streamlit native buttons (ONLY reliable clickable method)
# HTML <a href> inside st.markdown is blocked by Streamlit overlay divs.
# st.button() receives click events reliably and we style them to match HTML.
# ─────────────────────────────────────────────
date_disp = "—"
if has_dates:
    mn = df_raw["tanggal"].min(); mx = df_raw["tanggal"].max()
    if pd.notna(mn) and pd.notna(mx):
        date_disp = f"{mn.strftime('%d %b')} – {mx.strftime('%d %b %Y')}"

# Real clickable buttons as topbar tabs
_PANEL_TAB_DEFS = [
    ("overview","Ringkasan"),("sentiment","Sentimen"),("trend","Tren"),
    ("sna","SNA"),("keywords","Kata Kunci"),("pipeline","Metodologi"),("tweets","Tweet"),
]
# 7 equal tab cols + 1 wider badge col
_tcols = st.columns([1,1,1,1,1.2,1,0.8,1.5])
for _tc, (_tp, _tlbl) in zip(_tcols[:7], _PANEL_TAB_DEFS):
    with _tc:
        if st.button(_tlbl, key=f"topbtn_{_tp}", use_container_width=True,
                     type="primary" if panel==_tp else "secondary"):
            st.query_params["panel"] = _tp
            st.rerun()
with _tcols[7]:
    st.markdown(
        f'<div style="display:flex;align-items:center;justify-content:flex-end;height:36px;padding-right:4px;">',
        unsafe_allow_html=True
    )

# ─────────────────────────────────────────────
# CONTENT WRAPPER
# ─────────────────────────────────────────────
with st.container(key="content_wrap"):

    if fdf.empty and panel != "pipeline":
        st.markdown(ins("Tidak ada data yang cocok dengan filter yang dipilih. Gunakan sidebar untuk mengubah filter.", "muted"), unsafe_allow_html=True)

    # ══════════════════════════════════════════════
    # PANEL 0 — RINGKASAN KLASTER
    # ══════════════════════════════════════════════
    elif panel == "overview":
        # HERO
        st.markdown(f"""
        <div class="hero">
          <div class="hero-deco"></div>
          <div class="shape shape-sq" style="top:16px;right:118px;"></div>
          <div class="shape shape-cross" style="top:38px;right:78px;"></div>
          <div class="shape shape-sq-sm" style="bottom:18px;right:155px;"></div>
          <div class="shape shape-dot" style="bottom:24px;right:98px;"></div>
          <div class="shape shape-ring" style="top:-10px;right:28px;"></div>
          <div class="shape shape-tri" style="bottom:12px;right:54px;"></div>
          <div class="shape shape-bar" style="top:16px;right:188px;"></div>
          <div class="hero-h">Bansos <em>Insight</em><br>Dashboard</div>
          <div class="hero-p">Eksplorasi percakapan publik seputar bantuan sosial — sentimen InSet Lexicon, klasterisasi TF-IDF K-Means, tren waktu, dan Social Network Analysis.</div>
        </div>
        """, unsafe_allow_html=True)

        # KPI
        total_tw  = len(fdf)
        n_cls     = fdf[C_COL].nunique()
        total_rt  = int(fdf["retweet_count"].sum()) if has_rt else 0
        total_fav = int(fdf["favorite_count"].sum()) if has_fav else 0

        st.markdown(f"""
        <div class="kpi-row">
          <div class="kpi">
            <div class="shape shape-sq-sm" style="top:10px;right:12px;"></div>
            <div class="kpi-icon">{svg("tweet",13,FIRE)}</div>
            <div class="kpi-label">Total Tweet</div>
            <div class="kpi-num">{fmt_n(total_tw)}</div>
            <div class="kpi-sub">{total_tw:,} setelah filter</div>
          </div>
          <div class="kpi">
            <div class="kpi-icon">{svg("cluster",13,FIRE)}</div>
            <div class="kpi-label">Klaster Aktif</div>
            <div class="kpi-num">{n_cls}</div>
            <div class="kpi-sub">Kategori diskusi</div>
          </div>
          <div class="kpi">
            <div class="kpi-icon">{svg("retweet",13,FIRE)}</div>
            <div class="kpi-label">Total Retweet</div>
            <div class="kpi-num">{fmt_n(total_rt)}</div>
            <div class="kpi-sub">~{total_rt/max(total_tw,1):.1f} per tweet</div>
          </div>
          <div class="kpi">
            <div class="kpi-icon">{svg("heart",13,FIRE)}</div>
            <div class="kpi-label">Total Like</div>
            <div class="kpi-num">{fmt_n(total_fav)}</div>
            <div class="kpi-sub">~{total_fav/max(total_tw,1):.1f} per tweet</div>
          </div>
        </div>
        """, unsafe_allow_html=True)

        # Cluster distribution
        cl_count = fdf[C_COL].value_counts().reset_index()
        cl_count.columns = ["Klaster","Jumlah"]
        total = cl_count["Jumlah"].sum()
        cl_count["Persen"] = (cl_count["Jumlah"] / total * 100).round(1)
        dom = cl_count.iloc[0]

        st.markdown(ins(
            f'Dari <strong>{total:,} tweet</strong>, klaster dominan adalah '
            f'<strong>{dom["Klaster"]}</strong> dengan <strong>{dom["Persen"]}%</strong> '
            f'({dom["Jumlah"]:,} tweet).'
        ), unsafe_allow_html=True)

        col_pie, col_bars = st.columns(2)
        with col_pie:
            with st.container(key="card_pie"):
                st.markdown('<div class="card-label">Komposisi</div><div class="card-title">Proporsi Diskusi per Klaster</div>', unsafe_allow_html=True)
                fig_pie = px.pie(cl_count, names="Klaster", values="Jumlah", hole=0.54,
                                 color_discrete_sequence=CLUSTER_PALETTE)
                fig_pie.update_traces(textinfo="none",
                                      marker=dict(line=dict(color=INK, width=2.5)))
                fig_pie.update_layout(**{**PLOTLY_LAYOUT, "showlegend":False, "height":200, "margin":dict(l=8,r=8,t=8,b=8)})
                fig_pie.add_annotation(
                    text=f"<b>{total:,}</b><br><span style='font-size:9px;color:{T3};'>total tweet</span>",
                    x=0.5, y=0.5, showarrow=False, font=dict(size=20, color=T1, family="Poppins,sans-serif"),
                    align="center",
                )
                st.plotly_chart(fig_pie, use_container_width=True)
                leg_items = "".join(
                    f'<div class="leg-i"><div class="leg-sq" style="background:{CLUSTER_PALETTE[i%len(CLUSTER_PALETTE)]};"></div>'
                    f'{row["Klaster"]} ({row["Persen"]}%)</div>'
                    for i, row in cl_count.iterrows()
                )
                st.markdown(f'<div class="legend">{leg_items}</div>', unsafe_allow_html=True)

        with col_bars:
            with st.container(key="card_bars"):
                st.markdown('<div class="card-label">Distribusi</div><div class="card-title">Tweet & Engagement per Klaster</div>', unsafe_allow_html=True)
                max_j = cl_count["Jumlah"].max()
                bars_html = ""
                for i, row in cl_count.iterrows():
                    w = row["Jumlah"] / max_j * 100
                    bars_html += f'<div class="bar-row"><div class="bar-label">{row["Klaster"]}</div><div class="bar-track"><div class="bar-fill" style="width:{w:.1f}%;background:{CLUSTER_PALETTE[i%len(CLUSTER_PALETTE)]};"></div></div><div class="bar-val">{row["Jumlah"]:,}</div></div>'
                st.markdown(bars_html, unsafe_allow_html=True)

                if has_rt or has_fav:
                    st.markdown(f'<div style="margin-top:16px;font-size:9px;font-weight:600;color:{FIRE};text-transform:uppercase;letter-spacing:1.2px;margin-bottom:7px;">Engagement per Klaster</div>', unsafe_allow_html=True)
                    eng_cols = [C_COL]
                    if has_rt: eng_cols.append("retweet_count")
                    if has_fav: eng_cols.append("favorite_count")
                    eng_df = fdf[eng_cols].copy()
                    eng_df["eng"] = 0
                    if has_rt: eng_df["eng"] += eng_df["retweet_count"]
                    if has_fav: eng_df["eng"] += eng_df["favorite_count"]
                    eng_agg = eng_df.groupby(C_COL)["eng"].sum().sort_values(ascending=False)
                    max_e = eng_agg.max()
                    eb_html = ""
                    for i, (cl, ev) in enumerate(eng_agg.items()):
                        w = ev / max(max_e, 1) * 100
                        bg = f"linear-gradient(90deg,{FIRE},{FIRE2})" if i == 0 else CLUSTER_PALETTE[min(i,len(CLUSTER_PALETTE)-1)]
                        eb_html += f'<div class="bar-row"><div class="bar-label">{cl}</div><div class="bar-track"><div class="bar-fill" style="width:{w:.1f}%;background:{bg};"></div></div><div class="bar-val">{fmt_n(ev)}</div></div>'
                    st.markdown(eb_html, unsafe_allow_html=True)

        # K-Means evaluation metrics
        st.markdown('<div class="divider"></div>', unsafe_allow_html=True)
        st.markdown(sec("Evaluasi K-Means","Metrik Kualitas Klasterisasi"), unsafe_allow_html=True)
        sil_v = cm.get("silhouette"); dbi_v = cm.get("davies_bouldin")
        chi_v = cm.get("calinski_harabasz"); k_v = cm.get("optimal_k")
        st.markdown(f"""
        <div class="g3">
          <div class="met"><div class="met-v" style="color:{FIRE};">{"%.4f"%sil_v if sil_v else "—"}</div><div class="met-l">Silhouette Score</div><div class="met-cap">baik jika &gt; 0.5</div></div>
          <div class="met"><div class="met-v" style="color:{T2};">{"%.4f"%dbi_v if dbi_v else "—"}</div><div class="met-l">Davies-Bouldin</div><div class="met-cap">baik jika &lt; 1.0</div></div>
          <div class="met"><div class="met-v" style="color:{T1};">{"%.0f"%chi_v if chi_v else "—"}</div><div class="met-l">Calinski-Harabasz</div><div class="met-cap">makin besar makin baik</div></div>
        </div>
        """, unsafe_allow_html=True)

        # ── Scatter Plot K-Means SVD 2D + Centroid (replika notebook cell "K-Means clustering final") ──
        st.markdown('<div class="divider"></div>', unsafe_allow_html=True)
        st.markdown(sec("Visualisasi K-Means","Scatter Plot SVD 2D — Klasterisasi K-Means"), unsafe_allow_html=True)
        st.markdown(ins(
            "Visualisasi hasil K-Means clustering setelah reduksi dimensi TF-IDF → SVD (2 komponen). "
            "Setiap titik adalah satu tweet. Tanda <strong>✕</strong> menunjukkan posisi centroid tiap klaster."
        ), unsafe_allow_html=True)
        try:
            from sklearn.feature_extraction.text import TfidfVectorizer as _TFV
            from sklearn.decomposition import TruncatedSVD as _SVD
            from sklearn.cluster import KMeans as _KM
            _texts_sc = fdf[T_COL].fillna("").astype(str).tolist()
            _texts_sc = [t for t in _texts_sc if t.strip()]
            if len(_texts_sc) >= 6:
                _vec_sc = _TFV(max_features=1000, min_df=2)
                _Xsc = _vec_sc.fit_transform(_texts_sc)
                _n_comp = min(2, _Xsc.shape[1] - 1)
                _svd_sc = _SVD(n_components=max(_n_comp, 1), random_state=42)
                _X2d_sc = _svd_sc.fit_transform(_Xsc)
                _k_sc   = int(k_v) if k_v else 3
                _k_sc   = min(_k_sc, len(_texts_sc) - 1)
                _km_sc  = _KM(n_clusters=_k_sc, random_state=42, n_init=10)
                _lbls_sc = _km_sc.fit_predict(_X2d_sc)
                _cents_sc = _km_sc.cluster_centers_

                # Map numeric cluster labels ke cluster_label names if possible
                _cl_labels = fdf[C_COL].values[:len(_lbls_sc)]
                _scatter_df = pd.DataFrame({
                    "SVD 1": _X2d_sc[:, 0],
                    "SVD 2": _X2d_sc[:, 1] if _X2d_sc.shape[1] > 1 else [0.0] * len(_X2d_sc),
                    "Klaster": [str(l) for l in _lbls_sc],
                })
                with st.container(key="card_kmeans_scatter"):
                    st.markdown('<div class="card-label">K-Means</div><div class="card-title">Visualisasi K-Means Clustering — Diskusi Bansos di Twitter/X</div>', unsafe_allow_html=True)
                    fig_sc = px.scatter(
                        _scatter_df, x="SVD 1", y="SVD 2", color="Klaster",
                        color_discrete_sequence=CLUSTER_PALETTE,
                        opacity=0.7,
                    )
                    fig_sc.update_traces(marker=dict(size=5))
                    # Tambah centroid marker ✕
                    for _ci, _cpt in enumerate(_cents_sc):
                        fig_sc.add_trace(go.Scatter(
                            x=[_cpt[0]],
                            y=[_cpt[1] if _cents_sc.shape[1] > 1 else 0.0],
                            mode="markers+text",
                            marker=dict(symbol="x", size=16, color=T1,
                                        line=dict(width=2.5, color=T1)),
                            text=["✕"],
                            textposition="top center",
                            textfont=dict(color=T1, size=11),
                            name=f"Centroid {_ci}",
                            showlegend=True,
                        ))
                    fig_sc.update_layout(**{
                        **PLOTLY_LAYOUT,
                        "height": 380,
                        "xaxis_title": "SVD Komponen 1",
                        "yaxis_title": "SVD Komponen 2",
                        "legend": dict(bgcolor="rgba(0,0,0,0)", font=dict(color=T2, size=10)),
                    })
                    st.plotly_chart(fig_sc, use_container_width=True)
            else:
                st.markdown(ins("Data tidak cukup untuk scatter plot K-Means.", "muted"), unsafe_allow_html=True)
        except ImportError:
            st.markdown(ins("scikit-learn belum terinstall — scatter plot K-Means tidak tersedia.", "muted"), unsafe_allow_html=True)
        except Exception as _e_sc:
            st.markdown(ins(f"Scatter plot K-Means tidak dapat dibuat: <code>{_e_sc}</code>", "muted"), unsafe_allow_html=True)

        # Cluster sample tweets
        st.markdown('<div class="divider"></div>', unsafe_allow_html=True)
        st.markdown(sec("Contoh Tweet","Preview 5 Tweet per Klaster"), unsafe_allow_html=True)
        for i, cl in enumerate(sorted(fdf[C_COL].dropna().astype(str).unique())):
            sub_cl = fdf[fdf[C_COL].astype(str) == cl]
            kw_cl  = top_words(sub_cl[T_COL], n=6)
            kw_str = ", ".join(kw_cl["Kata"].tolist()) if not kw_cl.empty else "—"
            clr = CLUSTER_PALETTE[i % len(CLUSTER_PALETTE)]
            with st.expander(f"Klaster: {cl}  ·  {len(sub_cl):,} tweet", expanded=False):
                st.markdown(
                    f'<div style="font-size:11px;color:{T3};margin-bottom:10px;">'
                    f'{svg("tag",11,T3)} <strong style="color:{T2};">Kata dominan:</strong> {kw_str}</div>',
                    unsafe_allow_html=True
                )
                for _, row in sub_cl[[T_COL]].head(5).iterrows():
                    st.markdown(f'<div class="tweet fire">{row[T_COL]}</div>', unsafe_allow_html=True)

        # ── Hierarchical Clustering Dendrogram — replika notebook cell "hierarchical clustering" ──
        st.markdown('<div class="divider"></div>', unsafe_allow_html=True)
        st.markdown(sec("Hierarchical Clustering","Dendrogram — 40 Data Pertama"), unsafe_allow_html=True)
        st.markdown(ins(
            "Dendrogram menampilkan struktur pengelompokan hierarki (Ward linkage) untuk 40 tweet pertama. "
            "Sumbu Y menunjukkan jarak Euclidean antar kelompok — semakin tinggi cabang, semakin berbeda dua kelompok tersebut."
        ), unsafe_allow_html=True)
        try:
            from sklearn.feature_extraction.text import TfidfVectorizer
            from sklearn.decomposition import TruncatedSVD
            from scipy.cluster.hierarchy import dendrogram as _dendrogram, linkage as _linkage
            import plotly.figure_factory as ff

            _texts_dend = fdf[T_COL].fillna("").astype(str).head(40).tolist()
            _texts_dend = [t for t in _texts_dend if t.strip()]
            if len(_texts_dend) >= 4:
                _vec_d = TfidfVectorizer(max_features=500, min_df=1)
                _Xt = _vec_d.fit_transform(_texts_dend)
                _svd_d = TruncatedSVD(n_components=min(2, _Xt.shape[1]-1), random_state=42)
                _X2d_d = _svd_d.fit_transform(_Xt)
                _Z = _linkage(_X2d_d, method="ward")

                # Build dendrogram via scipy to get leaf ordering
                _dend_data = _dendrogram(_Z, no_plot=True)
                _ivl = _dend_data["ivl"]
                _dcoord = _dend_data["dcoord"]
                _icoord = _dend_data["icoord"]

                # Plot via Plotly Scatter
                _fig_dend = go.Figure()
                for xs, ys in zip(_icoord, _dcoord):
                    _fig_dend.add_trace(go.Scatter(
                        x=xs, y=ys,
                        mode="lines",
                        line=dict(color=FIRE, width=1.2),
                        hoverinfo="skip",
                        showlegend=False,
                    ))
                _fig_dend.update_layout(**{
                    **PLOTLY_LAYOUT,
                    "height": 300,
                    "xaxis_title": "Indeks Data",
                    "yaxis_title": "Jarak (Euclidean)",
                    "xaxis": dict(showgrid=False, zeroline=False, showticklabels=False,
                                  linecolor=LINE, gridcolor=LINE, tickfont=dict(color=T3)),
                    "yaxis": dict(showgrid=True, gridcolor=LINE, linecolor=LINE, zeroline=False,
                                  tickfont=dict(color=T3)),
                    "title": dict(text="Dendrogram Hierarchical Clustering (40 Data Pertama)",
                                  font=dict(size=12, color=T1)),
                })
                with st.container(key="card_overview_dendro"):
                    st.plotly_chart(_fig_dend, use_container_width=True)
            else:
                st.markdown(ins("Data tidak cukup untuk membuat dendrogram (min 4 baris).", "muted"), unsafe_allow_html=True)
        except ImportError as e_imp:
            st.markdown(ins(f"Library <code>{e_imp}</code> belum tersedia untuk dendrogram.", "muted"), unsafe_allow_html=True)
        except Exception as e_dend:
            st.markdown(ins(f"Dendrogram tidak dapat dibuat: <code>{e_dend}</code>", "muted"), unsafe_allow_html=True)

    # ══════════════════════════════════════════════
    # PANEL 1 — ANALISIS SENTIMEN
    # ══════════════════════════════════════════════
    elif panel == "sentiment":
        has_sent = "sentiment_label" in fdf.columns
        if not has_sent:
            st.markdown(ins("Kolom <code>sentiment_label</code> belum tersedia.", "muted"), unsafe_allow_html=True)
        else:
            sc = fdf["sentiment_label"].value_counts().reset_index()
            sc.columns = ["Sentimen","Jumlah"]
            sc["Persen"] = (sc["Jumlah"] / sc["Jumlah"].sum() * 100).round(1)
            dom_s = sc.iloc[0]
            tone = "kritis / negatif" if "neg" in dom_s["Sentimen"] else ("mendukung / positif" if "pos" in dom_s["Sentimen"] else "netral")
            st.markdown(ins(
                f'Dari <strong>{len(fdf):,} tweet</strong>, sentimen dominan adalah '
                f'<strong>{dom_s["Sentimen"].capitalize()}</strong> ({dom_s["Persen"]}% · {dom_s["Jumlah"]:,} tweet) — '
                f'mengindikasikan persepsi publik yang bersifat <strong>{tone}</strong> terhadap program bansos.'
            ), unsafe_allow_html=True)

            col_s1, col_s2 = st.columns(2)
            with col_s1:
                with st.container(key="card_sent_pie"):
                    st.markdown('<div class="card-label">Distribusi</div><div class="card-title">Proporsi Sentimen (Keseluruhan)</div>', unsafe_allow_html=True)
                    colors_s = [SENTIMENT_COLORS.get(s, T2) for s in sc["Sentimen"]]
                    fig_sdnt = px.pie(sc, names="Sentimen", values="Jumlah", hole=0.55,
                                      color_discrete_sequence=colors_s)
                    fig_sdnt.update_traces(textinfo="none",
                                           marker=dict(line=dict(color=INK, width=2.5)))
                    fig_sdnt.update_layout(**{**PLOTLY_LAYOUT, "showlegend":False, "height":200, "margin":dict(l=8,r=8,t=8,b=8)})
                    fig_sdnt.add_annotation(
                        text=f"<b>{int(sc['Jumlah'].sum()):,}</b><br><span style='font-size:9px;color:{T3};'>total tweet</span>",
                        x=0.5, y=0.5, showarrow=False, font=dict(size=20, color=T1, family="Poppins,sans-serif"),
                        align="center",
                    )
                    st.plotly_chart(fig_sdnt, use_container_width=True)
                    leg2 = "".join(
                        f'<div class="leg-i"><div class="leg-sq" style="background:{colors_s[i]};"></div>{row["Sentimen"].capitalize()} {row["Persen"]}%</div>'
                        for i, row in sc.iterrows()
                    )
                    st.markdown(f'<div class="legend">{leg2}</div>', unsafe_allow_html=True)

            with col_s2:
                with st.container(key="card_sent_vol"):
                    st.markdown('<div class="card-label">Volume</div><div class="card-title">Tweet per Sentimen</div>', unsafe_allow_html=True)
                    max_s = sc["Jumlah"].max()
                    for i, row in sc.iterrows():
                        w = row["Jumlah"] / max_s * 100
                        clr = SENTIMENT_COLORS.get(row["Sentimen"], T3)
                        st.markdown(
                            f'<div class="bar-row">'
                            f'<div class="bar-label" style="color:{clr};">{row["Sentimen"].capitalize()}</div>'
                            f'<div class="bar-track"><div class="bar-fill" style="width:{w:.1f}%;background:{clr};"></div></div>'
                            f'<div class="bar-val">{row["Jumlah"]:,}</div></div>',
                            unsafe_allow_html=True
                        )
                    if has_rt:
                        st.markdown(f'<div style="margin-top:16px;font-size:9px;font-weight:600;color:{FIRE};text-transform:uppercase;letter-spacing:1.2px;margin-bottom:7px;">Rata-rata Retweet per Sentimen</div>', unsafe_allow_html=True)
                        rt_by_sent = fdf.groupby("sentiment_label")["retweet_count"].mean().sort_values(ascending=False)
                        max_r2 = rt_by_sent.max()
                        for snt, val in rt_by_sent.items():
                            w = val / max(max_r2, 0.01) * 100
                            clr = SENTIMENT_COLORS.get(snt, T3)
                            st.markdown(
                                f'<div class="bar-row">'
                                f'<div class="bar-label" style="color:{clr};">{snt.capitalize()}</div>'
                                f'<div class="bar-track"><div class="bar-fill" style="width:{w:.1f}%;background:{clr};"></div></div>'
                                f'<div class="bar-val">{val:.1f}</div></div>',
                                unsafe_allow_html=True
                            )

            # Stacked sentiment per cluster
            st.markdown('<div class="divider"></div>', unsafe_allow_html=True)
            with st.container(key="card_sent_stack"):
                st.markdown('<div class="card-label">Breakdown</div><div class="card-title">Sentimen per Klaster (Komposisi %)</div>', unsafe_allow_html=True)
                cross = fdf.groupby([C_COL,"sentiment_label"]).size().reset_index(name="Jumlah")
                cross.columns = ["Klaster","Sentimen","Jumlah"]
                pivot = cross.pivot_table(index="Klaster", columns="Sentimen", values="Jumlah", fill_value=0)
                pivot_pct = pivot.div(pivot.sum(axis=1), axis=0) * 100
                pivot_pct = pivot_pct.reset_index().melt(id_vars="Klaster", var_name="Sentimen", value_name="Persen")
                fig_100 = px.bar(pivot_pct, x="Klaster", y="Persen", color="Sentimen", barmode="stack",
                                 color_discrete_map={k:v for k,v in SENTIMENT_COLORS.items()},
                                 text=pivot_pct["Persen"].round(1).astype(str)+"%")
                fig_100.update_traces(textposition="inside", textfont_size=10, insidetextanchor="middle")
                fig_100.update_layout(**{**PLOTLY_LAYOUT, "height":260, "yaxis_title":"Persen (%)", "xaxis_title":""})
                st.plotly_chart(fig_100, use_container_width=True)

            # ──────────────────────────────────────────
            # 3 GRAFIK REPLIKA NOTEBOOK (InSet Lexicon)
            # 1) Bar Chart Distribusi  2) Pie Chart Persentase  3) Rata-rata InSet Score
            # ──────────────────────────────────────────
            st.markdown('<div class="divider"></div>', unsafe_allow_html=True)
            st.markdown(sec("InSet Lexicon","Visualisasi Hasil Labeling Sentimen"), unsafe_allow_html=True)

            # Urutan tetap: Positive → Neutral → Negative (sesuai notebook)
            _ORDER = ["positif", "netral", "negatif"]
            _LABEL_DISP = {"positif": "Positive", "netral": "Neutral", "negatif": "Negative"}
            sc_ord = (fdf["sentiment_label"].value_counts()
                        .reindex(_ORDER).fillna(0).astype(int).reset_index())
            sc_ord.columns = ["Sentimen", "Jumlah"]
            sc_ord["Label"] = sc_ord["Sentimen"].map(_LABEL_DISP)
            sc_ord["Persen"] = (sc_ord["Jumlah"] / max(sc_ord["Jumlah"].sum(), 1) * 100).round(2)

            col_nb1, col_nb2 = st.columns(2)

            # ── (1) Bar Chart — Distribusi Sentimen Bansos ──
            with col_nb1:
                with st.container(key="card_nb_bar"):
                    st.markdown('<div class="card-label">Distribusi</div><div class="card-title">Distribusi Sentimen Bansos (InSet Lexicon)</div>', unsafe_allow_html=True)
                    fig_nb_bar = go.Figure(go.Bar(
                        x=sc_ord["Label"],
                        y=sc_ord["Jumlah"],
                        text=sc_ord["Jumlah"],
                        textposition="outside",
                        textfont=dict(color=T2, size=12),
                        marker=dict(
                            color=[SENTIMENT_COLORS.get(s, T3) for s in sc_ord["Sentimen"]],
                            cornerradius=4,
                        ),
                        cliponaxis=False,
                    ))
                    fig_nb_bar.update_layout(**{
                        **PLOTLY_LAYOUT,
                        "height": 320,
                        "xaxis_title": "Sentimen",
                        "yaxis_title": "Jumlah Tweet",
                        "showlegend": False,
                    })
                    st.plotly_chart(fig_nb_bar, use_container_width=True)

            # ── (2) Pie Chart — Persentase Sentimen Bansos ──
            with col_nb2:
                with st.container(key="card_nb_pie"):
                    st.markdown('<div class="card-label">Persentase</div><div class="card-title">Persentase Sentimen Bansos</div>', unsafe_allow_html=True)
                    fig_nb_pie = go.Figure(go.Pie(
                        labels=[f"{l}<br>({p:.2f}%)" for l, p in zip(sc_ord["Label"], sc_ord["Persen"])],
                        values=sc_ord["Jumlah"],
                        marker=dict(
                            colors=[SENTIMENT_COLORS.get(s, T3) for s in sc_ord["Sentimen"]],
                            line=dict(color=INK, width=2.5),
                        ),
                        textinfo="percent",
                        textfont=dict(size=12, color=T1),
                        pull=[0.04, 0.04, 0.04],
                        hovertemplate="<b>%{label}</b><br>Jumlah: %{value}<br>%{percent}<extra></extra>",
                        sort=False,
                        direction="clockwise",
                    ))
                    fig_nb_pie.update_layout(**{
                        **PLOTLY_LAYOUT,
                        "height": 320,
                        "showlegend": True,
                        "legend": dict(
                            orientation="v", yanchor="middle", y=0.5,
                            xanchor="left", x=1.02,
                            bgcolor="rgba(0,0,0,0)", font=dict(color=T2, size=11),
                        ),
                    })
                    st.plotly_chart(fig_nb_pie, use_container_width=True)

            # ── (3) Rata-rata InSet Score per Sentimen ──
            with st.container(key="card_nb_avg"):
                st.markdown('<div class="card-label">Skor</div><div class="card-title">Rata-rata InSet Score per Sentimen</div>', unsafe_allow_html=True)
                if "sentiment_score" in fdf.columns and fdf["sentiment_score"].notna().any():
                    avg_score = (fdf.groupby("sentiment_label")["sentiment_score"]
                                    .mean().reindex(_ORDER).fillna(0))
                    avg_df = pd.DataFrame({
                        "Label":   [_LABEL_DISP[s] for s in avg_score.index],
                        "Score":   avg_score.values,
                        "Sentimen": list(avg_score.index),
                    })
                    fig_nb_avg = go.Figure(go.Bar(
                        x=avg_df["Label"],
                        y=avg_df["Score"],
                        text=[f"{v:.2f}" for v in avg_df["Score"]],
                        textposition="outside",
                        textfont=dict(color=T2, size=12),
                        marker=dict(
                            color=[SENTIMENT_COLORS.get(s, T3) for s in avg_df["Sentimen"]],
                            cornerradius=4,
                        ),
                        cliponaxis=False,
                    ))
                    # Garis horizontal y=0 (referensi netral) — style notebook
                    fig_nb_avg.add_hline(y=0, line=dict(color=T2, width=1.2, dash="dash"))
                    fig_nb_avg.update_layout(**{
                        **PLOTLY_LAYOUT,
                        "height": 320,
                        "xaxis_title": "",
                        "yaxis_title": "Average InSet Score",
                        "showlegend": False,
                    })
                    st.plotly_chart(fig_nb_avg, use_container_width=True)
                else:
                    st.markdown(ins(
                        "Kolom <code>sentiment_score</code> belum tersedia di data, "
                        "jadi rata-rata InSet score tidak bisa dihitung. Pastikan notebook meng-export "
                        "kolom <code>sentiment_score</code> ke <code>klaster/hasil_klasterisasi_bansos.csv</code>.",
                        "muted"
                    ), unsafe_allow_html=True)

            # ── Tabel Ringkasan Sentimen (replika notebook: sentimen_summary) ──
            st.markdown('<div class="divider"></div>', unsafe_allow_html=True)
            st.markdown(sec("Ringkasan","Tabel Ringkasan Sentimen InSet Lexicon"), unsafe_allow_html=True)
            with st.container(key="card_sent_summary"):
                st.markdown('<div class="card-label">Ringkasan</div><div class="card-title">Sentimen — Jumlah & Persentase</div>', unsafe_allow_html=True)
                _ORDER_SUM = ["positif", "netral", "negatif"]
                _LABEL_SUM = {"positif": "Positif", "netral": "Netral", "negatif": "Negatif"}
                _sum_counts = (fdf["sentiment_label"].value_counts()
                                 .reindex(_ORDER_SUM).fillna(0).astype(int))
                _sum_total  = _sum_counts.sum()
                _sum_df = pd.DataFrame({
                    "Sentimen":    [_LABEL_SUM.get(s, s.capitalize()) for s in _ORDER_SUM],
                    "Jumlah":      _sum_counts.values,
                    "Persentase":  [f"{v/_sum_total*100:.2f}%" if _sum_total > 0 else "0%" for v in _sum_counts.values],
                })
                # HTML tabel custom sesuai .tbl style dashboard
                _tbl_rows = ""
                for _, _r in _sum_df.iterrows():
                    _snt_key = _r["Sentimen"].lower()
                    _clr = SENTIMENT_COLORS.get(_snt_key, T2)
                    _tbl_rows += (
                        f'<tr><td style="color:{_clr};font-weight:600;">{_r["Sentimen"]}</td>'
                        f'<td style="color:{T1};font-variant-numeric:tabular-nums;">{int(_r["Jumlah"]):,}</td>'
                        f'<td style="color:{T2};">{_r["Persentase"]}</td></tr>'
                    )
                st.markdown(f"""
                <table class="tbl">
                  <thead><tr><th>Sentimen</th><th>Jumlah Tweet</th><th>Persentase</th></tr></thead>
                  <tbody>{_tbl_rows}
                    <tr style="border-top:1px solid {LINE2};">
                      <td style="color:{T2};font-weight:600;">Total</td>
                      <td style="color:{T1};font-weight:600;">{int(_sum_total):,}</td>
                      <td style="color:{T2};">100.00%</td>
                    </tr>
                  </tbody>
                </table>""", unsafe_allow_html=True)

            # Sample tweets per sentiment
            st.markdown('<div class="divider"></div>', unsafe_allow_html=True)
            st.markdown(sec("Contoh Tweet","Tweet Representatif per Sentimen"), unsafe_allow_html=True)
            for snt_val in sc["Sentimen"].unique():
                if "pos" in snt_val:   badge = '<span class="bdg bdg-pos">POSITIF</span>'
                elif "neg" in snt_val: badge = '<span class="bdg bdg-neg">NEGATIF</span>'
                else:                  badge = '<span class="bdg bdg-neu">NETRAL</span>'
                sample_sent = fdf[fdf["sentiment_label"] == snt_val]
                with st.expander(f"Contoh — {snt_val.capitalize()} ({len(sample_sent):,} tweet)", expanded=False):
                    top5 = sample_sent.nlargest(5,"retweet_count") if has_rt else sample_sent.head(5)
                    for _, row in top5.iterrows():
                        rt_  = f'<span>{svg("retweet",10,T3)} {int(row["retweet_count"])}</span>' if has_rt else ""
                        fav_ = f'<span>{svg("heart",10,T3)} {int(row["favorite_count"])}</span>' if has_fav else ""
                        st.markdown(f'<div class="tweet fire">{badge}&nbsp;{row[T_COL]}<div class="tweet-meta">{rt_}{fav_}</div></div>', unsafe_allow_html=True)

    # ══════════════════════════════════════════════
    # PANEL 2 — TREN & WAKTU
    # ══════════════════════════════════════════════
    elif panel == "trend":
        if not fdf["tanggal"].notna().any():
            st.info("Data tanggal tidak tersedia untuk analisis tren.")
        else:
            daily = (fdf.dropna(subset=["tanggal"])
                        .groupby(fdf["tanggal"].dt.date)
                        .size().reset_index(name="Jumlah Tweet"))
            daily.columns = ["Tanggal","Jumlah Tweet"]
            peak_row = daily.loc[daily["Jumlah Tweet"].idxmax()]
            st.markdown(ins(
                f'Volume tertinggi terjadi pada <strong>{peak_row["Tanggal"]}</strong> dengan '
                f'<strong>{int(peak_row["Jumlah Tweet"]):,} tweet</strong> dalam satu hari.'
            ), unsafe_allow_html=True)

            with st.container(key="card_trend_vol"):
                st.markdown('<div class="card-label">Volume Harian</div><div class="card-title">Tren Percakapan Bansos dari Waktu ke Waktu</div>', unsafe_allow_html=True)
                fig_vol = px.area(daily, x="Tanggal", y="Jumlah Tweet", markers=True, color_discrete_sequence=[FIRE2])
                fig_vol.update_traces(line_width=2, marker_size=4, fillcolor="rgba(255,122,48,0.04)")
                fig_vol.update_layout(**{**PLOTLY_LAYOUT, "height":200, "xaxis_title":"Tanggal", "yaxis_title":"Jumlah Tweet"})
                st.plotly_chart(fig_vol, use_container_width=True)

            col_cl, col_eng = st.columns(2)
            with col_cl:
                with st.container(key="card_trend_cluster"):
                    st.markdown('<div class="card-label">Klaster × Waktu</div><div class="card-title">Volume per Klaster per Hari</div>', unsafe_allow_html=True)
                    cl_daily = (fdf.dropna(subset=["tanggal"])
                                   .groupby([fdf["tanggal"].dt.date, C_COL])
                                   .size().reset_index(name="Jumlah Tweet"))
                    cl_daily.columns = ["Tanggal","Klaster","Jumlah Tweet"]
                    fig_cld = px.bar(cl_daily, x="Tanggal", y="Jumlah Tweet", color="Klaster",
                                     color_discrete_sequence=CLUSTER_PALETTE)
                    fig_cld.update_layout(**{**PLOTLY_LAYOUT, "height":195, "barmode":"stack", "xaxis_title":"Tanggal", "yaxis_title":"Tweet"})
                    st.plotly_chart(fig_cld, use_container_width=True)

            with col_eng:
                if has_rt or has_fav:
                    with st.container(key="card_trend_eng"):
                        st.markdown('<div class="card-label">Engagement Harian</div><div class="card-title">Retweet & Like per Hari</div>', unsafe_allow_html=True)
                        eng_daily = fdf.dropna(subset=["tanggal"]).groupby(fdf["tanggal"].dt.date).agg({
                            **({} if not has_rt  else {"retweet_count": "sum"}),
                            **({} if not has_fav else {"favorite_count": "sum"}),
                        }).reset_index()
                        eng_daily = eng_daily.rename(columns={eng_daily.columns[0]: "Tanggal"})
                        traces = []
                        if has_rt:
                            traces.append(go.Scatter(x=eng_daily["Tanggal"], y=eng_daily["retweet_count"],
                                                     name="Retweet", line=dict(color="#666666", width=1.6), mode="lines+markers", marker=dict(size=3)))
                        if has_fav:
                            traces.append(go.Scatter(x=eng_daily["Tanggal"], y=eng_daily["favorite_count"],
                                                     name="Like", line=dict(color=FIRE, width=1.6), mode="lines+markers", marker=dict(size=3)))
                        fig_engt = go.Figure(traces)
                        fig_engt.update_layout(**{**PLOTLY_LAYOUT, "height":195, "xaxis_title":"Tanggal", "yaxis_title":"Jumlah"})
                        st.plotly_chart(fig_engt, use_container_width=True)

            fdf_t = fdf.dropna(subset=["tanggal"]).copy()
            fdf_t["hari"] = fdf_t["tanggal"].dt.day_name()
            # Jam: gunakan created_at (punya timestamp lengkap) bukan tanggal (date only)
            if "created_at" in fdf_t.columns:
                _dt_full = pd.to_datetime(fdf_t["created_at"], format="%a %b %d %H:%M:%S %z %Y", errors="coerce")
                if _dt_full.isna().all():
                    _dt_full = pd.to_datetime(fdf_t["created_at"], errors="coerce")
                fdf_t["jam"] = _dt_full.dt.hour
            else:
                fdf_t["jam"] = fdf_t["tanggal"].dt.hour
            DAY_ORDER = ["Monday","Tuesday","Wednesday","Thursday","Friday","Saturday","Sunday"]
            DAY_ID    = ["Senin","Selasa","Rabu","Kamis","Jumat","Sabtu","Minggu"]
            hari_count = fdf_t["hari"].value_counts().reindex(DAY_ORDER).fillna(0).reset_index()
            hari_count.columns = ["Hari","Jumlah"]
            hari_count["Hari_ID"] = DAY_ID
            jam_count = fdf_t["jam"].value_counts().sort_index().reset_index()
            jam_count.columns = ["Jam","Jumlah"]

            col_d, col_j = st.columns(2)
            with col_d:
                with st.container(key="card_trend_weekly"):
                    st.markdown('<div class="card-label">Pola Mingguan</div><div class="card-title">Tweet per Hari dalam Seminggu</div>', unsafe_allow_html=True)
                    max_h = hari_count["Jumlah"].max()
                    bars_day = ""
                    for _, row in hari_count.iterrows():
                        w = row["Jumlah"] / max(max_h, 1) * 100
                        clr = FIRE if w == 100 else (FIRE2 if w > 80 else ("#555555" if w > 40 else "#2d2d2d"))
                        bars_day += f'<div class="bar-row"><div class="bar-label">{row["Hari_ID"]}</div><div class="bar-track"><div class="bar-fill" style="width:{w:.1f}%;background:{clr};"></div></div><div class="bar-val">{int(row["Jumlah"])}</div></div>'
                    st.markdown(bars_day, unsafe_allow_html=True)

            with col_j:
                with st.container(key="card_trend_hourly"):
                    st.markdown('<div class="card-label">Pola Harian</div><div class="card-title">Tweet per Jam (WIB)</div>', unsafe_allow_html=True)
                    fig_jam = px.bar(jam_count, x="Jam", y="Jumlah",
                                     color="Jumlah", color_continuous_scale=[INK4, FIRE])
                    fig_jam.update_layout(**{**PLOTLY_LAYOUT, "coloraxis_showscale":False,
                                             "height":195, "xaxis_title":"Jam", "yaxis_title":"Tweet",
                                             "xaxis":dict(tickmode="linear", dtick=2, showgrid=True, gridcolor=LINE)})
                    st.plotly_chart(fig_jam, use_container_width=True)

            # ── Trending Keywords Top 15 (replika notebook cell "trending keywords keseluruhan bansos") ──
            st.markdown('<div class="divider"></div>', unsafe_allow_html=True)
            st.markdown(sec("Trending Keywords","Top 15 Trending Keywords — Diskusi Bansos"), unsafe_allow_html=True)
            st.markdown(ins(
                "Frekuensi kemunculan kata yang paling sering muncul di seluruh tweet setelah preprocessing. "
                "Kata-kata ini mencerminkan topik yang paling banyak diperbincangkan publik terkait bansos."
            ), unsafe_allow_html=True)
            _trending_df = top_words(fdf[T_COL], n=15)
            if not _trending_df.empty:
                with st.container(key="card_trend_keywords"):
                    st.markdown('<div class="card-label">Kata Kunci</div><div class="card-title">Top 15 Trending Keywords — Frekuensi Kemunculan</div>', unsafe_allow_html=True)
                    _trend_sorted = _trending_df.sort_values("Frekuensi")
                    _bg_trend = []
                    for i, _ in enumerate(_trend_sorted.itertuples()):
                        rank_from_top = len(_trend_sorted) - 1 - i
                        if rank_from_top == 0:  _bg_trend.append(FIRE)
                        elif rank_from_top < 3: _bg_trend.append("rgba(245,90,0,0.55)")
                        elif rank_from_top < 6: _bg_trend.append("rgba(245,90,0,0.30)")
                        else:                   _bg_trend.append("rgba(102,102,102,0.45)")
                    fig_trending = go.Figure(go.Bar(
                        x=_trend_sorted["Frekuensi"],
                        y=_trend_sorted["Kata"],
                        orientation="h",
                        text=_trend_sorted["Frekuensi"],
                        textposition="outside",
                        textfont=dict(color=T3, size=10),
                        marker=dict(color=_bg_trend, cornerradius=3),
                        cliponaxis=False,
                    ))
                    fig_trending.update_layout(**{**PLOTLY_LAYOUT, "height": 480,
                                                  "xaxis_title": "Frekuensi Kemunculan", "yaxis_title": "",
                                                  "showlegend": False})
                    st.plotly_chart(fig_trending, use_container_width=True)
                    # Tabel ranking
                    _trending_rank = _trending_df.assign(Rank=range(1, len(_trending_df)+1))[["Rank","Kata","Frekuensi"]]
                    st.dataframe(_trending_rank, use_container_width=True, hide_index=True)

            # ── Tabel Pivot Volume per Klaster per Hari (replika notebook: pivot_klaster) ──
            st.markdown('<div class="divider"></div>', unsafe_allow_html=True)
            st.markdown(sec("Volume per Klaster","Tabel Pivot — Jumlah Tweet per Klaster per Hari"), unsafe_allow_html=True)
            st.markdown(ins(
                "Tabel ini menampilkan jumlah tweet per klaster untuk setiap tanggal. "
                "Baris = tanggal, Kolom = klaster, Nilai = jumlah tweet."
            ), unsafe_allow_html=True)
            _fdf_piv = fdf.dropna(subset=["tanggal"]).copy()
            if not _fdf_piv.empty:
                with st.container(key="card_trend_pivot"):
                    st.markdown('<div class="card-label">Pivot</div><div class="card-title">Jumlah Tweet per Klaster per Hari</div>', unsafe_allow_html=True)
                    _vol_per_klaster = (
                        _fdf_piv.groupby([_fdf_piv["tanggal"].dt.date, C_COL])
                        .size().reset_index(name="Jumlah Tweet")
                    )
                    _vol_per_klaster.columns = ["Tanggal", "Klaster", "Jumlah Tweet"]
                    _pivot_klaster = (
                        _vol_per_klaster.pivot_table(
                            index="Tanggal", columns="Klaster",
                            values="Jumlah Tweet", fill_value=0
                        )
                    )
                    _pivot_klaster.columns.name = None
                    _pivot_klaster = _pivot_klaster.reset_index()
                    _pivot_klaster["Total"] = _pivot_klaster.iloc[:, 1:].sum(axis=1).astype(int)
                    # Format Tanggal sebagai string
                    _pivot_klaster["Tanggal"] = _pivot_klaster["Tanggal"].astype(str)
                    st.dataframe(_pivot_klaster, use_container_width=True, hide_index=True, height=300)

    # ══════════════════════════════════════════════
    # PANEL 3 — SOCIAL NETWORK ANALYSIS
    # ══════════════════════════════════════════════
    elif panel == "sna":
        st.markdown(ins(
            f'Tab ini menampilkan analisis jaringan interaksi reply antar pengguna Twitter/X. '
            f'Data dari kolom <code>in_reply_to_screen_name</code>. Graf mencakup 80 node teratas by degree centrality.'
        ), unsafe_allow_html=True)

        has_sna = ("in_reply_to_screen_name" in fdf.columns and
                   ("user_id_str" in fdf.columns or "username" in fdf.columns))

        if has_sna:
            try:
                import networkx as nx
                sna_df = fdf[fdf["in_reply_to_screen_name"].notnull()].copy()
                src_col_sna = "username" if ("username" in sna_df.columns and sna_df["username"].notna().any()) else "user_id_str"
                sna_df["source"] = sna_df[src_col_sna].astype(str)
                sna_df["target"] = sna_df["in_reply_to_screen_name"].astype(str)
                sna_df = sna_df[sna_df["source"] != sna_df["target"]]
                G = nx.from_pandas_edgelist(sna_df, source="source", target="target", create_using=nx.DiGraph())

                n_nodes = G.number_of_nodes()
                n_edges = G.number_of_edges()
                density = nx.density(G)
                G_un    = G.to_undirected()
                comps   = nx.number_connected_components(G_un)

                st.markdown(f"""
                <div class="g4">
                  <div class="met"><div class="met-v" style="color:{T1};">{n_nodes:,}</div><div class="met-l">Node / Aktor</div></div>
                  <div class="met"><div class="met-v" style="color:{FIRE};">{n_edges:,}</div><div class="met-l">Edge / Interaksi</div></div>
                  <div class="met"><div class="met-v" style="color:{T2};">{density:.4f}</div><div class="met-l">Density Graf</div></div>
                  <div class="met"><div class="met-v" style="color:{T2};">{comps}</div><div class="met-l">Komponen</div></div>
                </div>
                """, unsafe_allow_html=True)

                deg_c  = nx.degree_centrality(G)
                btwn_c = nx.betweenness_centrality(G)
                clos_c = nx.closeness_centrality(G)
                try:
                    eig_c = nx.eigenvector_centrality(G, max_iter=1000)
                except Exception:
                    try: eig_c = nx.eigenvector_centrality_numpy(G)
                    except: eig_c = {n: 0.0 for n in deg_c}

                cdf = pd.DataFrame({
                    "User":        list(deg_c.keys()),
                    "Degree":      list(deg_c.values()),
                    "Betweenness": [btwn_c[n] for n in deg_c],
                    "Closeness":   [clos_c[n] for n in deg_c],
                    "Eigenvector": [eig_c.get(n, 0.0) for n in deg_c],
                }).sort_values("Betweenness", ascending=False).head(10).reset_index(drop=True)

                col_bar_sna, col_tbl_sna = st.columns([3,2])
                with col_bar_sna:
                    with st.container(key="card_sna_centrality"):
                        st.markdown(sec("Sentralitas","Top 10 Aktor — Betweenness Centrality"), unsafe_allow_html=True)
                        fig_cent = px.bar(cdf, x="Betweenness", y="User", orientation="h",
                                          text=cdf["Betweenness"].round(4),
                                          color="Betweenness", color_continuous_scale=[INK4, FIRE])
                        fig_cent.update_traces(textposition="outside", textfont_size=10, textfont_color=T3)
                        fig_cent.update_layout(**{**PLOTLY_LAYOUT, "coloraxis_showscale":False, "height":340, "yaxis_title":""})
                        st.plotly_chart(fig_cent, use_container_width=True)

                with col_tbl_sna:
                    with st.container(key="card_sna_indegree"):
                        st.markdown(sec("In-Degree","User Paling Banyak Di-reply"), unsafe_allow_html=True)
                        in_deg = pd.Series(dict(G.in_degree())).sort_values(ascending=False).head(10)
                        max_id = in_deg.max()
                        for user, val in in_deg.items():
                            w = val / max(max_id,1) * 100
                            clr = FIRE if w == 100 else ("#555555" if w > 50 else "#333333")
                            st.markdown(
                                f'<div class="bar-row" style="margin-bottom:5px;">'
                                f'<div class="bar-label" style="color:{clr if w==100 else T2};width:90px;">{str(user)[:14]}</div>'
                                f'<div class="bar-track"><div class="bar-fill" style="width:{w:.1f}%;background:{clr};"></div></div>'
                                f'<div class="bar-val">{val}</div></div>',
                                unsafe_allow_html=True
                            )
                        st.markdown(f'<div style="margin-top:14px;font-size:9px;font-weight:600;color:{T3};text-transform:uppercase;letter-spacing:1.2px;margin-bottom:7px;">Konsep SNA</div>', unsafe_allow_html=True)
                        st.markdown(f"""
                        <div style="font-size:11px;color:{T3};line-height:2;">
                          <span style="color:{FIRE};font-weight:600;">Degree</span> — Jumlah koneksi langsung<br>
                          <span style="color:{T2};font-weight:600;">Betweenness</span> — Perantara jalur terpendek<br>
                          <span style="color:{T2};font-weight:600;">Closeness</span> — Kedekatan ke semua node<br>
                          <span style="color:{T2};font-weight:600;">Eigenvector</span> — Pengaruh dari kualitas koneksi
                        </div>
                        """, unsafe_allow_html=True)

                # ── Top 10 Degree Centrality (replika notebook: top_centrality[:10]) ──
                st.markdown('<div class="divider"></div>', unsafe_allow_html=True)
                with st.container(key="card_sna_degree"):
                    st.markdown(sec("Degree Centrality","Top 10 User — Degree Centrality"), unsafe_allow_html=True)
                    _deg_top10 = sorted(deg_c.items(), key=lambda x: x[1], reverse=True)[:10]
                    _max_deg = _deg_top10[0][1] if _deg_top10 else 1
                    for _di, (_du, _dv) in enumerate(_deg_top10):
                        _dw = _dv / max(_max_deg, 0.0001) * 100
                        _dclr = FIRE if _di == 0 else (FIRE2 if _di < 3 else ("#555555" if _dw > 40 else "#333333"))
                        st.markdown(
                            f'<div class="bar-row" style="margin-bottom:5px;">'
                            f'<div style="font-size:9px;color:{T3};min-width:18px;text-align:right;flex-shrink:0;">{_di+1}.</div>'
                            f'<div class="bar-label" style="color:{_dclr};width:100px;">{str(_du)[:16]}</div>'
                            f'<div class="bar-track"><div class="bar-fill" style="width:{_dw:.1f}%;background:{_dclr};"></div></div>'
                            f'<div class="bar-val">{_dv:.4f}</div></div>',
                            unsafe_allow_html=True
                        )

                # ── Tabel Lengkap 4 Metrik Sentralitas (replika notebook: centrality_df) ──
                st.markdown('<div class="divider"></div>', unsafe_allow_html=True)
                with st.container(key="card_sna_full_centrality"):
                    st.markdown(sec("Metrik Sentralitas","Tabel Lengkap — Top 10 Aktor Berdasarkan Betweenness Centrality"), unsafe_allow_html=True)
                    _cdf_disp = cdf.copy()
                    for _col in ["Degree","Betweenness","Closeness","Eigenvector"]:
                        if _col in _cdf_disp.columns:
                            _cdf_disp[_col] = _cdf_disp[_col].round(4)
                    _cdf_disp.index = range(1, len(_cdf_disp)+1)
                    _tbl_sna_rows = ""
                    for _idx, _r in _cdf_disp.iterrows():
                        _row_clr = FIRE if _idx == 1 else T2
                        _tbl_sna_rows += (
                            f'<tr>'
                            f'<td style="color:{T3};">{_idx}</td>'
                            f'<td style="color:{_row_clr};font-weight:{"600" if _idx==1 else "400"};">{str(_r["User"])[:20]}</td>'
                            f'<td style="color:{T2};font-variant-numeric:tabular-nums;">{_r["Degree"]:.4f}</td>'
                            f'<td style="color:{FIRE if _idx==1 else T2};font-variant-numeric:tabular-nums;">{_r["Betweenness"]:.4f}</td>'
                            f'<td style="color:{T2};font-variant-numeric:tabular-nums;">{_r["Closeness"]:.4f}</td>'
                            f'<td style="color:{T2};font-variant-numeric:tabular-nums;">{_r["Eigenvector"]:.4f}</td>'
                            f'</tr>'
                        )
                    st.markdown(f"""
                    <table class="tbl">
                      <thead><tr><th>#</th><th>User</th><th>Degree</th><th>Betweenness</th><th>Closeness</th><th>Eigenvector</th></tr></thead>
                      <tbody>{_tbl_sna_rows}</tbody>
                    </table>""", unsafe_allow_html=True)

                # Network graph
                st.markdown('<div class="divider"></div>', unsafe_allow_html=True)
                with st.container(key="card_sna_network"):
                    st.markdown(sec("Visualisasi Jaringan","Graf Interaksi Reply — Top 80 Node"), unsafe_allow_html=True)
                    top_nodes = sorted(G.degree(), key=lambda x: x[1], reverse=True)[:80]
                    top_names = [n for n,_ in top_nodes]
                    subG = G.subgraph(top_names)
                    pos  = nx.spring_layout(subG, k=0.6, iterations=30, seed=42)
                    degs = dict(subG.degree())
                    ex, ey = [], []
                    for u, v in subG.edges():
                        x0,y0 = pos[u]; x1,y1 = pos[v]
                        ex += [x0, x1, None]; ey += [y0, y1, None]
                    fig_net = go.Figure()
                    fig_net.add_trace(go.Scatter(x=ex, y=ey, mode="lines",
                                                  line=dict(width=0.5, color=LINE2), hoverinfo="none"))
                    fig_net.add_trace(go.Scatter(
                        x=[pos[n][0] for n in subG.nodes()],
                        y=[pos[n][1] for n in subG.nodes()],
                        mode="markers",
                        marker=dict(size=[5+degs.get(n,1)*3 for n in subG.nodes()],
                                    color=[deg_c.get(n,0) for n in subG.nodes()],
                                    colorscale=[[0,INK4],[0.5,FIRE2],[1,FIRE]],
                                    showscale=True,
                                    colorbar=dict(title="Degree", thickness=10, tickfont=dict(color=T3))),
                        text=[f"{n}<br>Degree: {degs.get(n,0)}" for n in subG.nodes()],
                        hoverinfo="text"
                    ))
                    fig_net.update_layout(**{**PLOTLY_LAYOUT, "height":380, "showlegend":False,
                                             "xaxis":dict(showgrid=False, zeroline=False, showticklabels=False),
                                             "yaxis":dict(showgrid=False, zeroline=False, showticklabels=False)})
                    st.plotly_chart(fig_net, use_container_width=True)

                # ── Community Detection (Girvan-Newman) — replika notebook section 3.2 ──
                st.markdown('<div class="divider"></div>', unsafe_allow_html=True)
                with st.container(key="card_sna_community"):
                    st.markdown(sec("Deteksi Komunitas","Community Detection — Girvan-Newman"), unsafe_allow_html=True)
                    st.markdown(ins(
                        "Algoritma <strong>Girvan-Newman</strong> mendeteksi komunitas/kubu dalam jaringan diskusi bansos "
                        "berdasarkan pola interaksi reply. Node dengan warna sama berasal dari komunitas yang sama.",
                    ), unsafe_allow_html=True)
                    try:
                        from networkx.algorithms import community as nx_community
                        G_un = G.to_undirected()
                        comm_gen = nx_community.girvan_newman(G_un)
                        top_comms = next(comm_gen)
                        communities_list = sorted(map(sorted, top_comms))
                        n_comms = len(communities_list)

                        # Warna per komunitas
                        COMM_COLORS = [FIRE, "#4A90D9", GREEN, FIRE2, "#9B59B6", "#2ECC71", "#E67E22", "#E74C3C"]
                        node_comm_color = {}
                        for i, comm in enumerate(communities_list):
                            clr = COMM_COLORS[i % len(COMM_COLORS)]
                            for node in comm:
                                node_comm_color[node] = clr

                        # Subgraph top 80 untuk visualisasi
                        top80 = [n for n, _ in sorted(G.degree(), key=lambda x: x[1], reverse=True)[:80]]
                        subGc = G.to_undirected().subgraph(top80)
                        posc = nx.spring_layout(subGc, k=0.6, iterations=30, seed=42)

                        exc, eyc = [], []
                        for u, v in subGc.edges():
                            x0,y0 = posc[u]; x1,y1 = posc[v]
                            exc += [x0, x1, None]; eyc += [y0, y1, None]

                        fig_comm = go.Figure()
                        fig_comm.add_trace(go.Scatter(x=exc, y=eyc, mode="lines",
                                                       line=dict(width=0.5, color=LINE2), hoverinfo="none"))

                        # Plot per komunitas agar legend muncul
                        for i, comm in enumerate(communities_list):
                            comm_nodes = [n for n in comm if n in subGc.nodes()]
                            if not comm_nodes: continue
                            clr = COMM_COLORS[i % len(COMM_COLORS)]
                            degs_c = dict(subGc.degree())
                            fig_comm.add_trace(go.Scatter(
                                x=[posc[n][0] for n in comm_nodes],
                                y=[posc[n][1] for n in comm_nodes],
                                mode="markers",
                                name=f"Komunitas {i+1} ({len(comm)} node)",
                                marker=dict(
                                    size=[5 + degs_c.get(n, 1) * 3 for n in comm_nodes],
                                    color=clr,
                                    line=dict(color=INK, width=0.8),
                                ),
                                text=[f"{n}<br>Komunitas {i+1}" for n in comm_nodes],
                                hoverinfo="text",
                            ))

                        fig_comm.update_layout(**{**PLOTLY_LAYOUT, "height": 380,
                                                   "showlegend": True,
                                                   "legend": dict(bgcolor="rgba(0,0,0,0)", font=dict(color=T2, size=10),
                                                                  orientation="v", yanchor="top", y=1, xanchor="left", x=1.01),
                                                   "xaxis": dict(showgrid=False, zeroline=False, showticklabels=False),
                                                   "yaxis": dict(showgrid=False, zeroline=False, showticklabels=False)})
                        st.plotly_chart(fig_comm, use_container_width=True)
                        st.markdown(ins(
                            f'Jaringan diskusi bansos terbagi menjadi <strong>{n_comms} komunitas</strong> '
                            f'berdasarkan pola interaksi reply. Setiap komunitas mencerminkan kelompok pengguna '
                            f'yang lebih banyak berinteraksi satu sama lain.'
                        ), unsafe_allow_html=True)
                    except Exception as e_comm:
                        st.markdown(ins(f"Community detection tidak dapat dijalankan: <code>{e_comm}</code>", "muted"), unsafe_allow_html=True)

            except ImportError:
                st.markdown(ins("Library <code>networkx</code> belum terinstall. Jalankan <code>pip install networkx</code>.", "red"), unsafe_allow_html=True)
            except Exception as e:
                st.markdown(ins(f"Error saat memproses SNA: <code>{e}</code>", "red"), unsafe_allow_html=True)
        else:
            st.markdown(ins("Kolom <code>in_reply_to_screen_name</code> tidak ditemukan. Di bawah ini penjelasan konsep SNA.", "muted"), unsafe_allow_html=True)
            concepts = [
                ("Degree Centrality",     FIRE,  "Jumlah koneksi langsung sebuah node. Tinggi = user aktif berinteraksi."),
                ("Betweenness Centrality",FIRE2, "Seberapa sering node menjadi perantara jalur terpendek. Tinggi = broker informasi."),
                ("Closeness Centrality",  T2,    "Seberapa dekat node ke semua node lain. Tinggi = informasi cepat menyebar."),
                ("Eigenvector Centrality",T3,    "Pengaruh berdasarkan kualitas koneksi (terhubung ke node berpengaruh = lebih berpengaruh)."),
            ]
            for name, color, desc in concepts:
                st.markdown(f"""
                <div class="pipe" style="border-left:2px solid {color};">
                  <div style="flex:1;">
                    <div class="pipe-ttl" style="color:{color};">{name}</div>
                    <div class="pipe-desc" style="margin:0;">{desc}</div>
                  </div>
                </div>""", unsafe_allow_html=True)

    # ══════════════════════════════════════════════
    # PANEL 4 — KATA KUNCI
    # ══════════════════════════════════════════════
    elif panel == "keywords":
        st.markdown(ins(
            "Kata kunci dihitung dari kolom teks yang sudah dibersihkan. "
            "Frekuensi tinggi mencerminkan topik paling banyak dibahas publik."
        ), unsafe_allow_html=True)

        words_df = top_words(fdf[T_COL], n=20)

        # ── Keyword search box — pure Streamlit, styled to match HTML ──
        st.markdown(f"""
        <div class="kw-search-wrap">
          <div class="kw-search-label">Cari Kata Kunci <span>·</span> pilih untuk menampilkan grafik</div>
        </div>
        """, unsafe_allow_html=True)

        # Search row: input + button side by side
        srch_col1, srch_col2 = st.columns([5, 1])
        with srch_col1:
            kw_input = st.text_input(
                "",
                placeholder="Ketik kata kunci… mis. bansos, korupsi, pkh",
                key="kw_explorer",
                label_visibility="collapsed"
            )
        with srch_col2:
            kw_search_clicked = st.button("🔍 Cari", key="kw_search_btn", use_container_width=True)

        # Suggestion chips — show top words as quick-select buttons
        if not words_df.empty:
            chip_cols = st.columns(min(10, len(words_df)))
            for i, row in words_df.head(10).iterrows():
                with chip_cols[i % 10]:
                    if st.button(row["Kata"], key=f"chip_{i}", use_container_width=True):
                        st.session_state["kw_explorer"] = row["Kata"]
                        st.rerun()

        if words_df.empty:
            st.info("Tidak ada kata yang bisa dianalisis.")
        else:
            # Resolve selected keyword
            selected_kw = kw_input.strip().lower() if kw_input else None
            if selected_kw and selected_kw not in words_df["Kata"].values:
                partial_matches = words_df[words_df["Kata"].str.contains(selected_kw, case=False)]
                if not partial_matches.empty:
                    selected_kw = partial_matches.iloc[0]["Kata"]
                else:
                    selected_kw = None

            col_wbar, col_wtab = st.columns([3,1])
            with col_wbar:
                if selected_kw:
                    st.markdown(sec("Analisis Teks", f'Frekuensi untuk "<span style="color:{FIRE};">{selected_kw}</span>" dibanding 20 kata teratas'), unsafe_allow_html=True)
                else:
                    st.markdown(sec("Analisis Teks","20 Kata Paling Sering Muncul dalam Tweet"), unsafe_allow_html=True)

                plot_df = words_df.sort_values("Frekuensi").copy()
                bg = []
                for i, w in enumerate(plot_df["Kata"]):
                    rank = words_df[words_df["Kata"]==w].index[0]
                    if selected_kw and w == selected_kw: bg.append(FIRE)
                    elif rank == 0: bg.append(FIRE)
                    elif rank < 3:  bg.append("rgba(245,90,0,0.4)")
                    elif rank < 7:  bg.append("rgba(245,90,0,0.2)")
                    else:           bg.append("rgba(46,46,46,0.6)")
                fig_kw_bar = go.Figure(go.Bar(
                    x=plot_df["Frekuensi"], y=plot_df["Kata"], orientation="h",
                    text=plot_df["Frekuensi"], textposition="outside",
                    textfont=dict(color=T3, size=10),
                    marker=dict(color=bg, cornerradius=3), cliponaxis=False,
                ))
                fig_kw_bar.update_layout(**{**PLOTLY_LAYOUT, "height":500, "yaxis_title":"", "xaxis_title":"Frekuensi"})
                st.plotly_chart(fig_kw_bar, use_container_width=True)

            with col_wtab:
                st.markdown(sec("Ranking","Top 20 Kata"), unsafe_allow_html=True)
                rdf = words_df.assign(Rank=range(1, len(words_df)+1))[["Rank","Kata","Frekuensi"]]
                st.dataframe(rdf, use_container_width=True, hide_index=True, height=500)

        # Per klaster
        st.markdown('<div class="divider"></div>', unsafe_allow_html=True)
        st.markdown(sec("Per Klaster","Top 8 Kata per Klaster"), unsafe_allow_html=True)
        uniq_cls  = sorted(fdf[C_COL].dropna().astype(str).unique())
        n_c       = min(len(uniq_cls), 3)
        grid_cols = st.columns(n_c)
        for i, cl in enumerate(uniq_cls):
            with grid_cols[i % n_c]:
                sub = fdf[fdf[C_COL].astype(str) == cl]
                w   = top_words(sub[T_COL], n=8)
                if w.empty: continue
                clr = CLUSTER_PALETTE[i % len(CLUSTER_PALETTE)]
                with st.container(key=f"card_kw_{i}"):
                    st.markdown(f'<div style="font-size:10.5px;font-weight:700;color:{clr};margin-bottom:8px;display:flex;align-items:center;gap:5px;"><div style="width:5px;height:5px;border-radius:1px;background:{clr};transform:rotate(12deg);"></div>Klaster {cl}</div>', unsafe_allow_html=True)
                    max_wf = w["Frekuensi"].max()
                    for _, row in w.iterrows():
                        ww = row["Frekuensi"] / max_wf * 100
                        st.markdown(
                            f'<div class="bar-row"><div class="bar-label" style="font-size:10px;width:80px;">{row["Kata"]}</div>'
                            f'<div class="bar-track"><div class="bar-fill" style="width:{ww:.1f}%;background:{clr};"></div></div>'
                            f'<div class="bar-val">{row["Frekuensi"]}</div></div>',
                            unsafe_allow_html=True
                        )

        # Wordcloud
        wc_path = "klaster/wordcloud_per_klaster.png"
        if os.path.exists(wc_path):
            st.markdown('<div class="divider"></div>', unsafe_allow_html=True)
            st.markdown(sec("Wordcloud","Visualisasi Kata per Klaster"), unsafe_allow_html=True)
            st.image(wc_path, use_container_width=True)
        else:
            st.markdown(f"""
            <div class="card" style="margin-top:10px;">
              <div class="wc-ph">
                {svg("tag",24,T3)}<br><br>
                <strong>Wordcloud belum tersedia</strong><br>
                Jalankan sel wordcloud di notebook, lalu simpan ke <code>klaster/wordcloud_per_klaster.png</code>
              </div>
            </div>""", unsafe_allow_html=True)

    # ══════════════════════════════════════════════
    # PANEL 5 — METODOLOGI / PIPELINE
    # ══════════════════════════════════════════════
    elif panel == "pipeline":
        sil_v = cm.get("silhouette"); dbi_v = cm.get("davies_bouldin")
        chi_v = cm.get("calinski_harabasz"); k_v = cm.get("optimal_k")

        st.markdown(ins(
            f'Dashboard dibangun melalui <strong>5 tahap analisis</strong> terhadap '
            f'<strong>{len(df_raw):,} tweet</strong> publik bertopik bansos di Twitter/X.'
        ), unsafe_allow_html=True)

        st.markdown(sec("Metodologi","Tahapan Analisis — Bansos Insight"), unsafe_allow_html=True)

        PIPELINE = [
            ("01", FIRE,      "tweet",   "Crawling & Pengumpulan Data",
             "Tweet yang membahas topik bansos dikumpulkan dari Twitter/X, mencakup teks, tanggal, dan metrik engagement (retweet, like, reply, quote).",
             f"Total <strong>{len(df_raw):,} tweet</strong> mentah · Kolom: <em>full_text, created_at, retweet_count, favorite_count</em>"),
            ("02", FIRE2,     "filter",  "Text Preprocessing",
             "Teks dibersihkan dari URL, mention (@user), hashtag, tanda baca, angka. Stopwords dihapus agar model fokus pada kata bermakna.",
             "Output: kolom <em>clean_text</em> · Total stopwords: <strong>120+ kata</strong> (NLTK Indonesia + custom)"),
            ("03", "#C9622E", "smile",   "Analisis Sentimen — InSet Lexicon",
             "Setiap tweet diberi skor sentimen menggunakan leksikon InSet (kata positif & negatif berbobot bahasa Indonesia). Diklasifikasikan menjadi Positif, Negatif, atau Netral.",
             "Leksikon tanpa data latih · Output: <em>sentiment_score, sentiment_label</em>"),
            ("04", "#9A9A9A", "cluster", "Klasterisasi Topik — TF-IDF + SVD + K-Means",
             "Teks bersih diubah ke TF-IDF (max 1000 fitur, min_df=2), dimensi dikurangi SVD (n=2), lalu K-Means mengelompokkan tweet. Jumlah optimal dipilih via Elbow + Silhouette.",
             f"TF-IDF shape: <strong>{len(df_raw)} × 1000</strong> · K optimal = <strong>{k_v}</strong> · Output: <em>cluster, cluster_label</em>"),
            ("05", "#7A7A7A", "network", "Social Network Analysis & Dashboard",
             "Interaksi reply antar pengguna dipetakan sebagai graf terarah via NetworkX untuk melihat aktor paling berpengaruh (degree, betweenness, closeness centrality).",
             "Kolom SNA: <em>in_reply_to_screen_name</em> · Stack: Streamlit + Plotly via Cloudflare Tunnel"),
        ]

        for no, color, icon, title, desc, detail in PIPELINE:
            st.markdown(f"""
            <div class="pipe" style="border-left:2px solid {color};">
              <div class="pipe-num" style="color:{color};">{no}</div>
              <div style="flex:1;">
                <div class="pipe-ttl" style="display:flex;align-items:center;gap:8px;">
                  {svg(icon,14,color)} {title}
                </div>
                <div class="pipe-desc">{desc}</div>
                <div class="pipe-code">→ {detail}</div>
              </div>
            </div>
            """, unsafe_allow_html=True)

        st.markdown('<div class="divider"></div>', unsafe_allow_html=True)
        st.markdown(sec("Evaluasi Klaster","Kualitas Klasterisasi K-Means"), unsafe_allow_html=True)
        st.markdown(ins(
            f'Jumlah klaster optimal dipilih <strong>K = {k_v}</strong> berdasarkan Elbow Method & Silhouette Score. '
            f'Skor evaluasi menunjukkan pemisahan antar-klaster yang '
            f'{"cukup baik" if (sil_v or 0) >= 0.5 else "wajar"} untuk data teks pendek seperti tweet.'
        ), unsafe_allow_html=True)

        st.markdown(f"""
        <div class="g4">
          <div class="met"><div class="met-v" style="color:{FIRE};">K = {k_v}</div><div class="met-l">Klaster Optimal</div><div class="met-cap">Elbow + Silhouette</div></div>
          <div class="met"><div class="met-v" style="color:{FIRE2};">{"%.4f"%sil_v if sil_v else "—"}</div><div class="met-l">Silhouette Score</div><div class="met-cap">baik jika &gt; 0.5</div></div>
          <div class="met"><div class="met-v" style="color:{T2};">{"%.4f"%dbi_v if dbi_v else "—"}</div><div class="met-l">Davies-Bouldin</div><div class="met-cap">baik jika &lt; 1.0</div></div>
          <div class="met"><div class="met-v" style="color:{T1};">{"%.0f"%chi_v if chi_v else "—"}</div><div class="met-l">Calinski-Harabasz</div><div class="met-cap">makin besar makin besar</div></div>
        </div>
        """, unsafe_allow_html=True)

        k_range  = cm.get("k_range") or []
        sil_list = cm.get("silhouette_per_k") or []
        wcss_list= cm.get("wcss_per_k") or []

        if k_range and sil_list and len(k_range) == len(sil_list):
            ev1, ev2 = st.columns(2)
            with ev1:
                st.markdown(sec("Silhouette per K","Pemilihan Jumlah Klaster"), unsafe_allow_html=True)
                best_i = int(max(range(len(sil_list)), key=lambda j: sil_list[j]))
                fig_sil = go.Figure(go.Scatter(x=list(k_range), y=list(sil_list), mode="lines+markers",
                                               line=dict(color=FIRE, width=2), marker=dict(size=6, color=FIRE)))
                fig_sil.add_trace(go.Scatter(x=[k_range[best_i]], y=[sil_list[best_i]], mode="markers",
                                             marker=dict(size=12, color=FIRE2, line=dict(color=INK, width=2)),
                                             name="Terbaik"))
                fig_sil.update_layout(**{**PLOTLY_LAYOUT, "height":240, "showlegend":False,
                                         "xaxis_title":"Jumlah Klaster (K)", "yaxis_title":"Silhouette Score"})
                st.plotly_chart(fig_sil, use_container_width=True)
            with ev2:
                if wcss_list and len(wcss_list) == len(k_range):
                    st.markdown(sec("Elbow Method","WCSS per K"), unsafe_allow_html=True)
                    fig_wcss = go.Figure(go.Scatter(x=list(k_range), y=list(wcss_list), mode="lines+markers",
                                                    line=dict(color="#9A9A9A", width=2), marker=dict(size=6, color="#9A9A9A")))
                    fig_wcss.update_layout(**{**PLOTLY_LAYOUT, "height":240, "showlegend":False,
                                              "xaxis_title":"Jumlah Klaster (K)", "yaxis_title":"WCSS (Inertia)"})
                    st.plotly_chart(fig_wcss, use_container_width=True)
                else:
                    cs = cm.get("cluster_sizes") or {}
                    if cs:
                        st.markdown(sec("Ukuran Klaster","Distribusi Anggota"), unsafe_allow_html=True)
                        cs_df = pd.DataFrame({"Klaster": list(cs.keys()), "Jumlah": [int(v) for v in cs.values()]})
                        fig_cs = px.bar(cs_df, x="Klaster", y="Jumlah", text="Jumlah",
                                        color="Jumlah", color_continuous_scale=[INK4, FIRE])
                        fig_cs.update_traces(textposition="outside", textfont_color=T3)
                        fig_cs.update_layout(**{**PLOTLY_LAYOUT, "coloraxis_showscale":False, "height":240})
                        st.plotly_chart(fig_cs, use_container_width=True)

        st.markdown('<div class="divider"></div>', unsafe_allow_html=True)
        st.markdown(sec("Keterbatasan","Yang Perlu Diingat Saat Membaca Dashboard"), unsafe_allow_html=True)
        st.markdown(ins(
            f'{svg("warning",12,RED_SOFT)} <strong>Sentimen berbasis leksikon</strong> tidak memahami sarkasme dan bahasa gaul.<br>'
            f'{svg("warning",12,RED_SOFT)} <strong>K-Means</strong> mengelompokkan berdasarkan kemiripan kata, bukan makna semantik.<br>'
            f'{svg("warning",12,RED_SOFT)} Data mencerminkan satu rentang waktu pengambilan, bukan opini permanen.',
            "muted"
        ), unsafe_allow_html=True)

    # ══════════════════════════════════════════════
    # PANEL 6 — DATA TWEET
    # ══════════════════════════════════════════════
    elif panel == "tweets":
        st.markdown(sec("Top Engagement","5 Tweet dengan Retweet Terbanyak"), unsafe_allow_html=True)
        if has_rt:
            top5 = fdf.nlargest(5,"retweet_count")
            for _, row in top5.iterrows():
                rt_  = f'<span>{svg("retweet",10,T3)} {int(row["retweet_count"])}</span>' if has_rt else ""
                fav_ = f'<span>{svg("heart",10,T3)} {int(row["favorite_count"])}</span>' if has_fav else ""
                tgl_ = f'<span>{svg("calendar",10,T3)} {str(row["tanggal"].date()) if pd.notna(row["tanggal"]) else ""}</span>'
                snt = ""
                if "sentiment_label" in row:
                    sl = str(row["sentiment_label"])
                    if "pos" in sl:   snt = '<span class="bdg bdg-pos">POSITIF</span>'
                    elif "neg" in sl: snt = '<span class="bdg bdg-neg">NEGATIF</span>'
                    else:             snt = '<span class="bdg bdg-neu">NETRAL</span>'
                st.markdown(f'<div class="tweet fire">{snt}&nbsp;{row[T_COL]}<div class="tweet-meta">{rt_}{fav_}{tgl_}</div></div>', unsafe_allow_html=True)

        st.markdown('<div class="divider"></div>', unsafe_allow_html=True)
        st.markdown(sec("Raw Data","Data Tweet Hasil Filter"), unsafe_allow_html=True)
        default_cols = [c for c in ["tanggal","full_text","clean_text","cluster","cluster_label",
                                    "sentiment_label","favorite_count","retweet_count","reply_count"] if c in fdf.columns]
        sel_cols = st.multiselect("Kolom yang ditampilkan:", options=fdf.columns.tolist(), default=default_cols)
        if sel_cols:
            st.dataframe(fdf[sel_cols].reset_index(drop=True), use_container_width=True, hide_index=True, height=400)
            csv_bytes = fdf[sel_cols].to_csv(index=False).encode("utf-8")
            st.download_button(
                label=f"{svg('download',13,FIRE2)} Download CSV hasil filter",
                data=csv_bytes, file_name="bansos_filtered.csv", mime="text/csv",
            )
        else:
            st.info("Pilih setidaknya satu kolom untuk ditampilkan.")