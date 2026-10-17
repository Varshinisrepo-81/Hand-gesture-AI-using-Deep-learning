"""
UI Design System and Styling Module for Streamlit Dashboard.
Provides high-fidelity CSS theming, custom metric cards, badges, and reusable components.
"""

import streamlit as st
from typing import Dict, Any, Optional

CUSTOM_CSS = """
<style>
    /* Google Fonts */
    @import url('https://fonts.googleapis.com/css2?family=Outfit:wght@300;400;500;600;700;800&family=JetBrains+Mono:wght@400;600&display=swap');

    html, body, [class*="css"] {
        font-family: 'Outfit', -apple-system, BlinkMacSystemFont, sans-serif;
    }

    code, pre {
        font-family: 'JetBrains Mono', monospace !important;
    }

    /* Main Container Padding */
    .block-container {
        padding-top: 1.8rem;
        padding-bottom: 2.5rem;
        max-width: 1400px;
    }

    /* Gradient Brand Header */
    .brand-hero {
        background: linear-gradient(135deg, rgba(20, 26, 40, 0.85) 0%, rgba(13, 17, 23, 0.95) 100%);
        border: 1px solid rgba(88, 166, 255, 0.2);
        border-radius: 16px;
        padding: 24px 30px;
        margin-bottom: 24px;
        box-shadow: 0 8px 32px 0 rgba(0, 0, 0, 0.37);
        backdrop-filter: blur(10px);
        display: flex;
        justify-content: space-between;
        align-items: center;
        flex-wrap: wrap;
        gap: 15px;
    }

    .brand-title {
        font-size: 2.1rem;
        font-weight: 800;
        background: linear-gradient(90deg, #58a6ff 0%, #39d353 50%, #79c0ff 100%);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        margin: 0;
        line-height: 1.2;
    }

    .brand-subtitle {
        color: #8b949e;
        font-size: 1.05rem;
        margin-top: 6px;
        font-weight: 400;
    }

    /* Status Pill Badges */
    .pill-badge {
        display: inline-flex;
        align-items: center;
        gap: 6px;
        padding: 6px 14px;
        border-radius: 9999px;
        font-size: 0.82rem;
        font-weight: 600;
        letter-spacing: 0.03em;
        text-transform: uppercase;
    }

    .pill-success {
        background: rgba(46, 160, 67, 0.15);
        color: #3fb950;
        border: 1px solid rgba(63, 185, 80, 0.35);
    }

    .pill-warning {
        background: rgba(210, 153, 34, 0.15);
        color: #d29922;
        border: 1px solid rgba(210, 153, 34, 0.35);
    }

    .pill-danger {
        background: rgba(248, 81, 73, 0.15);
        color: #f85149;
        border: 1px solid rgba(248, 81, 73, 0.35);
    }

    .pill-info {
        background: rgba(88, 166, 255, 0.15);
        color: #58a6ff;
        border: 1px solid rgba(88, 166, 255, 0.35);
    }

    /* Glass Cards */
    .glass-card {
        background: rgba(22, 27, 34, 0.7);
        border: 1px solid rgba(240, 246, 252, 0.1);
        border-radius: 14px;
        padding: 20px;
        box-shadow: 0 4px 20px rgba(0, 0, 0, 0.25);
        backdrop-filter: blur(8px);
        margin-bottom: 18px;
        transition: transform 0.2s ease, border-color 0.2s ease;
    }

    .glass-card:hover {
        border-color: rgba(88, 166, 255, 0.35);
    }

    /* Gesture Display Tile in Live Telemetry */
    .telemetry-tile {
        background: linear-gradient(145deg, #161b22, #0d1117);
        border: 1px solid rgba(88, 166, 255, 0.25);
        border-radius: 14px;
        padding: 20px;
        text-align: center;
        margin-bottom: 15px;
    }

    .telemetry-gesture-title {
        font-size: 1.8rem;
        font-weight: 700;
        margin-top: 8px;
        margin-bottom: 4px;
    }

    .telemetry-conf-label {
        color: #8b949e;
        font-size: 0.9rem;
    }

    /* Custom Metric Box */
    .stat-metric-card {
        background: #161b22;
        border: 1px solid #30363d;
        border-radius: 12px;
        padding: 16px 20px;
        margin-bottom: 12px;
        display: flex;
        flex-direction: column;
        justify-content: center;
    }

    .stat-metric-val {
        font-size: 1.9rem;
        font-weight: 700;
        color: #58a6ff;
        line-height: 1.1;
    }

    .stat-metric-label {
        font-size: 0.85rem;
        color: #8b949e;
        text-transform: uppercase;
        letter-spacing: 0.05em;
        margin-top: 4px;
    }

    /* Mini Softmax Bar */
    .prob-bar-container {
        margin-bottom: 10px;
    }

    .prob-bar-header {
        display: flex;
        justify-content: space-between;
        font-size: 0.85rem;
        margin-bottom: 4px;
    }

    .prob-bar-bg {
        background: #21262d;
        height: 8px;
        border-radius: 4px;
        overflow: hidden;
    }

    .prob-bar-fill {
        height: 100%;
        border-radius: 4px;
        transition: width 0.3s ease;
    }

    /* Styled buttons */
    .stButton>button {
        border-radius: 10px;
        font-weight: 600;
        padding: 0.5rem 1.2rem;
        transition: all 0.2s ease;
    }

    .stButton>button:hover {
        transform: translateY(-1px);
    }

    /* Sidebar Clean Styling */
    section[data-testid="stSidebar"] {
        background-color: #0d1117;
        border-right: 1px solid #30363d;
    }

    .sidebar-brand {
        padding: 10px 0 20px 0;
        border-bottom: 1px solid #30363d;
        margin-bottom: 15px;
    }
</style>
"""

GESTURE_ICONS = {
    "Open Palm": "🖐️",
    "Fist": "✊",
    "Thumbs Up": "👍",
    "Thumbs Down": "👎",
    "Peace Sign": "✌️",
    "Pointing Up": "☝️",
    "OK Sign": "👌",
    "Rock Sign": "🤘",
    "Call Me": "🤙",
    "Three Fingers": "🤟",
    "Four Fingers": "🖐️",
    "Number 0": "0️⃣",
    "Number 1": "1️⃣",
    "Number 2": "2️⃣",
    "Number 3": "3️⃣",
    "Number 4": "4️⃣",
    "Number 5": "5️⃣",
    "Gun Sign": "🔫",
    "Love You Sign": "🤟",
    "Spock Salute": "🖖",
    "Pinky Up": "🤙",
    "Pinch": "🤏",
    "Crossed Fingers": "🤞",
    "Stop Sign": "🛑",
    "Claw Hand": "🦅",
    "Pointing Forward": "👉",
    "Thumb Left": "👈",
    "Thumb Right": "👉",
    "Two Fingers Down": "👇",
    "Salute": "🫡",
    "Ring Extension": "💍",
    "C-Shape": "🧲",
    "Uncertain / Low Confidence": "⚠️",
    "Model Not Loaded": "❌",
    "No Hand Detected": "🔍"
}

GESTURE_DESCRIPTIONS = {
    "Open Palm": "All five fingers fully extended outward with flat palm facing the camera.",
    "Fist": "All five digits fully curled and clenched into the palm center.",
    "Thumbs Up": "Thumb pointing vertically upward while remaining four fingers are curled.",
    "Thumbs Down": "Thumb pointing vertically downward while remaining four fingers are curled.",
    "Peace Sign": "Index and middle fingers extended in a V-shape; thumb, ring, and pinky curled.",
    "Pointing Up": "Index finger extended straight up; all other four digits curled.",
    "OK Sign": "Thumb tip touching index fingertip in an O-ring; middle, ring, and pinky extended.",
    "Rock Sign": "Index and pinky fingers extended; middle, ring, and thumb curled over knuckles.",
    "Call Me": "Thumb and pinky extended laterally outward; index, middle, and ring fingers curled.",
    "Three Fingers": "Thumb, index, and middle fingers extended; ring and pinky curled into palm.",
    "Four Fingers": "Index, middle, ring, and pinky upright together; thumb folded flat across palm.",
    "Number 0": "All 4 fingertips curved downward to meet the thumb tip, forming a closed oval.",
    "Number 1": "Index finger held straight up with thumb locked over curled knuckles.",
    "Number 2": "Index and middle fingers extended parallel straight up; thumb over curled ring and pinky.",
    "Number 3": "Index, middle, and ring fingers extended parallel; pinky held down by thumb.",
    "Number 4": "All four fingers held upright side-by-side with thumb tucked across the palm.",
    "Number 5": "All five fingers and thumb fully extended and splayed wide apart.",
    "Gun Sign": "Thumb pointing straight up like a hammer, index extended horizontally, others curled.",
    "Love You Sign": "American Sign Language ILY: thumb, index, and pinky extended; middle and ring curled.",
    "Spock Salute": "Vulcan salute: index and middle together, ring and pinky together, split in a V.",
    "Pinky Up": "Only pinky finger extended straight up; other four digits curled into a fist.",
    "Pinch": "Thumb tip and index fingertip held very close together; other three digits curled.",
    "Crossed Fingers": "Middle finger crossed over index finger for good luck; others curled.",
    "Stop Sign": "Rigid flat open hand with fingers together facing outward as a barrier.",
    "Claw Hand": "All five fingers curved and hooked in half-flexion like an eagle's claw.",
    "Pointing Forward": "Index finger pointing directly forward toward the camera lens.",
    "Thumb Left": "Thumb extended pointing horizontally left while four fingers are curled.",
    "Thumb Right": "Thumb extended pointing horizontally right while four fingers are curled.",
    "Two Fingers Down": "Index and middle fingers pointing straight downward; others curled.",
    "Salute": "Rigid flat hand tilted diagonally as a military salute gesture.",
    "Ring Extension": "Thumb tip touching middle fingertip in a ring; index, ring, and pinky extended.",
    "C-Shape": "Fingers and thumb curved facing each other forming the letter C arc."
}


def inject_custom_css():
    """Injects high-fidelity custom stylesheet into Streamlit app."""
    st.markdown(CUSTOM_CSS, unsafe_allow_html=True)


def render_brand_header(title: str, subtitle: str, badge_text: str = "Deep Learning AI"):
    """Renders sleek top hero header with gradient branding and status badge."""
    html = f"""
    <div class="brand-hero">
        <div>
            <h1 class="brand-title">{title}</h1>
            <p class="brand-subtitle">{subtitle}</p>
        </div>
        <div>
            <span class="pill-badge pill-info">⚡ {badge_text}</span>
        </div>
    </div>
    """
    st.markdown(html, unsafe_allow_html=True)


def render_metric_card(label: str, value: str, subtext: Optional[str] = None):
    """Renders a styled dark metric card."""
    sub_html = f"<div style='font-size:0.75rem; color:#6e7681; margin-top:3px;'>{subtext}</div>" if subtext else ""
    html = f"""
    <div class="stat-metric-card">
        <div class="stat-metric-val">{value}</div>
        <div class="stat-metric-label">{label}</div>
        {sub_html}
    </div>
    """
    st.markdown(html, unsafe_allow_html=True)


def render_gesture_cards():
    """Renders informative interactive cards for the 32 supported gestures in categorized tabs."""
    categories = {
        "🌟 Popular & Classic": [
            ("Open Palm", "🖐️", "5 Extended", "#2ea043"),
            ("Fist", "✊", "All Curled", "#f85149"),
            ("Thumbs Up", "👍", "Thumb Up", "#58a6ff"),
            ("Thumbs Down", "👎", "Thumb Down", "#f85149"),
            ("Peace Sign", "✌️", "Index + Mid", "#a371f7"),
            ("Pointing Up", "☝️", "Index Only", "#d29922"),
            ("Stop Sign", "🛑", "Flat Barrier", "#f85149"),
            ("Salute", "🫡", "Angled Rigid", "#39d353")
        ],
        "🤘 Signs & Signals": [
            ("OK Sign", "👌", "Thumb + Index", "#58a6ff"),
            ("Rock Sign", "🤘", "Index + Pinky", "#f85149"),
            ("Call Me", "🤙", "Thumb + Pinky", "#a371f7"),
            ("Gun Sign", "🔫", "Thumb Up + Index", "#d29922"),
            ("Love You Sign", "🤟", "Thumb+Idx+Pnk", "#f778ba"),
            ("Spock Salute", "🖖", "Vulcan V-Split", "#79c0ff"),
            ("Pinky Up", "🤙", "Pinky Only", "#a371f7"),
            ("Pinch", "🤏", "Thumb & Idx Close", "#e3b341"),
            ("Crossed Fingers", "🤞", "Mid over Idx", "#58a6ff")
        ],
        "🔢 Numbers 0 to 5": [
            ("Number 0", "0️⃣", "Closed Oval", "#8b949e"),
            ("Number 1", "1️⃣", "Index Upright", "#58a6ff"),
            ("Number 2", "2️⃣", "2 Upright Parallel", "#39d353"),
            ("Number 3", "3️⃣", "3 Upright (No Pinky)", "#d29922"),
            ("Number 4", "4️⃣", "4 Upright Flat", "#a371f7"),
            ("Number 5", "5️⃣", "5 Splayed Wide", "#f778ba"),
            ("Three Fingers", "🤟", "Thumb+Idx+Mid", "#39d353"),
            ("Four Fingers", "🖐️", "4 Upright", "#58a6ff")
        ],
        "📐 Poses & Geometry": [
            ("Claw Hand", "🦅", "5 Half-Hooked", "#d29922"),
            ("Pointing Forward", "👉", "Index to Screen", "#58a6ff"),
            ("Thumb Left", "👈", "Thumb to Left", "#39d353"),
            ("Thumb Right", "👉", "Thumb to Right", "#79c0ff"),
            ("Two Fingers Down", "👇", "2 Pointing Down", "#f85149"),
            ("Ring Extension", "💍", "Thumb & Mid Ring", "#f778ba"),
            ("C-Shape", "🧲", "C Curved Arc", "#a371f7")
        ]
    }

    tabs = st.tabs(list(categories.keys()))
    for tab, (cat_name, items) in zip(tabs, categories.items()):
        with tab:
            cols = st.columns(min(len(items), 4))
            for i, (name, icon, rule, color) in enumerate(items):
                col = cols[i % len(cols)]
                with col:
                    st.markdown(
                        f"""
                        <div class="glass-card" style="text-align:center; padding:12px 8px; margin-bottom:10px;">
                            <div style="font-size:2rem; margin-bottom:4px;">{icon}</div>
                            <div style="font-weight:700; font-size:0.95rem; color:{color};">{name}</div>
                            <div style="font-size:0.75rem; color:#8b949e; margin-top:3px;">{rule}</div>
                        </div>
                        """,
                        unsafe_allow_html=True
                    )


def render_softmax_bars(probs_dict: Dict[str, float], top_k: int = 5):
    """Renders ranked horizontal bars for the top-k highest predicted classes."""
    if not probs_dict:
        return

    st.markdown(f"<div style='font-weight:600; font-size:0.95rem; margin-bottom:10px; color:#c9d1d9;'>Top-{top_k} Predictions</div>", unsafe_allow_html=True)

    # Sort classes by probability descending
    sorted_probs = sorted(probs_dict.items(), key=lambda item: item[1], reverse=True)
    top_items = sorted_probs[:top_k]

    colors = ["#2ea043", "#58a6ff", "#a371f7", "#d29922", "#f778ba"]

    for idx, (gesture, p) in enumerate(top_items):
        icon = GESTURE_ICONS.get(gesture, "•")
        pct = int(p * 100)
        bar_color = colors[idx % len(colors)]
        st.markdown(
            f"""
            <div class="prob-bar-container">
                <div class="prob-bar-header">
                    <span>{icon} <b>{gesture}</b></span>
                    <span style="font-family:'JetBrains Mono'; color:#8b949e;">{pct}%</span>
                </div>
                <div class="prob-bar-bg">
                    <div class="prob-bar-fill" style="width: {pct}%; background-color: {bar_color};"></div>
                </div>
            </div>
            """,
            unsafe_allow_html=True
        )

    # Expander for full 32 probability breakdown
    with st.expander(f"🔍 View all {len(probs_dict)} class probabilities"):
        for gesture, p in sorted_probs:
            icon = GESTURE_ICONS.get(gesture, "•")
            st.write(f"{icon} **{gesture}**: `{p * 100:.2f}%`")
