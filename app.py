import streamlit as st
import os
from dotenv import load_dotenv
from langchain_openrouter import ChatOpenRouter
import plotly.express as px
import pandas as pd
import json
from datetime import datetime

load_dotenv()
os.environ["OPENROUTER_API_KEY"] = st.secrets["OPENROUTER_API_KEY"]

# ---- NEW: Import the CrewAI pipeline ----
from agents import create_crew

st.set_page_config(page_title="Product Feedback Agent", page_icon="📊", layout="centered")

st.title("Product Feedback Agent System")
st.markdown("**Helping companies improve their products using Large Language Models**")

# Sidebar
st.sidebar.title("Analysis Settings")
agent_mode = st.sidebar.selectbox(
    "Choose Analysis Mode",
    ["1. Full 5-Agent Analysis (Recommended)", "2. Quick Summary Agent",
     "3. Theme & Sentiment Deep Dive", "4. Recommendation-Focused Agent",
     "5. Professional Executive Report"]
)
temperature = st.sidebar.slider("Temperature (Creativity)", 0.0, 1.0, 0.65)

# ==================== HELPER FUNCTIONS ====================
def clear_history():
    if 'history' in st.session_state:
        st.session_state.history = []

def save_to_history(comments_preview, comments_full, executive_summary, key_themes,
                    sentiment, insights, recommendations, raw_report=None):  # NEW: raw_report optional
    if 'history' not in st.session_state:
        st.session_state.history = []
    entry_id = len(st.session_state.history) + 1
    timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    st.session_state.history.append({
        "id": entry_id,
        "timestamp": timestamp,
        "comments_preview": comments_preview[:100] + ("..." if len(comments_preview) > 100 else ""),
        "comments_full": comments_full,
        "executive_summary": executive_summary,
        "key_themes": key_themes,   # can be list of strings or list of dicts
        "sentiment": sentiment,
        "insights": insights,
        "recommendations": recommendations,
        "raw_report": raw_report     # NEW: store the complete markdown report
    })

def display_analysis(result):
    """Display a single analysis result (used in history view)."""
    # If we have a raw report from the Crew, show it directly
    if result.get("raw_report"):
        st.markdown(result["raw_report"])
        return

    # Otherwise, fall back to the structured display (for older entries)
    st.markdown("### Executive Summary")
    st.write(result.get("executive_summary", "No summary"))
    st.markdown("### Key Themes")
    themes = result.get("key_themes", [])
    for theme in themes:
        if isinstance(theme, dict):
            st.write(f"• **{theme.get('theme', 'Unknown')}** ({theme.get('sentiment', 'N/A')}, ~{theme.get('frequency', '?')} mentions)")
        else:
            st.write(f"• {theme}")
    st.markdown("### Recommendations")
    for rec in result.get("recommendations", []):
        st.write(f"• {rec}")

# ==================== TOP TABS ====================
def export_history_csv():
    if 'history' not in st.session_state or not st.session_state.history:
        return None

    history_data = []
    for entry in st.session_state.history:
        # Convert key_themes to a string safely
        themes = entry.get('key_themes', [])
        if isinstance(themes, list):
            themes_str = ", ".join(str(t) for t in themes)  # handles dicts or strings
        else:
            themes_str = str(themes)

        recs = entry.get('recommendations', [])
        recs_str = "\n".join(str(r) for r in recs)

        history_data.append({
            "ID": entry.get('id', ''),
            "Timestamp": entry.get('timestamp', ''),
            "Comments Preview": entry.get('comments_preview', ''),
            "Full Comments": entry.get('comments_full', ''),
            "Executive Summary": entry.get('executive_summary', ''),
            "Key Themes": themes_str,
            "Positive %": entry.get('sentiment', {}).get('positive', 0),
            "Negative %": entry.get('sentiment', {}).get('negative', 0),
            "Neutral %": entry.get('sentiment', {}).get('neutral', 0),
            "Insights": entry.get('insights', ''),
            "Recommendations": recs_str
        })

    return pd.DataFrame(history_data)


tab1, tab2 = st.tabs(["New Analysis", "History"])

# Main Area
# ==================== TAB 1: NEW ANALYSIS ====================
with tab1:
    st.subheader("Paste user reviews / comments")

    uploaded_file = st.file_uploader("Upload a .txt or .csv file", type=["txt", "csv"])

    comments = st.text_area(
        "Enter customer comments here:",
        height=180,
        placeholder="Paste reviews here..."
    )

    # Sample Dropdown
    st.subheader("Quick Demo Samples")
    sample_option = st.selectbox(
        "Load a sample dataset:",
        ["Select a sample...", "Phone App Feedback", "Food Delivery Feedback",
         "Laptop Feedback", "Gaming Console Feedback"]
    )

    samples = {
        "Phone App Feedback": "The app keeps crashing on Android. Battery drains way too fast. Beautiful design but too expensive. UI is confusing.",
        "Food Delivery Feedback": "Food always arrives cold. Delivery is late. Great variety but prices too high. Driver was rude.",
        "Laptop Feedback": "Screen is bright. Keyboard feels cheap. Fast performance. Overheats during gaming. Best laptop in this price.",
        "Gaming Console Feedback": "Graphics amazing. Controller feels great. Loading times too long. Overheats after 2 hours."
    }

    if sample_option != "Select a sample...":
        st.text_area("Sample Comments (copy & paste above):",
                     value=samples[sample_option], height=100, disabled=True)

    # ====================== ANALYZE BUTTON ======================
    if st.button("Analyze Feedback", type="primary", use_container_width=True):
        final_comments = comments
        if uploaded_file:
            final_comments = final_comments + "\n\n" + uploaded_file.getvalue().decode("utf-8")

        if not final_comments or len(final_comments.strip()) < 30:
            st.error("Please enter comments or upload a file.")
        else:
            with st.spinner("Analyzing feedback and generating charts..."):
                # ========== NEW: CrewAI branch for mode 1 ==========
                if agent_mode == "1. Full 5-Agent Analysis (Recommended)":
                    # Run the CrewAI pipeline
                    crew = create_crew(final_comments)
                    result = crew.kickoff()      # result is a string (the final report)

                    st.success("Analysis Complete!")
                    st.markdown(result)           # display the report

                    # Save to history with the raw report
                    save_to_history(
                        comments_preview=final_comments[:100],
                        comments_full=final_comments,
                        executive_summary="",      # not used for raw report
                        key_themes=[],             # not used
                        sentiment={},              # not used
                        insights="",               # not used
                        recommendations=[],        # not used
                        raw_report=result
                    )

                # ========== Original logic for other modes ==========
                else:
                    llm = ChatOpenRouter(
                        model="meta-llama/llama-3.3-70b-instruct",
                        temperature=temperature,
                        max_tokens=2500
                    )

                    # (same prompt building as before, but note that we removed the mode 1 case from here)
                    if agent_mode == "2. Quick Summary Agent":
                        prompt = f"""You are a fast executive summary agent.
                    Focus ONLY on producing a concise 3-5 sentence summary of the key strengths and pain points.

                    Comments:
                    {final_comments}

                    Do not list themes or recommendations. Summary only."""

                    elif agent_mode == "3. Theme & Sentiment Deep Dive":
                        prompt = f"""You are a theme & sentiment analysis specialist.
                    Extract and list the top 6–10 themes from the comments.
                    For each theme:
                    - Name the theme
                    - Sentiment (positive / negative / neutral)
                    - Approximate frequency (% or count)
                    - 1–2 short example quotes

                    Comments:
                    {final_comments}

                    Output only a markdown bullet list. No summary or recommendations."""

                    elif agent_mode == "4. Recommendation-Focused Agent":
                        prompt = f"""You are a product improvement strategist.
                    Read the comments and generate ONLY a numbered list of 7–10 concrete, prioritized recommendations for the next product version.

                    Each recommendation should include:
                    - Priority (High/Medium/Low)
                    - What to change/add/fix
                    - Why (based on user feedback)

                    Comments:
                    {final_comments}

                    Output only the numbered list. No summary or theme list."""

                    else:  # "5. Professional Executive Report"
                        prompt = f"""You are a professional business report writer.
                    Create a polished executive-level report from the customer feedback.

                    Include:
                    1. Executive Summary (3–5 sentences)
                    2. Key Strengths
                    3. Main Pain Points
                    4. Strategic Recommendations (5–7 items)

                    Use formal, confident, business-oriented language.

                    Comments:
                    {final_comments}

                    Output the full report in markdown format with headings."""

                    response = llm.invoke(prompt)
                    st.success("Analysis Complete!")
                    st.markdown(response.content)

                    # Save simple structured data for history
                    if agent_mode == "2. Quick Summary Agent":
                        exec_summary = response.content.strip()
                        key_themes = []
                        sentiment_dist = {}
                        recommendations = []
                    elif agent_mode == "3. Theme & Sentiment Deep Dive":
                        exec_summary = ""
                        key_themes = [line.strip() for line in response.content.splitlines() if line.strip().startswith("-")]
                        sentiment_dist = {}
                        recommendations = []
                    elif agent_mode == "4. Recommendation-Focused Agent":
                        exec_summary = ""
                        key_themes = []
                        sentiment_dist = {}
                        recommendations = [line.strip() for line in response.content.splitlines() if line.strip().startswith(("1.", "2.", "3.", "4.", "5.", "6.", "7."))]
                    else:  # mode 5
                        exec_summary = ""
                        key_themes = []
                        sentiment_dist = {}
                        recommendations = []

                    save_to_history(
                        comments_preview=final_comments[:100],
                        comments_full=final_comments,
                        executive_summary=exec_summary,
                        key_themes=key_themes,
                        sentiment=sentiment_dist,
                        insights="",
                        recommendations=recommendations,
                        raw_report=None
                    )

# ==================== TAB 2: HISTORY ====================
with tab2:
    st.subheader("Analysis History")

    col1, col2 = st.columns([3, 1])
    with col1:
        st.write(f"**Total Analyses:** {len(st.session_state.get('history', []))}")
    with col2:
        df = export_history_csv()
        if df is not None:
            csv = df.to_csv(index=False)
            st.download_button("Export CSV", csv, "history.csv", "text/csv")

    if 'history' in st.session_state and st.session_state.history:
        # Display history in a table
        history_data = []
        for entry in reversed(st.session_state.history):
            pos = entry.get('sentiment', {}).get('positive', 0)
            sentiment_text = "Positive" if pos > 0.6 else "Mixed" if pos > 0.3 else "Negative"

            # Safely convert key_themes to a string for display
            themes = entry.get('key_themes', [])
            if isinstance(themes, list):
                themes_preview = ", ".join(str(t)[:50] for t in themes[:2])  # show first two themes, truncated
            else:
                themes_preview = str(themes)[:100]

            history_data.append({
                "ID": entry.get('id', ''),
                "Date": entry.get('timestamp', '')[:10],
                "Comments": entry.get('comments_preview', ''),
                "Themes": themes_preview,
                "Sentiment": sentiment_text
            })

        df_history = pd.DataFrame(history_data)

        # Make it clickable
        for idx, row in df_history.iterrows():
            cols = st.columns([1, 2, 4, 3, 1])
            with cols[0]:
                st.write(f"#{row['ID']}")
            with cols[1]:
                st.write(row['Date'])
            with cols[2]:
                st.write(row['Comments'])
            with cols[3]:
                st.write(row['Themes'])
            with cols[4]:
                st.write(row['Sentiment'])

            if st.button("View", key=f"view_{row['ID']}"):
                st.session_state.viewing_entry = row['ID']
            st.divider()

        # Show selected history entry
        if 'viewing_entry' in st.session_state:
            entry = next((e for e in st.session_state.history if e.get('id') == st.session_state.viewing_entry), None)
            if entry:
                st.markdown("---")
                st.subheader(f"Full Analysis #{entry.get('id')}")
                st.caption(f"Analyzed on: {entry.get('timestamp', '')}")

                with st.expander("View Full Comments"):
                    st.write(entry.get('comments_full', ''))

                # Use display_analysis to show the stored data
                display_analysis(entry)
    else:
        st.info("No history yet. Run an analysis in the 'New Analysis' tab!")

    if st.button("Clear All History"):
        clear_history()
        st.rerun()

st.caption("Powered by OpenRouter • Click history entries to view full analysis")
