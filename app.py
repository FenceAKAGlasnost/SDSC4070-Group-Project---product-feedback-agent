import streamlit as st
import os
from dotenv import load_dotenv
from langchain_openrouter import ChatOpenRouter
import plotly.express as px
import pandas as pd
import json

load_dotenv()
os.environ["OPENROUTER_API_KEY"] = st.secrets["OPENROUTER_API_KEY"]

st.set_page_config(page_title="Product Feedback Agent", page_icon="📊", layout="centered")

st.title("📊 Product Feedback Agent System")
st.markdown("**Helping companies improve their products using Large Language Models**")

# Sidebar
st.sidebar.title("⚙️ Analysis Settings")
agent_mode = st.sidebar.selectbox(
    "Choose Analysis Mode",
    ["1. Full 5-Agent Analysis (Recommended)", "2. Quick Summary Agent", 
     "3. Theme & Sentiment Deep Dive", "4. Recommendation-Focused Agent", 
     "5. Professional Executive Report"]
)
temperature = st.sidebar.slider("Temperature (Creativity)", 0.0, 1.0, 0.65)

# ==================== TOP TABS ====================
tab1, tab2 = st.tabs(["📝 New Analysis", "📋 History"])

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
st.subheader("📋 Quick Demo Samples")
sample_option = st.selectbox(
    "Load a sample dataset:",
    ["Select a sample...", "📱 Phone App Feedback", "🍔 Food Delivery Feedback", 
     "💻 Laptop Feedback", "🎮 Gaming Console Feedback"]
)

samples = {
    "📱 Phone App Feedback": "The app keeps crashing on Android. Battery drains way too fast. Beautiful design but too expensive. UI is confusing.",
    "🍔 Food Delivery Feedback": "Food always arrives cold. Delivery is late. Great variety but prices too high. Driver was rude.",
    "💻 Laptop Feedback": "Screen is bright. Keyboard feels cheap. Fast performance. Overheats during gaming. Best laptop in this price.",
    "🎮 Gaming Console Feedback": "Graphics amazing. Controller feels great. Loading times too long. Overheats after 2 hours."
}

if sample_option != "Select a sample...":
    st.text_area("Sample Comments (copy & paste above):", 
                 value=samples[sample_option], height=100, disabled=True)

# ====================== ANALYZE BUTTON ======================
if st.button("🚀 Analyze Feedback", type="primary", use_container_width=True):
    final_comments = comments
    if uploaded_file:
        final_comments = final_comments + "\n\n" + uploaded_file.getvalue().decode("utf-8")

    if not final_comments or len(final_comments.strip()) < 30:
        st.error("Please enter comments or upload a file.")
    else:
        with st.spinner("Analyzing feedback and generating charts..."):
            llm = ChatOpenRouter(
                model="meta-llama/llama-3.3-70b-instruct",
                temperature=temperature,
                max_tokens=2500
            )

            # ────────────────────────────────────────────────
               #  Five clearly different prompt branches
            # ────────────────────────────────────────────────

            if agent_mode == "1. Full 5-Agent Analysis (Recommended)":
                prompt = f"""You are simulating a full multi-agent product feedback analysis pipeline.
            Analyze the following comments and return **only valid JSON** (no other text).

            Comments:
            {comments}

            Return exactly this JSON structure:
            {{
              "executive_summary": "3-5 sentence summary",
              "key_themes": [
                {{"theme": "Battery life", "sentiment": "negative", "frequency": 12, "examples": ["drains fast", "only 4 hours"]}},
                ...
              ],
              "sentiment_distribution": {{"positive": 0.25, "negative": 0.60, "neutral": 0.15}},
              "recommendations": ["Fix memory leaks - high priority because...", ...]
            }}
            
            Be accurate. Use percentages for sentiment that sum to 1.0. Estimate frequency realistically."""

            elif agent_mode == "2. Quick Summary Agent":
                prompt = f"""You are a fast executive summary agent.
               Focus ONLY on producing a concise 3-5 sentence summary of the key strengths and pain points.

               Comments:
               {comments}

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
               {comments}

               Output only a markdown bullet list. No summary or recommendations."""

            elif agent_mode == "4. Recommendation-Focused Agent":
                   prompt = f"""You are a product improvement strategist.
               Read the comments and generate ONLY a numbered list of 7–10 concrete, prioritized recommendations for the next product version.

               Each recommendation should include:
               - Priority (High/Medium/Low)
               - What to change/add/fix
               - Why (based on user feedback)

               Comments:
               {comments}

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
               {comments}

               Output the full report in markdown format with headings."""

# Invoke LLM
            response = llm.invoke(prompt)

            st.success("✅ Analysis Complete!")

            # ────────────────────────────────────────────────
            # Special handling for Agent 1 (with charts)
            # ────────────────────────────────────────────────
            if agent_mode == "1. Full 5-Agent Analysis (Recommended)":
             try:
                # Clean the response content (remove extra whitespace, backticks, etc.)
                raw = response.content.strip()
                if raw.startswith("```json"):
                    raw = raw.split("```json", 1)[1].split("```", 1)[0]
                elif raw.startswith("```"):
                    raw = raw.split("```", 2)[1]
            
                data = json.loads(raw)
            
                # ──────────────── TEXT REPORT ────────────────
                st.markdown("### Executive Summary")
                st.write(data.get("executive_summary", "No summary available"))

                st.markdown("### Key Themes")
                for theme in data.get("key_themes", []):
                    st.write(f"• **{theme.get('theme', 'Unknown')}** ({theme.get('sentiment', 'N/A')}, ~{theme.get('frequency', '?')} mentions)")

                # ──────────────── CHARTS ────────────────
                st.markdown("### Visualizations")
                col1, col2 = st.columns(2)

                # Pie chart: Sentiment Distribution
                with col1:
                    sent = data.get("sentiment_distribution", {"positive": 0.33, "negative": 0.33, "neutral": 0.34})
                    fig_pie = px.pie(
                        values=list(sent.values()),
                        names=list(sent.keys()),
                        title="Sentiment Distribution",
                        hole=0.4,
                        color_discrete_sequence=["#66c2a5", "#fc8d62", "#8da0cb"]
                    )
                    st.plotly_chart(fig_pie, use_container_width=True)

                # Bar chart: Top Themes by Frequency
                with col2:
                    themes = data.get("key_themes", [])
                    if themes:
                        df = pd.DataFrame(themes)
                        fig_bar = px.bar(
                            df,
                            x="theme",
                            y="frequency",
                            color="sentiment",
                            title="Top Themes by Mention Frequency",
                            color_discrete_map={"positive": "#66c2a5", "negative": "#fc8d62", "neutral": "#8da0cb"}
                        )
                        st.plotly_chart(fig_bar, use_container_width=True)
                    else:
                        st.info("No theme frequency data available for chart")

                # Recommendations
                st.markdown("### Recommendations")
                for rec in data.get("recommendations", []):
                    st.write(f"• {rec}")

             except json.JSONDecodeError as e:
                st.warning(f"JSON parsing failed: {e}")
                st.markdown("Raw LLM output (fallback):")
                st.code(response.content, language="json")
             except Exception as e:
                st.error(f"Unexpected error: {e}")
                st.markdown(response.content)
                 
# ==================== TAB 2: HISTORY ====================
with tab2:
    st.subheader("📋 Analysis History")
    
    col1, col2 = st.columns([3, 1])
    with col1:
        st.write(f"**Total Analyses:** {len(st.session_state.get('history', []))}")
    with col2:
        df = export_history_csv()
        if df is not None:
            csv = df.to_csv(index=False)
            st.download_button("📥 Export CSV", csv, "history.csv", "text/csv")
    
    if 'history' in st.session_state and st.session_state.history:
        # Display history in a table
        history_data = []
        for entry in reversed(st.session_state.history):
            pos = entry['sentiment'].get('positive', 0)
            sentiment_icon = "🟢" if pos > 0.6 else "🟡" if pos > 0.3 else "🔴"
            
            history_data.append({
                "ID": entry['id'],
                "Date": entry['timestamp'][:10],
                "Comments": entry['comments_preview'],
                "Themes": ", ".join(entry['key_themes'][:2]),
                "Sentiment": sentiment_icon
            })
        
        df_history = pd.DataFrame(history_data)
        
        # Make it clickable
        for idx, row in df_history.iterrows():
            col1, col2, col3, col4, col5 = st.columns([1, 2, 4, 3, 1])
            with col1:
                st.write(f"#{row['ID']}")
            with col2:
                st.write(row['Date'])
            with col3:
                st.write(row['Comments'])
            with col4:
                st.write(row['Themes'])
            with col5:
                st.write(row['Sentiment'])
            with col3:  # Use an extra column for the button
                if st.button("View", key=f"view_{row['ID']}"):
                    st.session_state.viewing_entry = row['ID']
            
            st.divider()
        
        # Show selected history entry
        if 'viewing_entry' in st.session_state:
            entry = next((e for e in st.session_state.history if e['id'] == st.session_state.viewing_entry), None)
            if entry:
                st.markdown("---")
                st.subheader(f"Full Analysis #{entry['id']}")
                st.caption(f"Analyzed on: {entry['timestamp']}")
                
                with st.expander("View Full Comments"):
                    st.write(entry['comments_full'])
                
                # Recreate result dict for display
                result = {
                    'executive_summary': entry['executive_summary'],
                    'key_themes': entry['key_themes'],
                    'sentiment': entry['sentiment'],
                    'insights': entry['insights'],
                    'recommendations': entry['recommendations']
                }
                display_analysis(result)
    else:
        st.info("No history yet. Run an analysis in the 'New Analysis' tab!")

    if st.button("Clear All History"):
        clear_history()
        st.rerun()

st.caption("Powered by OpenRouter • Click history entries to view full analysis")

