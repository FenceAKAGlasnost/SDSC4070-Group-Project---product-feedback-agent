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

# Main Area
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

            elif agent_mode == "1. Full 5-Agent Analysis (Recommended)":
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

            response = llm.invoke(prompt)
            
            try:
                data = json.loads(response.content)

                # Text parts
                st.markdown("### Executive Summary")
                st.write(data["executive_summary"])

                st.markdown("### Key Themes")
                for t in data["key_themes"]:
                    st.write(f"• **{t['theme']}** ({t['sentiment']}, ~{t['frequency']} mentions)")

                # Pie chart – Sentiment
                sentiment_df = pd.DataFrame({
                    "Sentiment": list(data["sentiment_distribution"].keys()),
                    "Percentage": list(data["sentiment_distribution"].values())
                })
                fig_pie = px.pie(sentiment_df, values="Percentage", names="Sentiment",
                                 title="Overall Sentiment Distribution",
                                 color_discrete_sequence=["#66c2a5", "#fc8d62", "#8da0cb"])
                st.plotly_chart(fig_pie, use_container_width=True)

                # Bar chart – Top themes by frequency
                theme_df = pd.DataFrame(data["key_themes"])
                if not theme_df.empty:
                    fig_bar = px.bar(theme_df, x="theme", y="frequency",
                                     title="Top Themes by Mention Frequency",
                                     color="sentiment",
                                     color_discrete_map={"positive": "#66c2a5", "negative": "#fc8d62", "neutral": "#8da0cb"})
                    st.plotly_chart(fig_bar, use_container_width=True)

                # Recommendations
                st.markdown("### Recommendations")
                for r in data["recommendations"]:
                    st.write(f"• {r}")

            except Exception as e:
                st.error("Could not parse structured output. Showing raw text instead.")
                st.markdown(response.content)

st.caption("SDSC4070 Large Language Models • Product Feedback Agent System")
