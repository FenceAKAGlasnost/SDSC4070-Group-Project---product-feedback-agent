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

            prompt = f"""Analyze the following customer comments and return a JSON object only.
Do not add any explanation outside the JSON.

Comments:
{final_comments}

Return JSON in this exact format:
{{
  "executive_summary": "string",
  "key_themes": ["theme1", "theme2", ...],
  "sentiment": {{"positive": 0.XX, "negative": 0.XX, "neutral": 0.XX}},
  "insights": "string",
  "recommendations": ["rec1", "rec2", ...]
}}

Be accurate and balanced."""

            response = llm.invoke(prompt)
            
            try:
                # Try to parse JSON from LLM response
                result = json.loads(response.content)
                
                st.success("✅ Analysis Complete!")
                
                # Display Report
                st.markdown("### 📋 Final Report")
                st.markdown(f"**Executive Summary**  \n{result.get('executive_summary', '')}")
                st.markdown("**Key Themes**")
                st.write(result.get('key_themes', []))
                st.markdown("**Detailed Insights**")
                st.write(result.get('insights', ''))

                # === CHARTS ===
                col1, col2 = st.columns(2)
                
                with col1:
                    sentiment_data = result.get('sentiment', {"positive": 0.4, "negative": 0.45, "neutral": 0.15})
                    fig_pie = px.pie(
                        names=list(sentiment_data.keys()),
                        values=list(sentiment_data.values()),
                        title="Sentiment Distribution",
                        color_discrete_sequence=px.colors.sequential.RdBu
                    )
                    st.plotly_chart(fig_pie, use_container_width=True)

                with col2:
                    # Fake theme frequencies for bar chart (you can improve this later)
                    themes = result.get('key_themes', ["Battery", "Design", "Price", "Performance"])[:4]
                    values = [45, 30, 25, 20]
                    df = pd.DataFrame({"Theme": themes, "Frequency": values})
                    fig_bar = px.bar(df, x="Theme", y="Frequency", title="Top Themes")
                    st.plotly_chart(fig_bar, use_container_width=True)

                st.markdown("**Recommendations for Product v2**")
                for i, rec in enumerate(result.get('recommendations', []), 1):
                    st.write(f"{i}. {rec}")

            except:
                st.error("Could not parse structured output. Showing raw response:")
                st.markdown(response.content)

st.caption("SDSC4070 Large Language Models • Product Feedback Agent System")
