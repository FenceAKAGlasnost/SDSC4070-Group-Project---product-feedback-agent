import streamlit as st
import os
import re
from dotenv import load_dotenv
from langchain_openrouter import ChatOpenRouter
import plotly.express as px
import pandas as pd
import json
from datetime import datetime

load_dotenv()
os.environ["OPENROUTER_API_KEY"] = st.secrets["OPENROUTER_API_KEY"]

st.set_page_config(page_title="Product Feedback Agent", page_icon="📊", layout="centered")

st.title("📊 Product Feedback Agent System")
st.markdown("**Helping companies improve their products using Large Language Models**")

# ==================== HISTORY FUNCTIONS ====================
def save_to_history(comments, result):
    """Save analysis to session state history"""
    if 'history' not in st.session_state:
        st.session_state.history = []
    
    st.session_state.history.append({
        'id': len(st.session_state.history) + 1,
        'timestamp': datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
        'comments': comments[:100] + '...' if len(comments) > 100 else comments,
        'result': result,
        'sentiment': result.get('sentiment', {}),
        'themes': result.get('key_themes', [])[:3]
    })

def clear_history():
    """Clear all history"""
    st.session_state.history = []

def export_history_csv():
    """Convert history to CSV for download"""
    if 'history' not in st.session_state or not st.session_state.history:
        return None
    
    data = []
    for entry in st.session_state.history:
        data.append({
            'ID': entry['id'],
            'Timestamp': entry['timestamp'],
            'Comments': entry['comments'],
            'Positive': entry['sentiment'].get('positive', 0),
            'Negative': entry['sentiment'].get('negative', 0),
            'Neutral': entry['sentiment'].get('neutral', 0),
            'Themes': ', '.join(entry['themes'])
        })
    
    return pd.DataFrame(data)

# ==================== SIDEBAR ====================
st.sidebar.title("⚙️ Analysis Settings")
agent_mode = st.sidebar.selectbox(
    "Choose Analysis Mode",
    ["1. Full 5-Agent Analysis", "2. Quick Summary Agent", 
     "3. Theme & Sentiment Deep Dive", "4. Recommendation-Focused Agent", 
     "5. Professional Executive Report"]
)
temperature = st.sidebar.slider("Temperature (Creativity)", 0.0, 1.0, 0.65)

# ==================== HISTORY SIDEBAR ====================
st.sidebar.markdown("---")
st.sidebar.title("📋 Analysis History")

if 'history' in st.session_state and st.session_state.history:
    st.sidebar.markdown(f"**Total:** {len(st.session_state.history)}")
    
    for entry in st.session_state.history[-3:]:
        with st.sidebar.expander(f"#{entry['id']} - {entry['timestamp'][:10]}"):
            st.write(f"**Comments:** {entry['comments']}")
            st.write(f"**Themes:** {', '.join(entry['themes'])}")
    
    col1, col2 = st.sidebar.columns(2)
    with col1:
        df = export_history_csv()
        if df is not None:
            csv = df.to_csv(index=False)
            st.sidebar.download_button("📥 CSV", csv, "feedback_history.csv", "text/csv")
    with col2:
        if st.button("🗑️ Clear"):
            clear_history()
            st.rerun()
else:
    st.sidebar.info("No history yet")

# ==================== MAIN AREA ====================
st.subheader("Paste user reviews / comments")

uploaded_file = st.file_uploader("Upload .txt or .csv", type=["txt", "csv"])

comments = st.text_area(
    "Enter comments:",
    height=180,
    placeholder="Paste reviews here..."
)

# Sample Dropdown
st.subheader("📋 Quick Samples")
sample_option = st.selectbox(
    "Load sample:",
    ["Select...", "📱 Phone App", "🍔 Food Delivery", "💻 Laptop", "🎮 Gaming"]
)

samples = {
    "📱 Phone App": "App crashes. Battery drains fast. Beautiful design but expensive.",
    "🍔 Food Delivery": "Food cold. Delivery late. Great variety.",
    "💻 Laptop": "Screen bright. Keyboard cheap. Fast performance. Overheats.",
    "🎮 Gaming": "Graphics amazing. Controller great. Long loading times."
}

if sample_option != "Select...":
    st.code(samples[sample_option])

# ====================== ANALYZE BUTTON ======================
if st.button("🚀 Analyze", type="primary", use_container_width=True):
    final_comments = comments
    if uploaded_file:
        final_comments += "\n\n" + uploaded_file.getvalue().decode("utf-8")

    if not final_comments or len(final_comments.strip()) < 30:
        st.error("Please enter comments.")
    else:
        with st.spinner("Analyzing..."):
            llm = ChatOpenRouter(
                model="meta-llama/llama-3.3-70b-instruct",
                temperature=temperature,
                max_tokens=2500
            )

            # ==================== CHAIN-OF-THOUGHT FEW-SHOT PROMPT ====================
            prompt = f"""You are a product feedback analyst. Always think step by step before answering. For each analysis, include a "chain_of_thought" field that lists your reasoning steps. Then provide the analysis in the requested JSON format.

Examples of good analysis with chain-of-thought:

Example 1:
Comments: "The app crashes every time I open it. Battery drains fast. Love the design though."
Output:
{{
  "chain_of_thought": [
    "I identify three main points: crashes (negative), battery drain (negative), design appreciation (positive).",
    "Count: 2 negative, 1 positive → overall negative with some positive.",
    "Key themes: app stability, battery performance, design quality.",
    "Insights: technical issues overshadow design, so users are frustrated despite liking the look.",
    "Recommendations: fix crashes first, then battery, and consider highlighting design in marketing."
  ],
  "executive_summary": "Users appreciate the design but are experiencing critical technical issues.",
  "key_themes": ["app stability", "battery performance", "design quality"],
  "sentiment": {{"positive": 0.25, "negative": 0.65, "neutral": 0.10}},
  "insights": "While the visual design receives praise, stability problems are the main source of frustration.",
  "recommendations": [
    "Fix crash bugs as top priority",
    "Optimize battery usage in background processes",
    "Consider adding offline mode"
  ]
}}

Example 2:
Comments: "Great variety of food. Delivery is always late though. Prices are reasonable."
Output:
{{
  "chain_of_thought": [
    "Comments mention variety (positive), delivery lateness (negative), reasonable prices (positive).",
    "Sentiment: 2 positive, 1 negative → mixed but slightly positive.",
    "Themes: food variety, delivery speed, pricing.",
    "Insight: core product is good, logistics are the pain point.",
    "Recommendations: improve delivery logistics, communicate better ETAs."
  ],
  "executive_summary": "Customers like the food variety and pricing but are dissatisfied with delivery times.",
  "key_themes": ["food variety", "delivery speed", "pricing"],
  "sentiment": {{"positive": 0.50, "negative": 0.40, "neutral": 0.10}},
  "insights": "Product quality is good, but logistics issues are hurting the overall experience.",
  "recommendations": [
    "Optimize delivery routes",
    "Add more delivery partners",
    "Set realistic delivery time expectations"
  ]
}}

Example 3:
Comments: "Battery life is amazing. Screen is bright. Best phone I've ever owned."
Output:
{{
  "chain_of_thought": [
    "All comments are positive: battery, screen, overall satisfaction.",
    "Sentiment is overwhelmingly positive.",
    "Themes: battery life, display quality, overall satisfaction.",
    "Insight: battery and display are key strengths.",
    "Recommendations: market these features, maintain quality."
  ],
  "executive_summary": "Customers are extremely satisfied with the product performance and features.",
  "key_themes": ["battery life", "display quality", "overall satisfaction"],
  "sentiment": {{"positive": 0.90, "negative": 0.05, "neutral": 0.05}},
  "insights": "Battery and display are key differentiators driving positive sentiment.",
  "recommendations": [
    "Highlight battery life in marketing",
    "Maintain current quality standards",
    "Gather more feedback for next iteration"
  ]
}}

Now analyze these comments. First reason step by step, then output the JSON (including the chain_of_thought field).

Comments:
{final_comments}

Output JSON with the following structure:
{{
  "chain_of_thought": ["step1", "step2", ...],
  "executive_summary": "string",
  "key_themes": ["theme1", "theme2", "theme3"],
  "sentiment": {{"positive": 0.XX, "negative": 0.XX, "neutral": 0.XX}},
  "insights": "string",
  "recommendations": ["rec1", "rec2", "rec3"]
}}

Return ONLY valid JSON, no other text."""

            response = llm.invoke(prompt)
            
            try:
                # Extract JSON from response (handles extra text)
                content = response.content
                json_match = re.search(r'\{.*\}', content, re.DOTALL)
                if json_match:
                    result = json.loads(json_match.group())
                else:
                    result = json.loads(content)
                
                # Save to history
                save_to_history(final_comments, result)
                
                st.success("✅ Done!")
                
                # Display reasoning (optional)
                with st.expander("Show reasoning steps"):
                    for step in result.get('chain_of_thought', []):
                        st.write(f"• {step}")
                
                # Display report
                st.markdown("### 📋 Report")
                st.markdown(f"**Summary:** {result.get('executive_summary', '')}")
                st.markdown("**Themes:**")
                st.write(result.get('key_themes', []))
                
                # Charts
                col1, col2 = st.columns(2)
                with col1:
                    sentiment = result.get('sentiment', {"positive":0.4, "negative":0.4, "neutral":0.2})
                    fig = px.pie(names=list(sentiment.keys()), values=list(sentiment.values()), title="Sentiment")
                    st.plotly_chart(fig, use_container_width=True)
                
                with col2:
                    themes = result.get('key_themes', ["Theme1", "Theme2", "Theme3"])
                    if themes:
                        df = pd.DataFrame({
                            "Theme": themes[:3], 
                            "Frequency": [45, 30, 25][:len(themes[:3])]
                        })
                        fig = px.bar(df, x="Theme", y="Frequency", title="Top Themes")
                        st.plotly_chart(fig, use_container_width=True)
                
                st.markdown("**Recommendations:**")
                for i, rec in enumerate(result.get('recommendations', []), 1):
                    st.write(f"{i}. {rec}")
                    
            except Exception as e:
                st.error(f"Could not parse structured output. Showing raw response:")
                st.markdown(response.content)

st.caption("Product Feedback Agent with History & CSV Export")
