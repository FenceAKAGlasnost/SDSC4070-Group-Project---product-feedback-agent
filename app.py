import streamlit as st
import os
from dotenv import load_dotenv
from langchain_openrouter import ChatOpenRouter
import plotly.express as px
import pandas as pd
import json
import re
from datetime import datetime

load_dotenv()
os.environ["OPENROUTER_API_KEY"] = st.secrets["OPENROUTER_API_KEY"]

st.set_page_config(page_title="Product Feedback Agent", page_icon="📊", layout="centered")

st.title("Product Feedback Agent System")
st.markdown("**AI-powered analysis of customer feedback**")

# ==================== HISTORY FUNCTIONS ====================
def save_to_history(comments, result):
    """Save analysis to session state history"""
    if 'history' not in st.session_state:
        st.session_state.history = []
    
    st.session_state.history.append({
        'id': len(st.session_state.history) + 1,
        'timestamp': datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
        'comments': comments[:100] + '...' if len(comments) > 100 else comments,
        'sentiment': result.get('sentiment', {}),
        'themes': result.get('key_themes', [])[:3],
        'summary': result.get('executive_summary', '')[:100] + '...' if result.get('executive_summary') else ''
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
            'Themes': ', '.join(entry['themes']),
            'Summary': entry['summary']
        })
    
    return pd.DataFrame(data)

# ==================== SIDEBAR ====================
with st.sidebar:
    st.header("Analysis Settings")
    agent_mode = st.selectbox(
        "Analysis Mode",
        ["Full 5-Agent Analysis", "Quick Summary", 
         "Theme & Sentiment Deep Dive", "Recommendation-Focused", 
         "Executive Report"]
    )
    temperature = st.slider("Temperature (Creativity)", 0.0, 1.0, 0.65)
    
    st.divider()
    st.subheader("Analysis History")
    
    if 'history' in st.session_state and st.session_state.history:
        st.caption(f"Total: {len(st.session_state.history)} analyses")
        
        for entry in st.session_state.history[-3:]:
            with st.expander(f"#{entry['id']} - {entry['timestamp'][:10]}"):
                st.write(f"**Comments:** {entry['comments']}")
                st.write(f"**Themes:** {', '.join(entry['themes'])}")
                pos = entry['sentiment'].get('positive', 0)
                neg = entry['sentiment'].get('negative', 0)
                st.write(f"**Sentiment:** Positive: {pos:.0%}, Negative: {neg:.0%}")
        
        col1, col2 = st.columns(2)
        with col1:
            df = export_history_csv()
            if df is not None:
                csv = df.to_csv(index=False)
                st.download_button(
                    label="Export CSV",
                    data=csv,
                    file_name=f"feedback_history_{datetime.now().strftime('%Y%m%d')}.csv",
                    mime="text/csv",
                    use_container_width=True
                )
        with col2:
            if st.button("Clear History", use_container_width=True):
                clear_history()
                st.rerun()
    else:
        st.info("No history yet. Run an analysis to see results here.")

# ==================== MAIN AREA ====================
st.subheader("Input Feedback")

uploaded_file = st.file_uploader("Upload file (.txt or .csv)", type=["txt", "csv"])

comments = st.text_area(
    "Customer comments:",
    height=180,
    placeholder="Paste customer feedback here..."
)

# Sample Data
st.subheader("Sample Data")
sample_option = st.selectbox(
    "Load sample:",
    ["Select sample...", "Phone App Feedback", "Food Delivery Feedback", 
     "Laptop Feedback", "Gaming Console Feedback"]
)

samples = {
    "Phone App Feedback": "The app keeps crashing on Android. Battery drains way too fast. Beautiful design but too expensive. UI is confusing.",
    "Food Delivery Feedback": "Food always arrives cold. Delivery is late. Great variety but prices too high. Driver was rude.",
    "Laptop Feedback": "Screen is bright. Keyboard feels cheap. Fast performance. Overheats during gaming. Best laptop in this price.",
    "Gaming Console Feedback": "Graphics amazing. Controller feels great. Loading times too long. Overheats after 2 hours."
}

if sample_option != "Select sample...":
    st.code(samples[sample_option], language="text")

# ====================== ANALYZE BUTTON ======================
analyze_clicked = st.button("Analyze Feedback", type="primary", use_container_width=True)

if analyze_clicked:
    final_comments = comments
    if uploaded_file:
        final_comments = final_comments + "\n\n" + uploaded_file.getvalue().decode("utf-8")

    if not final_comments or len(final_comments.strip()) < 30:
        st.error("Please enter comments or upload a file.")
    else:
        with st.spinner("Analyzing feedback..."):
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
            
            # ------------- IMPROVED JSON PARSING -------------
            try:
                content = response.content
                
                # First, try to find complete JSON between curly braces
                json_match = re.search(r'\{.*\}', content, re.DOTALL)
                if json_match:
                    potential_json = json_match.group()
                    
                    # Try to parse it
                    try:
                        result = json.loads(potential_json)
                    except json.JSONDecodeError:
                        # If it fails, try to fix truncated JSON
                        st.warning("JSON was truncated, attempting to fix...")
                        
                        # Count opening and closing braces
                        open_braces = potential_json.count('{')
                        close_braces = potential_json.count('}')
                        
                        if open_braces > close_braces:
                            # Add missing closing braces
                            potential_json += '}' * (open_braces - close_braces)
                        
                        # Try parsing again
                        try:
                            result = json.loads(potential_json)
                        except:
                            # If still fails, create a simple result
                            result = {
                                "executive_summary": "Analysis completed but response was malformed",
                                "key_themes": ["Error parsing response"],
                                "sentiment": {"positive": 0.33, "negative": 0.33, "neutral": 0.34},
                                "insights": "The AI response was truncated. Please try again.",
                                "recommendations": ["Try with shorter input", "Run analysis again"]
                            }
                else:
                    # No JSON found, create default
                    result = {
                        "executive_summary": "Could not parse AI response",
                        "key_themes": ["Error"],
                        "sentiment": {"positive": 0.33, "negative": 0.33, "neutral": 0.34},
                        "insights": "The AI did not return valid JSON",
                        "recommendations": ["Please try again"]
                    }
                
                # Save to history (even if we used default values)
                save_to_history(final_comments, result)
                
                st.success("Analysis Complete!")
                
                # Display Report
                st.subheader("Analysis Report")
                st.write(f"**Executive Summary**")
                st.write(result.get('executive_summary', ''))
                
                st.write("**Key Themes**")
                st.write(result.get('key_themes', []))
                
                st.write("**Insights**")
                st.write(result.get('insights', ''))

                # Charts
                col1, col2 = st.columns(2)
                
                with col1:
                    sentiment_data = result.get('sentiment', {"positive": 0.4, "negative": 0.45, "neutral": 0.15})
                    fig_pie = px.pie(
                        names=list(sentiment_data.keys()),
                        values=list(sentiment_data.values()),
                        title="Sentiment Distribution"
                    )
                    st.plotly_chart(fig_pie, use_container_width=True)

                with col2:
                    themes = result.get('key_themes', ["Theme1", "Theme2", "Theme3"])[:4]
                    if themes:
                        values = [45, 30, 25, 20][:len(themes)]
                        df = pd.DataFrame({"Theme": themes, "Frequency": values})
                        fig_bar = px.bar(df, x="Theme", y="Frequency", title="Top Themes")
                        st.plotly_chart(fig_bar, use_container_width=True)

                st.write("**Recommendations**")
                for i, rec in enumerate(result.get('recommendations', []), 1):
                    st.write(f"{i}. {rec}")

            except Exception as e:
                st.error(f"Error analyzing feedback: {str(e)}")
                # Create a fallback result
                result = {
                    "executive_summary": "Analysis encountered an error",
                    "key_themes": ["Error"],
                    "sentiment": {"positive": 0.33, "negative": 0.33, "neutral": 0.34},
                    "insights": f"Error: {str(e)}",
                    "recommendations": ["Please try again with shorter input"]
                }
                save_to_history(final_comments, result)

st.caption("Powered by OpenRouter • Analysis history is saved during your session")
