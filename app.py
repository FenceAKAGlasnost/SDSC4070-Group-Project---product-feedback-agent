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
        'comments_full': comments,
        'comments_preview': comments[:100] + '...' if len(comments) > 100 else comments,
        'executive_summary': result.get('executive_summary', ''),
        'key_themes': result.get('key_themes', []),
        'sentiment': result.get('sentiment', {}),
        'insights': result.get('insights', ''),
        'recommendations': result.get('recommendations', [])
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
            'Comments': entry['comments_preview'],
            'Positive': entry['sentiment'].get('positive', 0),
            'Negative': entry['sentiment'].get('negative', 0),
            'Neutral': entry['sentiment'].get('neutral', 0),
            'Themes': ', '.join(entry['key_themes'][:3]),
            'Summary': entry['executive_summary'][:100] + '...' if entry['executive_summary'] else ''
        })
    
    return pd.DataFrame(data)

def display_analysis(result):
    """Display analysis results with graphs"""
    st.subheader("Analysis Report")
    
    st.write("**Executive Summary**")
    st.write(result.get('executive_summary', ''))
    
    st.write("**Key Themes**")
    st.write(result.get('key_themes', []))
    
    st.write("**Insights**")
    st.write(result.get('insights', ''))

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

# ==================== SETTINGS SIDEBAR ====================
with st.sidebar:
    st.header("⚙️ Settings")
    agent_mode = st.selectbox(
        "Analysis Mode",
        ["Full 5-Agent Analysis (Recommended)", "Quick Summary Agent", "Theme & Sentiment Deep Dive", "Recommendation-Focused Agent", "Professional Executive Report"]
    )
    temperature = st.slider("Temperature", 0.0, 1.0, 0.65)

# ==================== TOP TABS ====================
tab1, tab2 = st.tabs(["📝 New Analysis", "📋 History"])

# ==================== TAB 1: NEW ANALYSIS ====================
with tab1:
    st.subheader("Input Feedback")
    
    uploaded_file = st.file_uploader("Upload file (.txt or .csv)", type=["txt", "csv"])
    
    comments = st.text_area(
        "Customer comments:",
        height=180,
        placeholder="Paste customer feedback here..."
    )
    
    # Sample Data
    with st.expander("📋 Load Sample Data"):
        sample_option = st.selectbox(
            "Choose a sample:",
            ["Select...", "Phone App", "Food Delivery", "Laptop", "Gaming Console"]
        )
        
        samples = {
            "Phone App": "The app keeps crashing on Android. Battery drains fast. Beautiful design but too expensive.",
            "Food Delivery": "Food always arrives cold. Delivery is late. Great variety but prices too high.",
            "Laptop": "Screen is bright. Keyboard feels cheap. Fast performance. Overheats during gaming.",
            "Gaming Console": "Graphics amazing. Controller great. Loading times too long. Overheats after 2 hours."
        }
        
        if sample_option != "Select...":
            st.code(samples[sample_option])
            if st.button("Use this sample"):
                comments = samples[sample_option]
                st.rerun()
    
    # Analyze Button
    analyze_clicked = st.button("🚀 Analyze Feedback", type="primary", use_container_width=True)
    
    if analyze_clicked:
        final_comments = comments
        if uploaded_file:
            final_comments += "\n\n" + uploaded_file.getvalue().decode("utf-8")
    
        if not final_comments or len(final_comments.strip()) < 30:
            st.error("Please enter comments or upload a file.")
        else:
            with st.spinner("Analyzing feedback..."):
                llm = ChatOpenRouter(
                    model="meta-llama/llama-3.3-70b-instruct",
                    temperature=temperature,
                    max_tokens=2500
                )
    
                # ────────────────────────────────────────────────
#  Five clearly different prompt branches
# ────────────────────────────────────────────────

if agent_mode == "1. Full 5-Agent Analysis (Recommended)":
    prompt = f"""You are simulating a full multi-agent pipeline for product feedback analysis.

Step 1 - Cleaning Agent: Remove duplicates, spam, short/irrelevant entries from these comments:
{comments}

Step 2 - Analysis Agent: From the cleaned comments, extract main themes and sentiment (positive/negative/neutral).

Step 3 - Summary Agent: Write a balanced 4-6 sentence executive summary.

Step 4 - Recommendation Agent: Provide 6-8 specific, prioritized recommendations for v2.

Step 5 - Report Agent: Compile everything into a clean markdown report.

Output the final report only, with clear section headings."""

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
 content = response.content
 json_match = re.search(r'\{.*\}', content, re.DOTALL)
    if json_match:
     result = json.loads(json_match.group())
    else:
     result = json.loads(content)
                    
            # Save to history
            save_to_history(final_comments, result)
                    
            st.success("✅ Analysis Complete!")
            display_analysis(result)
                    
    except Exception as e:
     st.error(f"Error parsing response")
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
