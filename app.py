import streamlit as st
import os
from dotenv import load_dotenv
from langchain_openrouter import ChatOpenRouter
import plotly.express as px
import pandas as pd
import json
from datetime import datetime

# Import the actual multi-agent pipeline
from agent import create_crew

load_dotenv()
os.environ["OPENROUTER_API_KEY"] = st.secrets["OPENROUTER_API_KEY"]

st.set_page_config(page_title="Product Feedback Agent System", page_icon="📊", layout="centered")  # page_icon still emoji; if you want to remove it too, change to None or a text icon

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

def save_to_history(comments_preview, comments_full, executive_summary, key_themes, sentiment, insights, recommendations):
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
        "key_themes": key_themes,
        "sentiment": sentiment,
        "insights": insights,
        "recommendations": recommendations
    })

def display_analysis(result):
    """Display a single analysis result (used in history view)."""
    st.markdown("### Executive Summary")
    st.write(result.get("executive_summary", "No summary"))
    st.markdown("### Key Themes")
    for theme in result.get("key_themes", []):
        st.write(f"• **{theme.get('theme', 'Unknown')}** ({theme.get('sentiment', 'N/A')}, ~{theme.get('frequency', '?')} mentions)")
    st.markdown("### Recommendations")
    for rec in result.get("recommendations", []):
        st.write(f"• {rec}")

# ==================== TOP TABS ====================
def export_history_csv():
    if 'history' not in st.session_state or not st.session_state.history:
        return None
    
    # Flatten the history entries into a simple DataFrame
    history_data = []
    for entry in st.session_state.history:
        history_data.append({
            "ID": entry.get('id', ''),
            "Timestamp": entry.get('timestamp', ''),
            "Comments Preview": entry.get('comments_preview', ''),
            "Full Comments": entry.get('comments_full', ''),
            "Executive Summary": entry.get('executive_summary', ''),
            "Key Themes": ", ".join(entry.get('key_themes', [])),
            "Positive %": entry.get('sentiment', {}).get('positive', 0),
            "Negative %": entry.get('sentiment', {}).get('negative', 0),
            "Neutral %": entry.get('sentiment', {}).get('neutral', 0),
            "Insights": entry.get('insights', ''),
            "Recommendations": "\n".join(entry.get('recommendations', []))
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
    # Use final_comments for all operations
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

            # Five clearly different prompt branches
            if agent_mode == "1. Full 5-Agent Analysis (Recommended)":
                try:
                    # Create and run the crew (use final_comments)
                    crew = create_crew(final_comments)
                    result = crew.kickoff()   # returns final markdown report

                    st.success("Multi-Agent Analysis Complete!")

                    # Display the full report (markdown)
                    st.markdown(result)

                    # Parse the report to extract structured data for history & charts
                    report_text = result
                    exec_summary = ""
                    key_themes_raw = []
                    recommendations = []

                    lines = report_text.splitlines()
                    current = None
                    for line in lines:
                        if "## 1. Executive Summary" in line:
                            current = "exec"
                            continue
                        elif "## 2. Key Themes" in line:
                            current = "themes"
                            continue
                        elif "## 3. Detailed Summary" in line:
                            current = "summary"
                            continue
                        elif "## 4. Actionable Recommendations" in line:
                            current = "recs"
                            continue

                        if current == "exec" and line.strip():
                            exec_summary += line.strip() + " "
                        elif current == "recs":
                            if line.strip().startswith(("1.", "2.", "3.", "4.", "5.", "6.", "7.", "8.", "9.")):
                                recommendations.append(line.strip())
                        elif current == "themes" and line.strip().startswith("**Theme:**"):
                            key_themes_raw.append(line.strip())

                    # Convert key themes into a list of dicts (for display and charts)
                    key_themes_list = []
                    for t in key_themes_raw:
                        # Extract theme name: "**Theme:** UI/UX"
                        parts = t.split("**Theme:**")
                        if len(parts) > 1:
                            theme_name = parts[1].split("**")[0].strip()
                            key_themes_list.append({"theme": theme_name, "sentiment": "mixed", "frequency": 0})

                    # For sentiment distribution, we'll approximate or leave empty (no chart)
                    sentiment_dist = {"positive": 0.33, "negative": 0.33, "neutral": 0.34}

                    # Save to history
                    save_to_history(
                        comments_preview=final_comments[:100],
                        comments_full=final_comments,
                        executive_summary=exec_summary,
                        key_themes=[t["theme"] for t in key_themes_list],
                        sentiment=sentiment_dist,
                        insights="",
                        recommendations=recommendations
                    )

                except Exception as e:
                    st.error(f"Error running multi-agent pipeline: {e}")

            else:
                # All other modes use a single LLM call (with final_comments)
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

                # Invoke LLM
                try:
                    response = llm.invoke(prompt)
                    st.success("Analysis Complete!")
                    st.markdown(response.content)

                    # For modes 2-5, we still want to save some data to history (simplified)
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
                        recommendations = [line.strip() for line in response.content.splitlines() if line.strip().startswith(("1.", "2."))]
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
                        recommendations=recommendations
                    )

                except Exception as e:
                    st.error(f"API call failed: {e}")

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
            pos = entry['sentiment'].get('positive', 0)
            # Using plain text sentiment icons (or you could just use text)
            sentiment_text = "Positive" if pos > 0.6 else "Mixed" if pos > 0.3 else "Negative"
            
            history_data.append({
                "ID": entry['id'],
                "Date": entry['timestamp'][:10],
                "Comments": entry['comments_preview'],
                "Themes": ", ".join(entry['key_themes'][:2]),
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
            entry = next((e for e in st.session_state.history if e['id'] == st.session_state.viewing_entry), None)
            if entry:
                st.markdown("---")
                st.subheader(f"Full Analysis #{entry['id']}")
                st.caption(f"Analyzed on: {entry['timestamp']}")
                with st.expander("View Full Comments"):
                    st.write(entry['comments_full'])
                # Display the analysis using the helper function
                display_analysis(entry)
    else:
        st.info("No history yet. Run an analysis in the 'New Analysis' tab!")

    if st.button("Clear All History"):
        clear_history()
        st.rerun()

st.caption("Powered by OpenRouter • Click history entries to view full analysis")
