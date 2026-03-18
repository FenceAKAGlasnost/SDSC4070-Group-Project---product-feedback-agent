import streamlit as st
import os
from dotenv import load_dotenv
from langchain_openrouter import ChatOpenRouter

load_dotenv()

os.environ["OPENROUTER_API_KEY"] = st.secrets["OPENROUTER_API_KEY"]

st.set_page_config(page_title="Product Feedback Agent", page_icon="📊", layout="centered")

st.title("📊 Product Feedback Agent System")
st.markdown("**Helping companies improve their products using Large Language Models**")

# ====================== SIDEBAR ======================
st.sidebar.title("⚙️ Analysis Settings")
agent_mode = st.sidebar.selectbox(
    "Choose Analysis Mode",
    [
        "1. Full 5-Agent Analysis (Recommended)",
        "2. Quick Summary Agent",
        "3. Theme & Sentiment Deep Dive",
        "4. Recommendation-Focused Agent",
        "5. Professional Executive Report"
    ]
)

temperature = st.sidebar.slider("Temperature (Creativity)", 0.0, 1.0, 0.65)

# ====================== MAIN AREA ======================
st.subheader("Paste user reviews / comments")

# File uploader
uploaded_file = st.file_uploader("Upload a .txt or .csv file", type=["txt", "csv"])

# Text input - using session state to make buttons work
if "comments" not in st.session_state:
    st.session_state.comments = ""

comments = st.text_area(
    "Enter customer comments here (one per line or paragraph):",
    value=st.session_state.comments,
    height=180,
    key="comments_area"
)

# ====================== SAMPLE BUTTONS ======================
st.caption("Quick Demo Samples:")
col1, col2, col3 = st.columns(3)

if col1.button("📱 Phone App", use_container_width=True):
    st.session_state.comments = "The app keeps crashing on Android. Battery drains way too fast. Beautiful design but too expensive. UI is confusing for new users. Love the new design!"
    st.rerun()

if col2.button("🍔 Food Delivery", use_container_width=True):
    st.session_state.comments = "Food always arrives cold. Delivery is late most of the time. Great variety but prices are too high. Driver was rude."
    st.rerun()

if col3.button("💻 Laptop", use_container_width=True):
    st.session_state.comments = "Screen is very bright. Keyboard feels cheap. Fast performance. Best laptop in this price range but it overheats during gaming."
    st.rerun()

# ====================== ANALYZE BUTTON ======================
if st.button("🚀 Analyze Feedback", type="primary", use_container_width=True):
    final_comments = comments

    if uploaded_file:
        file_content = uploaded_file.getvalue().decode("utf-8")
        final_comments = (final_comments + "\n\n" + file_content) if final_comments else file_content

    if not final_comments or len(final_comments.strip()) < 30:
        st.error("⚠️ Please enter comments or upload a file.")
    else:
        with st.spinner(f"🔍 Running {agent_mode}..."):
            llm = ChatOpenRouter(
                model="meta-llama/llama-3.3-70b-instruct",
                temperature=temperature,
                max_tokens=2000
            )

            prompt = f"""You are an expert Product Feedback Analysis Agent.

Examples of good analysis:
Example 1:
Comments: "App crashes all the time. Battery lasts only 4 hours."
→ Executive Summary: Users love the design but suffer from stability and battery issues.
→ Key Themes: Crashing (Strongly Negative), Battery Life (Negative)
→ Recommendations: Fix memory leaks and optimize battery.

Now analyze these comments:

Comments:
{final_comments}

Mode: {agent_mode}

Respond in this exact structure:
1. **Executive Summary**
2. **Key Themes** (5–8 themes with sentiment)
3. **Detailed Insights**
4. **Prioritized Recommendations for Product v2** (numbered with reasoning)

Use professional, constructive language."""

            response = llm.invoke(prompt)
            st.success("✅ Analysis Complete!")
            st.markdown("### 📋 Final Report")
            st.markdown(response.content)

st.caption("SDSC4070 Large Language Models • Product Feedback Agent System")
