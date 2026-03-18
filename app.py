import streamlit as st
import os
from dotenv import load_dotenv
from langchain_openrouter import ChatOpenRouter

load_dotenv()

# ====================== CONFIG ======================
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

# ====================== MAIN INTERFACE ======================
st.subheader("Paste user reviews / comments")

# File uploader
uploaded_file = st.file_uploader("Upload a .txt or .csv file", type=["txt", "csv"])

# Text input area
comments = st.text_area(
    "Enter customer comments here (one per line or paragraph):",
    height=220,
    placeholder="The app keeps crashing on Android...\nBeautiful design but too expensive..."
)

# Sample buttons for easy demo
st.caption("Quick Demo Samples:")
col1, col2, col3 = st.columns(3)
if col1.button("📱 Phone App"):
    comments = "Battery drains way too fast. The UI is beautiful but confusing. App crashes randomly. Best phone I've ever had except for the battery."
if col2.button("🍔 Food Delivery"):
    comments = "Food always arrives cold. Delivery is late most of the time. Great variety but prices are too high. Driver was very rude."
if col3.button("💻 Laptop"):
    comments = "Screen is very bright. Keyboard feels cheap. Fast performance. Best laptop in this price range. Overheats during gaming."

# ====================== ANALYZE BUTTON ======================
if st.button("🚀 Analyze Feedback", type="primary", use_container_width=True):
    if not comments and not uploaded_file:
        st.error("⚠️ Please enter comments or upload a file.")
    else:
        # Combine uploaded file with manual input
        if uploaded_file:
            file_content = uploaded_file.getvalue().decode("utf-8")
            comments = comments + "\n\n" + file_content if comments else file_content

        with st.spinner(f"🔍 Running {agent_mode}..."):
            llm = ChatOpenRouter(
                model="meta-llama/llama-3.3-70b-instruct",
                temperature=temperature,
                max_tokens=2000
            )

            # ==================== IMPROVED PROMPT (Few-Shot) ====================
            prompt = f"""You are an expert Product Feedback Analysis Agent. 
Your analysis is always professional, balanced, constructive, and actionable.

Here are high-quality examples of excellent analysis:

**Example 1:**
User Comments: "App crashes all the time. Battery lasts only 4 hours. Design is beautiful."
Analysis:
1. Executive Summary: Users are impressed by the design but frustrated with stability and battery performance.
2. Key Themes: Stability/Crashing (Strongly Negative), Battery Life (Negative), Design (Positive)
3. Detailed Insights: Crashing appears to be the most frequent complaint.
4. Recommendations for v2: 
   - Priority 1: Fix memory leaks causing crashes
   - Priority 2: Optimize battery consumption

**Example 2:**
User Comments: "Too expensive. Delivery slow. Food cold."
Analysis:
1. Executive Summary: Pricing and delivery reliability are major pain points affecting satisfaction.
2. Key Themes: Price (Negative), Delivery Speed (Negative), Food Quality (Negative)
3. Recommendations: Introduce value bundles and improve delivery partner standards.

Now analyze the following real customer comments:

User Comments:
{comments}

Mode: {agent_mode}

Please structure your response **exactly** in this format:

1. **Executive Summary**
2. **Key Themes** (list 5–8 themes with sentiment)
3. **Detailed Insights**
4. **Prioritized Recommendations for Product v2** (numbered, with clear reasoning)

Use professional yet friendly language. Be specific and constructive."""

            response = llm.invoke(prompt)
            
            st.success("✅ Analysis Complete!")
            st.markdown("### 📋 Final Report")
            st.markdown(response.content)

st.caption("SDSC4070 Large Language Models • Product Feedback Agent System")
