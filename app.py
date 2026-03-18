import streamlit as st
import os
from dotenv import load_dotenv
from langchain_openrouter import ChatOpenRouter

load_dotenv()

st.set_page_config(page_title="Product Feedback Agent", page_icon="📊", layout="centered")

st.title("📊 Product Feedback Agent System")
st.markdown("**Helping companies improve their products using Large Language Models**")

st.subheader("Paste user reviews / comments")

comments = st.text_area(
    "Enter customer comments here (one per line or paragraph):",
    height=250,
    placeholder="The app keeps crashing...\nLove the new design but it's slow...\nBattery drains too fast..."
)

if st.button("Analyze Feedback", type="primary"):
    if not comments or len(comments.strip()) < 30:
        st.error("Please enter enough comments to analyze.")
    else:
        with st.spinner("Analyzing feedback with LLM agent..."):
            try:
                llm = ChatOpenRouter(
                    model="meta-llama/llama-3.3-70b-instruct",
                    temperature=0.7,
                    api_key=os.getenv("OPENROUTER_API_KEY")
                )

                prompt = f"""You are an expert product feedback analysis agent.
Analyze the following user comments and provide a professional report.

User Comments:
{comments}

Please structure your response with these sections:
1. **Executive Summary** (2-3 sentences)
2. **Key Themes** (list top 5-7 themes with sentiment)
3. **Detailed Insights**
4. **Actionable Recommendations for Product v2** (prioritized)

Be constructive, positive, and specific."""

                response = llm.invoke(prompt)
                st.success("Analysis Complete!")
                st.markdown(response.content)

            except Exception as e:
                st.error(f"Error: {str(e)}")
                st.info("Make sure your OPENROUTER_API_KEY is added in Streamlit Secrets.")

st.caption("SDSC4070 Large Language Models | Product Feedback Agent System")
