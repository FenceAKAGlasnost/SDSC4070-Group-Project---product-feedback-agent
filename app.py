import os
import streamlit as st
# Set API key from Streamlit secrets
os.environ["OPENROUTER_API_KEY"] = st.secrets["OPENROUTER_API_KEY"] #et


import streamlit as st
from agents import create_crew

st.set_page_config(page_title="Product Feedback Agent", layout="wide")
st.title("📊 Product Feedback Agent System")
st.markdown("**Helping companies turn user comments into actionable insights for Product v2**")

st.subheader("Paste or upload user reviews/comments")

# Input method
input_method = st.radio("Choose input method:", ["Paste comments", "Upload file (txt or csv)"])

if input_method == "Paste comments":
    user_input = st.text_area("Paste user comments here (one per line or paragraph):", height=300)
else:
    uploaded_file = st.file_uploader("Upload a text or CSV file", type=["txt", "csv"])
    if uploaded_file is not None:
        user_input = uploaded_file.getvalue().decode("utf-8")
    else:
        user_input = ""

if st.button("🚀 Analyze Feedback", type="primary"):
    if not user_input or len(user_input.strip()) < 20:
        st.error("Please provide enough user comments.")
    else:
        with st.spinner("Agents are analyzing the feedback... This may take 20-60 seconds"):
            try:
                crew = create_crew(user_input)
                result = crew.kickoff()
                
                st.success("Analysis Complete!")
                st.markdown("### Final Report")
                st.markdown(result)
                
                # Optional: Show raw agent process
                with st.expander("Show Agent Thinking Process"):
                    st.write("Full agent execution log is shown in terminal for now.")
                    
            except Exception as e:
                st.error(f"Error: {e}")

st.caption("SDSC4070 - Large Language Models | Product Feedback Agent System")
