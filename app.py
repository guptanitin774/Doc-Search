import streamlit as st
import pandas as pd
import openai
import matplotlib.pyplot as plt
from io import BytesIO
from fpdf import FPDF

# Set OpenAI API key (ensure this is set as a secret in Streamlit Cloud)
openai.api_key = st.secrets["openai_api_key"]  # Ensure the key exists in secrets.toml

def analyze_data(df, query):
    """Uses OpenAI to extract insights from the dataset based on user query."""
    prompt = f"""
    You are a data analyst. Given the following dataset:
    {df.head().to_string()}
    
    Answer the following query in a structured format:
    {query}
    """
    
    response = openai.ChatCompletion.create(
        model="gpt-4",
        messages=[{"role": "system", "content": prompt}]
    )
    
    return response["choices"][0]["message"]["content"]

def generate_report(text):
    """Generates a downloadable PDF report."""
    pdf = FPDF()
    pdf.set_auto_page_break(auto=True, margin=15)
    pdf.add_page()
    pdf.set_font("Arial", size=12)
    
    for line in text.split('\n'):
        pdf.cell(200, 10, txt=line, ln=True, align='L')
    
    pdf_output = BytesIO()
    pdf.output(pdf_output, 'F')
    pdf_output.seek(0)
    return pdf_output

# Streamlit UI
st.title("AI-Powered Report Generator")

uploaded_file = st.file_uploader("Upload your CSV file", type=["csv"])

if uploaded_file is not None:
    df = pd.read_csv(uploaded_file)
    st.write("## Data Preview")
    st.dataframe(df.head())
    
    query = st.text_area("Enter your query about the data")
    
    if st.button("Generate Report"):
        with st.spinner("Analyzing data..."):
            report_text = analyze_data(df, query)
            st.subheader("Generated Report")
            st.write(report_text)
            
            pdf_file = generate_report(report_text)
            st.download_button("Download Report as PDF", pdf_file, "AI_Report.pdf", "application/pdf")
    
    st.write("## Data Insights")
    fig, ax = plt.subplots()
    df.hist(ax=ax)
    st.pyplot(fig)
