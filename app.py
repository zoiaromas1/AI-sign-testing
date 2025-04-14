import streamlit as st
import pandas as pd
import numpy as np
from scipy.stats import norm, ttest_rel
import re
from string import ascii_uppercase
import io

# === CONFIG ===
confidence_z_90 = 1.645  # Z-value for 90% confidence
confidence_z_80 = 0.84   # Z-value for 80% confidence

# Streamlit page config
st.set_page_config(page_title="Significance Testing Tool", layout="wide")
st.title("📊 Significance Testing App")

# === INSTRUCTIONS & UI CHANGES ===
st.subheader("Step-by-Step Manual")
st.markdown("""
1. **Step 1: Download the template and fill it in**  
   Download the Excel template and fill in your data with the appropriate columns. If not concept testing, leave the 'Concept' column consistent.

2. **Step 2: Upload your filled-in file**  
   Once the template is completed, upload the file for analysis.

3. **Step 3: View the results**  
   After the file is uploaded, the analysis results will be shown and available for download.
""")

# === TEMPLATE DOWNLOAD ===
@st.cache_data
def generate_template():
    output = io.BytesIO()
    with pd.ExcelWriter(output, engine='xlsxwriter') as writer:
        df_template = pd.DataFrame({
            "ID": ["001", "002"],
            "Concept": ["Concept 1", "Concept 2"],  # User can leave this the same if not concept testing
            "Breakout 1": ["Male", "Female"],
            "Breakout 2": ["18-34", "35-54"],
            "Attribute 1": [4, 5],
            "Attribute 2": [3, 4]
        })
        df_template.to_excel(writer, index=False, sheet_name="Template")
    output.seek(0)
    return output

st.download_button(
    label="📥 Download Excel Template",
    data=generate_template(),
    file_name="analysis_template.xlsx",
    mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet"
)

# === TEST DESIGN SELECTION ===
test_design = st.selectbox(
    "Choose your test design:",
    options=[
        "Independent Samples (default)",
        "Paired Samples",
        "Within-Subjects (Repeated Measures)",
        "Multiple Groups (3+ concepts)"
    ]
)

with st.expander("ℹ️ What does this mean?"):
    explanations = {
        "Independent Samples (default)": "**Use when:** different people rated each concept. **Test:** Z-test for comparing Top 2 Box %",
        "Paired Samples": "**Use when:** same people rated multiple concepts. **Test:** Paired t-test on binary T2B scores (1 = Top 2 Box)",
        "Within-Subjects (Repeated Measures)": "**Use when:** repeated evaluation by same person. **Test:** Paired t-test on binary scores",
        "Multiple Groups (3+ concepts)": "**Use when:** more than 2 groups. **
