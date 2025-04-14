import streamlit as st
import pandas as pd
import numpy as np
from scipy.stats import norm, ttest_rel
import re
from string import ascii_uppercase
import io

# === CONFIG ===
confidence_z_90 = 1.645  # For 90% confidence
confidence_z_80 = 0.84   # For 80% confidence

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

# === FILE UPLOAD ===
uploaded_file = st.file_uploader("📁 Upload your Excel file", type=["xlsx"])

if uploaded_file:
    df = pd.read_excel(uploaded_file)
    df.columns = df.columns.str.strip()

    if 'ID' in df.columns:
        df = df.rename(columns={'ID': 'Respondent'})
    else:
        st.error("❌ Missing required 'ID' column.")
        st.stop()

    # User selects breakout or concept for analysis
    breakout_columns = [col for col in df.columns if col.lower().startswith("breakout")]
    concept_column = "Concept"
    group_column = st.selectbox("Choose the column to differentiate between groups (Breakout or Concept)", breakout_columns + [concept_column])

    # Sort data by group
    df = df.sort_values(by=group_column)

    # Identify attributes
    attributes = [col for col in df.columns if col not in breakout_columns and col != 'Concept' and col != 'Respondent']

    # Function to calculate standard error (SE) and Z-score
    def calculate_significance(data, group_column, attributes, bucket_values=None):
        result_rows = []
        for attr in attributes:
            row_data = {}
            for group in data[group_column].dropna().unique():
                scores = data[data[group_column] == group][attr].dropna()
                n = len(scores)
                pct = scores.isin(bucket_values).sum() / n * 100 if n > 0 else np.nan
                # Standard Error calculation
                p_hat = pct / 100  # Convert percentage to proportion
                se = np.sqrt((p_hat * (1 - p_hat)) / n)

                # Calculate Z-score for comparing proportions
                for compare_group in data[group_column].dropna().unique():
                    if group == compare_group:
                        continue
                    compare_scores = data[data[group_column] == compare_group][attr].dropna()
                    compare_pct = compare_scores.isin(bucket_values).sum() / len(compare_scores) * 100 if len(compare_scores) > 0 else np.nan
                    compare_p_hat = compare_pct / 100
                    compare_se = np.sqrt((compare_p_hat * (1 - compare_p_hat)) / len(compare_scores))

                    # Pooled SE for the comparison
                    pooled_se = np.sqrt(se**2 + compare_se**2)

                    # Z-score calculation
                    z = (p_hat - compare_p_hat) / pooled_se

                    # Adjust based on confidence level (90% or 80%)
                    critical_value = confidence_z_90 if show_80_confidence else confidence_z_80

                    if z > critical_value:
                        better_than = ascii_uppercase[data[group_column].dropna().unique().tolist().index(compare_group)]
                        row_data[group] = f"{round(pct)}% {better_than}"

            result_rows.append(row_data)
        return result_rows

    # Build the output dataframe with results
    def build_output_df(data, group_column, attributes, bucket_values=None):
        all_rows, attribute_labels, group_labels = [], [], []
        rep_rows = calculate_significance(data, group_column, attributes, bucket_values)
        for attr, row in zip(attributes, rep_rows):
            all_rows.append(row)
            attribute_labels.append(attr)
        final_df = pd.DataFrame(all_rows)
        final_df.insert(0, "Attribute", attribute_labels)
        final_df.columns = ["Attribute"] + [f"Group {i+1}" for i in range(len(data[group_column].dropna().unique()))]
        return final_df

    # Run analysis
    bucket_values = [t2b_1, t2b_2]
    df_results = build_output_df(df, group_column, attributes, bucket_values)

    st.success("✅ Analysis complete!")
    st.dataframe(df_results)

    output = io.BytesIO()
    with pd.ExcelWriter(output, engine='xlsxwriter') as writer:
        df_results.to_excel(writer, index=False)
    output.seek(0)

    st.download_button(
        label="📥 Download Results",
        data=output,
        file_name="Significance_Results.xlsx",
        mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet"
    )
