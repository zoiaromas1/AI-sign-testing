import streamlit as st
import pandas as pd
import numpy as np
from scipy.stats import norm, ttest_rel, ttest_ind
import re
from string import ascii_uppercase
import io

# === CONFIG ===
confidence_z_90 = 1.645  # 90% confidence
confidence_z_80 = 0.84   # 80% confidence
min_effect_size = 5      # Minimum effect size for significance in percentage terms

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
        "Multiple Groups (3+ concepts)": "**Use when:** more than 2 groups. **Test:** Paired t-test between concepts on binary data"
    }
    st.markdown(explanations[test_design])

# === T1B / T2B CONTROLS ===
use_t1b = st.checkbox("🔘 Use Top 1 Box", value=False)
show_80_confidence = st.checkbox("Show 80% confidence (lowercase letters)", value=True)

if use_t1b:
    t1_value = st.number_input("Enter value for Top 1 Box", min_value=0, max_value=100, value=5)
    bucket_values = [t1_value]
else:
    t2b_1 = st.number_input("Enter first value for Top 2 Box", min_value=0, max_value=100, value=4)
    t2b_2 = st.number_input("Enter second value for Top 2 Box", min_value=0, max_value=100, value=5)
    bucket_values = [t2b_1, t2b_2]

# Confidence Level
if show_80_confidence:
    confidence_z = confidence_z_80  # Default 80% confidence (small letters)
else:
    confidence_z = confidence_z_90  # Default 90% confidence (big letters)

# === FILE UPLOAD ===
uploaded_file = st.file_uploader("📁 Upload your Excel file", type=["xlsx"])

if uploaded_file:
    df = pd.read_excel(uploaded_file)
    df.columns = df.columns.str.strip()

    if 'ID' in df.columns:
        df = df.rename(columns={'ID': 'Respondent'})
    elif test_design != "Independent Samples (default)":
        st.error("❌ Missing required 'ID' column for paired/within-subjects tests.")
        st.stop()

    # Ask the user to choose the column to differentiate between groups (e.g., Breakout columns or Concept)
    breakout_columns = [col for col in df.columns if col.lower().startswith("breakout")]
    concept_column = "Concept"  # Always use Concept column as an option
    group_column = st.selectbox("Choose the column to differentiate between groups (Breakout or Concept)", breakout_columns + [concept_column])

    # Sort data by the chosen group column
    df = df.sort_values(by=group_column)

    # Identify breakout columns and attributes
    attributes = [col for col in df.columns if col not in breakout_columns and col != 'Concept' and col != 'Respondent']

    # Exclude dynamically the chosen "group" column from being used in the "Group" column
    def calculate_significance(data, group_column, attributes, method="ztest", bucket_values=None):
        result_rows = []
        pivot = None
        if method != "ztest":
            pivot = data.copy()
            for attr in attributes:
                pivot[attr] = pivot[attr].apply(lambda x: 1 if x in bucket_values else 0)
            pivot = pivot.pivot_table(index='Respondent', columns=group_column, values=attributes)

        for attr in attributes:
            row_data = {}
            stats = {}
            for group in data[group_column].dropna().unique():
                if method == "ztest":
                    scores = data[data[group_column] == group][attr].dropna()
                    base = len(scores)
                    pct = scores.isin(bucket_values).sum() / base * 100 if base > 0 else np.nan
                    stats[group] = (pct, base)
                else:
                    if group in pivot[attr]:
                        stats[group] = pivot[attr][group]
                    else:
                        stats[group] = pd.Series(dtype=float)

            for group in data[group_column].dropna().unique():
                better_than = []
                for compare_group in data[group_column].dropna().unique():
                    if group == compare_group or group not in stats or compare_group not in stats:
                        continue
                    if method == "ztest":
                        p1, n1 = stats[group]
                        p2, n2 = stats[compare_group]
                        if n1 == 0 or n2 == 0:
                            continue
                        se = np.sqrt((p1 / 100 * (1 - p1 / 100) / n1) + (p2 / 100 * (1 - p2 / 100) / n2))
                        if se == 0:
                            continue
                        z = (p1 - p2) / se
                        
                        # Apply minimum effect size threshold
                        if abs(p1 - p2) < min_effect_size:
                            continue

                        letter = ascii_uppercase[data[group_column].dropna().unique().tolist().index(compare_group)]
                        if z > confidence_z_90:
                            better_than.append(letter)
                        elif show_80_confidence and z > confidence_z_80:
                            better_than.append(letter.lower())
                    else:
                        paired = pd.DataFrame({"c1": stats[group], "c2": stats[compare_group]}).dropna()
                        if len(paired) > 1:
                            t_stat, p_value = ttest_rel(paired['c1'], paired['c2'])
                            letter = ascii_uppercase[data[group_column].dropna().unique().tolist().index(compare_group)]
                            if p_value < 0.1:
                                better_than.append(letter.lower())
                            if p_value < 0.05:
                                better_than[-1] = letter

                label = f"{round(stats[group][0])}%" + (f" {', '.join(better_than)}" if better_than else "")
                row_data[group] = label
            result_rows.append(row_data)
        return result_rows

    def build_output_df(data, group_column, attributes, method, bucket_values=None):
        all_rows, attribute_labels, group_labels = [], [], []
        
        # Calculate the total number of respondents across all concepts (sum of respondents)
        total_respondents = data['Respondent'].nunique()

        rep_rows = calculate_significance(data, group_column, attributes, method, bucket_values)
        rep_bases = [len(data[data[group_column] == group]) for group in data[group_column].dropna().unique()]
        rep_n = total_respondents  # Total number of respondents across all concepts

        for attr, row in zip(attributes, rep_rows):
            all_rows.append(row)
            attribute_labels.append(attr)
            group_labels.append(f"REP [n={rep_n}]")

        seen_groups = set()
        for breakout in breakout_columns:
            for group_value in data[breakout].dropna().unique():
                if group_value in seen_groups:
                    continue
                seen_groups.add(group_value)
                group_df = data[data[breakout] == group_value]
                group_rows = calculate_significance(group_df, group_column, attributes, method, bucket_values)
                group_bases = [len(group_df[group_df[group_column] == group]) for group in data[group_column].dropna().unique()]
                group_n = int(np.round(np.mean(group_bases)))
                for attr, row in zip(attributes, group_rows):
                    all_rows.append(row)
                    attribute_labels.append(attr)
                    group_labels.append(f"{group_value} [n={group_n}]")

        final_df = pd.DataFrame(all_rows)
        final_df.insert(0, "Group", group_labels)
        final_df.insert(0, "Attribute", attribute_labels)
        group_labels_map = {group: f"{ascii_uppercase[i]}. {group}" for i, group in enumerate(data[group_column].dropna().unique())}
        final_df.columns = ["Attribute", "Group"] + [group_labels_map[g] for g in data[group_column].dropna().unique()]
        return final_df

    # Replace None with 0%
    df.fillna("0%", inplace=True)

    # Exclude the chosen group column dynamically from the attributes column
    if group_column in df.columns:
        attributes = [col for col in df.columns if col != group_column and col != "Concept" and col != "Respondent"]

    method = "ztest" if test_design == "Independent Samples (default)" else "paired"
    df_results = build_output_df(df, group_column, attributes, method, bucket_values)

    st.success("✅ Analysis complete!")
    st.dataframe(df_results)

    output = io.BytesIO()
    with pd.ExcelWriter(output, engine='xlsxwriter') as writer:
        sheet_name = 'T1B Analysis' if use_t1b else 'T2B Analysis'
        df_results.to_excel(writer, index=False, sheet_name=sheet_name)
    output.seek(0)

    st.download_button(
        label=f"📥 Download {sheet_name} Excel",
        data=output,
        file_name=f"Significance_{sheet_name.replace(' ', '_')}.xlsx",
        mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet"
    )
