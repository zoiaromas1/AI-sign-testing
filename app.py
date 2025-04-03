import streamlit as st
import pandas as pd
import numpy as np
from scipy.stats import norm, ttest_rel
import re
from string import ascii_uppercase
import io

# === CONFIG ===
confidence_z_90 = 1.645
confidence_z_80 = 0.84

st.set_page_config(page_title="Significance Testing Tool", layout="wide")
st.title("📊 Significance Testing App")

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

uploaded_file = st.file_uploader("📁 Upload your Excel file", type=["xlsx"])

# === USER INPUT FOR BUCKET CODING ===
use_t1b = st.checkbox("🔘 Use Top 1 Box", value=False)
show_80_confidence = st.checkbox("Show 80% confidence (lowercase letters)", value=True)

if use_t1b:
    t1_value = st.number_input("Enter value for Top 1 Box", min_value=0, max_value=100, value=5)
    bucket_values = [t1_value]
else:
    t2b_1 = st.number_input("Enter first value for Top 2 Box", min_value=0, max_value=100, value=4)
    t2b_2 = st.number_input("Enter second value for Top 2 Box", min_value=0, max_value=100, value=5)
    bucket_values = [t2b_1, t2b_2]

if uploaded_file:
    df = pd.read_excel(uploaded_file)
    df.columns = df.columns.str.strip()

    if 'ID' in df.columns:
        df = df.rename(columns={'ID': 'Respondent'})
    elif test_design != "Independent Samples (default)":
        st.error("❌ Missing required 'ID' column for paired/within-subjects tests.")
        st.stop()

    df['Concept_num'] = df['Concept'].str.extract(r'Concept (\d+)').astype(float)
    df = df.sort_values(by='Concept_num')

    concepts = df['Concept'].dropna().unique()
    breakout_cols = [col for col in df.columns if col.lower().startswith("breakout")]
    attributes = [col for col in df.columns[3:-1] if col not in breakout_cols]

    def calculate_significance(data, concepts, attributes, method="ztest", bucket_values=None):
        result_rows = []
        pivot = None
        if method != "ztest":
            pivot = data.copy()
            for attr in attributes:
                pivot[attr] = pivot[attr].apply(lambda x: 1 if x in bucket_values else 0)
            pivot = pivot.pivot_table(index='Respondent', columns='Concept', values=attributes)

        for attr in attributes:
            row_data = {}
            stats = {}
            for c1 in concepts:
                if method == "ztest":
                    scores = data[data['Concept'] == c1][attr].dropna()
                    base = len(scores)
                    pct = scores.isin(bucket_values).sum() / base * 100 if base > 0 else np.nan
                    stats[c1] = (pct, base)
                else:
                    if c1 in pivot[attr]:
                        stats[c1] = pivot[attr][c1]
                    else:
                        stats[c1] = pd.Series(dtype=float)

            for c1 in concepts:
                better_than = []
                for c2 in concepts:
                    if c1 == c2 or c1 not in stats or c2 not in stats:
                        continue
                    if method == "ztest":
                        p1, n1 = stats[c1]
                        p2, n2 = stats[c2]
                        if n1 == 0 or n2 == 0:
                            continue
                        se = np.sqrt((p1 / 100 * (1 - p1 / 100) / n1) + (p2 / 100 * (1 - p2 / 100) / n2))
                        if se == 0:
                            continue
                        z = (p1 - p2) / se
                        match = re.search(r'Concept (\d+)', c2)
                        if match:
                            letter = ascii_uppercase[int(match.group(1)) - 1]
                            if z > confidence_z_90:
                                better_than.append(letter)
                            elif show_80_confidence and z > confidence_z_80:
                                better_than.append(letter.lower())
                    else:
                        paired = pd.DataFrame({"c1": stats[c1], "c2": stats[c2]}).dropna()
                        if len(paired) > 1:
                            t_stat, p_value = ttest_rel(paired['c1'], paired['c2'])
                            match = re.search(r'Concept (\d+)', c2)
                            if match:
                                letter = ascii_uppercase[int(match.group(1)) - 1]
                                if p_value < 0.1:
                                    better_than.append(letter.lower())
                                if p_value < 0.05:
                                    better_than[-1] = letter

                if method == "ztest":
                    if c1 not in stats or pd.isna(stats[c1][0]):
                        label = "NA"
                    else:
                        val = round(stats[c1][0])
                        label = f"{val}%" + (f" {', '.join(better_than)}" if better_than else "")
                else:
                    scores = stats[c1]
                    if len(scores.dropna()) == 0:
                        label = "NA"
                    else:
                        pct = scores.mean() * 100
                        label = f"{round(pct)}%" + (f" {', '.join(better_than)}" if better_than else "")
                row_data[c1] = label
            result_rows.append(row_data)
        return result_rows

    def build_output_df(data, concepts, attributes, method, bucket_values=None):
        all_rows, attribute_labels, group_labels = [], [], []
        rep_rows = calculate_significance(data, concepts, attributes, method, bucket_values)
        rep_bases = [len(data[data['Concept'] == concept]) for concept in concepts]
        rep_n = int(np.round(np.mean(rep_bases)))
        for attr, row in zip(attributes, rep_rows):
            all_rows.append(row)
            attribute_labels.append(attr)
            group_labels.append(f"REP [n={rep_n}]")

        seen_groups = set()
        for breakout in breakout_cols:
            for group_value in data[breakout].dropna().unique():
                if group_value in seen_groups:
                    continue
                seen_groups.add(group_value)
                group_df = data[data[breakout] == group_value]
                group_rows = calculate_significance(group_df, concepts, attributes, method, bucket_values)
                group_bases = [len(group_df[group_df['Concept'] == concept]) for concept in concepts]
                group_n = int(np.round(np.mean(group_bases)))
                for attr, row in zip(attributes, group_rows):
                    all_rows.append(row)
                    attribute_labels.append(attr)
                    group_labels.append(f"{group_value} [n={group_n}]")

        final_df = pd.DataFrame(all_rows)
        final_df.insert(0, "Group", group_labels)
        final_df.insert(0, "Attribute", attribute_labels)
        concept_labels = {concept: f"{ascii_uppercase[i]}. {concept}" for i, concept in enumerate(concepts)}
        final_df.columns = ["Attribute", "Group"] + [concept_labels[c] for c in concepts]
        return final_df

    method = "ztest" if test_design == "Independent Samples (default)" else "paired"
    df_results = build_output_df(df, concepts, attributes, method, bucket_values)

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
