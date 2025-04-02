import streamlit as st
import pandas as pd
import numpy as np
from scipy.stats import norm, ttest_rel, ttest_ind
import re
from string import ascii_uppercase
import io

# === CONFIG ===
confidence_z_90 = 1.645  # for 90% confidence
confidence_z_80 = 0.84   # for 80% confidence

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
    if test_design == "Independent Samples (default)":
        st.markdown("""
        - **Use when:** different people rated each concept (monadic test)
        - **Statistical test:** Z-test for comparing proportions
        - **Ideal for:** T2B, T1B, Net Score comparisons
        """)
    elif test_design == "Paired Samples":
        st.markdown("""
        - **Use when:** same people rated multiple concepts
        - **Statistical test:** Paired t-test (for mean scores)
        - **Ideal for:** within-person comparisons, sequential exposure
        """)
    elif test_design == "Within-Subjects (Repeated Measures)":
        st.markdown("""
        - **Use when:** each person evaluated all concepts multiple times
        - **Statistical test:** Paired t-test or Wilcoxon signed-rank
        - **Ideal for:** longitudinal or psychometric setups
        """)
    elif test_design == "Multiple Groups (3+ concepts)":
        st.markdown("""
        - **Use when:** more than 2 groups/concepts are being compared simultaneously
        - **Statistical test:** ANOVA (means) or Chi-Square (proportions)
        - **Ideal for:** omnibus testing, follow-up with post-hocs
        """)

uploaded_file = st.file_uploader("📁 Upload your Excel file", type=["xlsx"])

# === USER OPTIONS ===
show_80_confidence = st.checkbox("Show 80% confidence (lowercase letters)", value=True)

if uploaded_file:
    st.markdown("---")
    df = pd.read_excel(uploaded_file)
    df.columns = df.columns.str.strip()
    df['Concept_num'] = df['Concept'].str.extract(r'Concept (\d+)').astype(float)
    df = df.sort_values(by='Concept_num')

    concepts = df['Concept'].dropna().unique()
    breakout_cols = [col for col in df.columns if col.lower().startswith("breakout")]
    attributes = [col for col in df.columns[3:-1] if col not in breakout_cols]

    def calculate_ztest_table(data, concepts, attributes, bucket_values):
        result_rows = []
        for attr in attributes:
            attr_result = {}
            for concept in concepts:
                scores = data[data['Concept'] == concept][attr].dropna()
                base = len(scores)
                selected = scores.isin(bucket_values).sum()
                pct = (selected / base) * 100 if base > 0 else np.nan
                attr_result[concept] = {"percent": pct, "base": base}

            row_data = {}
            for c1 in concepts:
                p1 = attr_result[c1]['percent']
                n1 = attr_result[c1]['base']
                better_than = []
                for c2 in concepts:
                    if c1 == c2:
                        continue
                    p2 = attr_result[c2]['percent']
                    n2 = attr_result[c2]['base']
                    if n1 == 0 or n2 == 0:
                        continue
                    p1_p = p1 / 100
                    p2_p = p2 / 100
                    se = np.sqrt((p1_p * (1 - p1_p) / n1) + (p2_p * (1 - p2_p) / n2))
                    if se == 0:
                        continue
                    z = (p1_p - p2_p) / se
                    match = re.search(r'Concept (\d+)', c2)
                    if match:
                        letter = ascii_uppercase[int(match.group(1)) - 1]
                        if z > confidence_z_90:
                            better_than.append(letter)
                        elif show_80_confidence and z > confidence_z_80:
                            better_than.append(letter.lower())

                label = "NA" if pd.isna(p1) else f"{round(p1)}%" + (f" {', '.join(better_than)}" if better_than else "")
                row_data[c1] = label

            result_rows.append(row_data)
        return result_rows

    def calculate_paired_ttest(data, concepts, attributes):
        result_rows = []
        pivot = data.pivot_table(index='Respondent', columns='Concept', values=attributes)

        for attr in attributes:
            row_data = {}
            for c1 in concepts:
                scores1 = pivot[attr][c1]
                label = "NA"
                better_than = []
                for c2 in concepts:
                    if c1 == c2:
                        continue
                    scores2 = pivot[attr][c2]
                    paired = pd.DataFrame({"c1": scores1, "c2": scores2}).dropna()
                    if len(paired) > 1:
                        t_stat, p_value = ttest_rel(paired['c1'], paired['c2'])
                        match = re.search(r'Concept (\d+)', c2)
                        if match:
                            letter = ascii_uppercase[int(match.group(1)) - 1]
                            if p_value < 0.1:
                                better_than.append(letter.lower())
                            if p_value < 0.05:
                                better_than[-1] = letter
                if scores1.count() > 0:
                    label = f"{scores1.mean():.2f}" + (f" {', '.join(better_than)}" if better_than else "")
                row_data[c1] = label
            result_rows.append(row_data)
        return result_rows

    def build_output_df(data, concepts, attributes, method, bucket_values=None):
        all_rows, attribute_labels, group_labels = [], [], []

        if method == "ztest":
            rep_rows = calculate_ztest_table(data, concepts, attributes, bucket_values)
        elif method == "paired":
            rep_rows = calculate_paired_ttest(data, concepts, attributes)

        rep_bases = [len(data[data['Concept'] == concept]) for concept in concepts]
        rep_n = int(np.round(np.mean(rep_bases)))

        for attr, row in zip(attributes, rep_rows):
            all_rows.append(row)
            attribute_labels.append(attr)
            group_labels.append(f"REP [n={rep_n}]")

        final_df = pd.DataFrame(all_rows)
        final_df.insert(0, "Group", group_labels)
        final_df.insert(0, "Attribute", attribute_labels)
        concept_labels = {concept: f"{ascii_uppercase[i]}. {concept}" for i, concept in enumerate(concepts)}
        final_df.columns = ["Attribute", "Group"] + [concept_labels[c] for c in concepts]
        return final_df

    if test_design == "Independent Samples (default)":
        t2b_df = build_output_df(df, concepts, attributes, method="ztest", bucket_values=[4, 5])
        t1b_df = build_output_df(df, concepts, attributes, method="ztest", bucket_values=[5])

        st.success("✅ Significance testing completed!")
        st.write("### 📊 T2B Analysis")
        st.dataframe(t2b_df)

        output = io.BytesIO()
        with pd.ExcelWriter(output, engine='xlsxwriter') as writer:
            t2b_df.to_excel(writer, index=False, sheet_name='T2B Analysis')
            t1b_df.to_excel(writer, index=False, sheet_name='T1B Analysis')
        output.seek(0)

        st.download_button(
            label="📥 Download Excel with T2B + T1B",
            data=output,
            file_name="Significance_Results.xlsx",
            mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet"
        )

    elif test_design == "Paired Samples":
        paired_df = build_output_df(df, concepts, attributes, method="paired")

        st.success("✅ Paired sample mean score comparison complete!")
        st.write("### 📊 Paired t-test Analysis")
        st.dataframe(paired_df)

        output = io.BytesIO()
        with pd.ExcelWriter(output, engine='xlsxwriter') as writer:
            paired_df.to_excel(writer, index=False, sheet_name='Paired Analysis')
        output.seek(0)

        st.download_button(
            label="📥 Download Paired Analysis",
            data=output,
            file_name="Paired_Analysis.xlsx",
            mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet"
        )

    else:
        st.warning("🚧 This test type is not yet implemented beyond Independent and Paired Samples.")
