import streamlit as st 
import pandas as pd
import numpy as np
from scipy.stats import norm
import re
from string import ascii_uppercase
import io

# === CONFIG ===
confidence_z_90 = 1.645  # for 90% confidence
confidence_z_80 = 0.84   # for 80% confidence

st.set_page_config(page_title="Significance Testing Tool")
st.title("📊 Significance Testing App")

uploaded_file = st.file_uploader("Upload your Excel file", type=["xlsx"])

# === USER OPTIONS ===
show_80_confidence = st.checkbox("Show 80% confidence (lowercase letters)", value=True)

if uploaded_file:
    df = pd.read_excel(uploaded_file)
    df.columns = df.columns.str.strip()
    df['Concept_num'] = df['Concept'].str.extract(r'Concept (\d+)').astype(float)
    df = df.sort_values(by='Concept_num')

    concepts = df['Concept'].dropna().unique()
    breakout_cols = [col for col in df.columns if col.lower().startswith("breakout")]
    attributes = [col for col in df.columns[3:-1] if col not in breakout_cols]

    def calculate_table(data, concepts, attributes, bucket_values, show_80_confidence):
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

                if pd.isna(p1):
                    label = "NA"
                else:
                    label = f"{round(p1)}%" + (f" {', '.join(better_than)}" if better_than else "")
                row_data[c1] = label

            result_rows.append(row_data)
        return result_rows

    def build_output_df(df, concepts, attributes, bucket_values, show_80_confidence):
        all_rows = []
        attribute_labels = []
        group_labels = []

        rep_rows = calculate_table(df, concepts, attributes, bucket_values, show_80_confidence)
        rep_bases = [len(df[df['Concept'] == concept]) for concept in concepts]
        rep_n = int(np.round(np.mean(rep_bases)))

        for attr, row in zip(attributes, rep_rows):
            all_rows.append(row)
            attribute_labels.append(attr)
            group_labels.append(f"REP [n={rep_n}]")

        seen_groups = set()
        for breakout in breakout_cols:
            for group_value in df[breakout].dropna().unique():
                if group_value in seen_groups:
                    continue
                seen_groups.add(group_value)

                group_df = df[df[breakout] == group_value]
                group_bases = [len(group_df[group_df['Concept'] == concept]) for concept in concepts]
                group_n = int(np.round(np.mean(group_bases)))

                group_rows = calculate_table(group_df, concepts, attributes, bucket_values, show_80_confidence)
                for attr, row in zip(attributes, group_rows):
                    all_rows.append(row)
                    attribute_labels.append(attr)
                    group_labels.append(f"{group_value} [n={group_n}]")

        final_df = pd.DataFrame(all_rows)
        final_df.insert(0, "Group", group_labels)
        final_df.insert(0, "Attribute", attribute_labels)

        concept_labels = {concept: f"{ascii_uppercase[i]}. {concept}" for i, concept in enumerate(concepts)}
        renamed_columns = ["Attribute", "Group"] + [concept_labels[c] for c in concepts]
        final_df.columns = renamed_columns
        return final_df

    # === Build Both Sheets
    t2b_df = build_output_df(df, concepts, attributes, [4, 5], show_80_confidence)
    t1b_df = build_output_df(df, concepts, attributes, [5], show_80_confidence)

    st.success("✅ Significance testing completed!")
    st.write("### 📊 T1B&T2B Analysis")
    st.dataframe(t2b_df)

    # === DOWNLOAD BUTTON
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
