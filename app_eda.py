# ============================================================
# Survey EDA Dashboard (Streamlit)
# + Excel-style XTab tables (All Questions) with:
#   - Streamlit preview as heatmap-like styled tables
#   - Excel export with conditional formatting heatmaps + sig letters
# ============================================================

import pandas as pd
import numpy as np

import streamlit as st
import matplotlib.pyplot as plt
import plotly.express as px

from scipy.stats import chi2_contingency
from scipy.stats import spearmanr
from io import BytesIO

from openpyxl import Workbook
from openpyxl.styles import PatternFill, Alignment
from openpyxl.utils.dataframe import dataframe_to_rows
from openpyxl.utils import get_column_letter
from openpyxl.formatting.rule import ColorScaleRule

import plotly.graph_objects as go
import re
from collections import Counter
from scipy.stats import fisher_exact, mannwhitneyu, kruskal


# -----------------------------
# Config
# -----------------------------
st.set_page_config(page_title="Survey EDA Dashboard", layout="wide")

DEFAULT_PATH = "usuarios_filtrados_clean.csv"  # change if needed

REGION_COL = "Which region best represents the primary jurisdiction of your regulatory agency?"
AI_USE_COL = "Have you personally used AI in your professional role as a regulator?"
TRAINING_COL = "Have you or your agency participated in any formal training or workshops related to AI in the last two years?"
POSITION_COL = "Which best describes your position level within your regulatory agency?"
YEARS_COL = "How many years have you worked in a regulatory role with oversight of the gambling/gaming industry?"

ISO3_COL_CANDIDATES = ["iso3", "ISO3", "ISO_3", "country_iso3"]
COUNTRY_NAME_CANDIDATES = ["Country", "country", "Primary Country", "Jurisdiction Country"]

ROLE_GROUP_COL = "role_group"
TENURE_GROUP_COL = "tenure_group"
REGION_GROUP_COL = "region_group"  # e.g., US vs Other
TRAINING_BIN_COL = "training_bin"  # Yes vs No

GREEN = PatternFill(start_color="C6EFCE", end_color="C6EFCE", fill_type="solid")   # p<0.05
YELLOW = PatternFill(start_color="FFEB9C", end_color="FFEB9C", fill_type="solid") # p<0.10


# -----------------------------
# Utilities
# -----------------------------
def clean_cat_series(s: pd.Series) -> pd.Series:
    s = s.replace({np.nan: "Unknown"}).astype(str).str.strip()
    return s.replace({"": "Unknown", "nan": "Unknown", "None": "Unknown"})


def strip_strings(df: pd.DataFrame) -> pd.DataFrame:
    df2 = df.copy()
    obj_cols = df2.select_dtypes(include=["object"]).columns
    for c in obj_cols:
        df2[c] = (
            df2[c]
            .astype(str)
            .str.strip()
            .replace({"": np.nan, "nan": np.nan, "None": np.nan})
        )
    return df2


def coerce_numeric(df: pd.DataFrame, cols: list[str]) -> pd.DataFrame:
    df2 = df.copy()
    for c in cols:
        if c in df2.columns:
            df2[c] = pd.to_numeric(df2[c], errors="coerce")
    return df2


def is_likert_1_5(series: pd.Series, min_valid: float = 0.70) -> bool:
    s = series.dropna()
    if s.empty:
        return False
    numeric = pd.to_numeric(s, errors="coerce")
    ok = numeric.between(1, 5).mean()
    return ok >= min_valid


def detect_likert_cols(df: pd.DataFrame) -> list[str]:
    likert = []
    for c in df.columns:
        if df[c].dtype == "object" or "int" in str(df[c].dtype) or "float" in str(df[c].dtype):
            if is_likert_1_5(df[c]):
                likert.append(c)
    return likert


def robust_barh(labels: pd.Series, values: pd.Series, title: str, xlabel: str):
    x_labels = [str(x) for x in labels.tolist()]
    y_values = values.astype(int).tolist()

    order = np.argsort(y_values)
    x_labels = [x_labels[i] for i in order]
    y_values = [y_values[i] for i in order]

    fig = plt.figure(figsize=(10, 5))
    plt.barh(x_labels, y_values)
    plt.xlabel(xlabel)
    plt.title(title)
    plt.tight_layout()
    st.pyplot(fig)


def select_filter(colname: str, df: pd.DataFrame):
    if colname not in df.columns:
        return None
    vals = df[colname].replace({np.nan: "Unknown"}).astype(str)
    options = sorted(vals.unique().tolist())
    selected = st.sidebar.multiselect(colname, options, default=options)
    return selected


@st.cache_data
def load_data(path: str) -> pd.DataFrame:
    df = pd.read_csv(path)
    df = strip_strings(df)

    df = coerce_numeric(df, [
        "Duration (in seconds)", "Progress", "Q_RecaptchaScore", "Score",
        "Q_UnansweredPercentage", "Q_UnansweredQuestions"
    ])

    if REGION_COL in df.columns:
        df[REGION_COL] = (
            df[REGION_COL]
            .replace({np.nan: "Unknown"})
            .astype(str)
            .str.strip()
            .replace({"Asia-Paciifc": "Asia-Pacific", "nan": "Unknown", "": "Unknown"})
        )

    return df


# -----------------------------
# Grouping helpers
# -----------------------------
def recode_role(series: pd.Series) -> pd.Series:
    s = clean_cat_series(series)
    leadership = {"Executive Leadership", "Director", "Commissioner", "Chair", "CEO", "President"}
    management = {"Management", "Manager", "Supervisor", "Team Lead"}
    non_mgmt = {"Non-Management", "Analyst", "Specialist", "Staff", "Individual Contributor"}

    def _map(x: str) -> str:
        if x in leadership: return "Executive Leadership"
        if x in management: return "Management"
        if x in non_mgmt: return "Non-Management"
        return "Other/Unknown"
    return s.map(_map)


def recode_training_yesno(series: pd.Series) -> pd.Series:
    s = clean_cat_series(series).str.lower()
    return s.map(lambda x: "Yes" if "yes" in x else ("No" if "no" in x else "Unknown"))


def recode_region_us_other(series: pd.Series) -> pd.Series:
    s = clean_cat_series(series).str.lower()
    return s.map(lambda x: "US" if ("united states" in x or x == "us" or "u.s." in x) else ("Other" if x != "unknown" else "Unknown"))


def recode_tenure_5yrs(series: pd.Series) -> pd.Series:
    s = clean_cat_series(series).str.lower()
    less = {"0-2 years", "0–2 years", "1-2 years", "3-5 years", "3–5 years", "less than 5 years", "<5 years"}
    more = {"6-10 years", "6–10 years", "10+ years", "10 years or more", "more than 5 years", ">5 years"}
    def _map(x: str) -> str:
        if x in less: return "<5 years"
        if x in more: return ">=5 years"
        return "Other/Unknown"
    return s.map(_map)


def add_group_vars(df: pd.DataFrame) -> pd.DataFrame:
    df2 = df.copy()
    if POSITION_COL in df2.columns:
        df2[ROLE_GROUP_COL] = recode_role(df2[POSITION_COL])
    else:
        df2[ROLE_GROUP_COL] = "Unknown"

    if TRAINING_COL in df2.columns:
        df2[TRAINING_BIN_COL] = recode_training_yesno(df2[TRAINING_COL])
    else:
        df2[TRAINING_BIN_COL] = "Unknown"

    if REGION_COL in df2.columns:
        df2[REGION_GROUP_COL] = recode_region_us_other(df2[REGION_COL])
    else:
        df2[REGION_GROUP_COL] = "Unknown"

    if YEARS_COL in df2.columns:
        df2[TENURE_GROUP_COL] = recode_tenure_5yrs(df2[YEARS_COL])
    else:
        df2[TENURE_GROUP_COL] = "Unknown"

    return df2


# -----------------------------
# XTab helpers (counts + %)
# -----------------------------
def xtab_percent_table(
    df: pd.DataFrame,
    row_var: str,
    col_var: str,
    drop_unknown_rows: bool = False,
    drop_unknown_cols: bool = False,
    sort_rows_by_total: bool = False,
) -> pd.DataFrame:
    tmp = df[[row_var, col_var]].copy()
    tmp[row_var] = clean_cat_series(tmp[row_var])
    tmp[col_var] = clean_cat_series(tmp[col_var])

    if drop_unknown_rows:
        tmp = tmp[tmp[row_var] != "Unknown"]
    if drop_unknown_cols:
        tmp = tmp[tmp[col_var] != "Unknown"]

    counts = pd.crosstab(tmp[row_var], tmp[col_var], dropna=False)
    counts["Total"] = counts.sum(axis=1)

    col_sums = counts.sum(axis=0).replace(0, np.nan)
    pct = (counts / col_sums) * 100
    pct = pct.round(1)

    if sort_rows_by_total:
        pct = pct.sort_values("Total", ascending=False)

    return pct


def xtab_count_table(
    df: pd.DataFrame,
    row_var: str,
    col_var: str,
    drop_unknown_rows: bool = False,
    drop_unknown_cols: bool = False,
    sort_rows_by_total: bool = False,
) -> pd.DataFrame:
    tmp = df[[row_var, col_var]].copy()
    tmp[row_var] = clean_cat_series(tmp[row_var])
    tmp[col_var] = clean_cat_series(tmp[col_var])

    if drop_unknown_rows:
        tmp = tmp[tmp[row_var] != "Unknown"]
    if drop_unknown_cols:
        tmp = tmp[tmp[col_var] != "Unknown"]

    counts = pd.crosstab(tmp[row_var], tmp[col_var], dropna=False)
    counts["Total"] = counts.sum(axis=1)

    if sort_rows_by_total:
        counts = counts.sort_values("Total", ascending=False)

    return counts


# -----------------------------
# Significance letters (a/b/c and A/B/C)
# - simple z-test on proportions within each banner column
# -----------------------------
def _ztest_prop(p1, n1, p2, n2):
    if n1 == 0 or n2 == 0:
        return np.nan
    x1 = p1 * n1
    x2 = p2 * n2
    p = (x1 + x2) / (n1 + n2)
    se = np.sqrt(p * (1 - p) * (1/n1 + 1/n2))
    if se == 0:
        return np.nan
    z = (p1 - p2) / se
    # two-sided p-value approx
    from math import erf, sqrt
    pval = 2 * (1 - 0.5*(1+erf(abs(z)/sqrt(2))))
    return pval


def sig_letter_matrix(df: pd.DataFrame, row_var: str, col_var: str, alpha95=0.05, alpha99=0.01, drop_unknown=True) -> pd.DataFrame:
    tmp = df[[row_var, col_var]].copy()
    tmp[row_var] = clean_cat_series(tmp[row_var])
    tmp[col_var] = clean_cat_series(tmp[col_var])
    if drop_unknown:
        tmp = tmp[(tmp[row_var] != "Unknown") & (tmp[col_var] != "Unknown")]

    ct = pd.crosstab(tmp[row_var], tmp[col_var], dropna=False)
    if ct.empty:
        return pd.DataFrame()

    col_totals = ct.sum(axis=0).replace(0, np.nan)
    pct = ct.div(col_totals, axis=1)

    cols = list(ct.columns)
    letters_95 = list("abcdefghijklmnopqrstuvwxyz")
    letters_99 = list("ABCDEFGHIJKLMNOPQRSTUVWXYZ")

    out = pd.DataFrame("", index=ct.index, columns=ct.columns, dtype=object)

    for c_idx, c in enumerate(cols):
        for r in ct.index:
            n1 = ct.loc[r, c]
            if pd.isna(col_totals[c]) or col_totals[c] == 0:
                continue
            p1 = pct.loc[r, c]

            sig95 = []
            sig99 = []
            for c2_idx, c2 in enumerate(cols):
                if c2 == c:
                    continue
                n2 = ct.loc[r, c2]
                if pd.isna(col_totals[c2]) or col_totals[c2] == 0:
                    continue
                p2 = pct.loc[r, c2]
                pval = _ztest_prop(p1, col_totals[c], p2, col_totals[c2])
                if pd.isna(pval):
                    continue
                if pval < alpha99:
                    sig99.append(letters_99[c2_idx % len(letters_99)])
                elif pval < alpha95:
                    sig95.append(letters_95[c2_idx % len(letters_95)])

            out.loc[r, c] = "".join(sig99 + sig95)

    return out


# -----------------------------
# Excel writer: XTab All Qs
# - Adds conditional formatting heatmap per question block
# -----------------------------
def build_xtab_allqs_excel(
    df: pd.DataFrame,
    banners: list[str],
    questions: list[str],
    drop_unknown: bool = True,
    percent: bool = True,
    show_sig: bool = True,
    alpha95: float = 0.05,
    alpha99: float = 0.01
) -> bytes:
    wb = Workbook()
    ws = wb.active
    ws.title = "XTab_AllQs"

    ws.append(["XTab All Questions (Column %)"])
    ws.append(["Notes:", "Each banner column sums to 100%. Heatmap shows higher % in greener shades."])
    ws.append([])

    header_fill = PatternFill(start_color="D9E1F2", end_color="D9E1F2", fill_type="solid")

    for q in questions:
        first_val_row = None
        last_val_row = None
        ws.append([q])
        ws.cell(row=ws.max_row, column=1).fill = header_fill

        for banner in banners:
            ws.append([None, f"Banner: {banner}"])
            ws.cell(row=ws.max_row, column=2).fill = header_fill

            pct_tbl = xtab_percent_table(
                df=df,
                row_var=q,
                col_var=banner,
                drop_unknown_rows=drop_unknown,
                drop_unknown_cols=drop_unknown,
                sort_rows_by_total=True,
            )

            cnt_tbl = xtab_count_table(
                df=df,
                row_var=q,
                col_var=banner,
                drop_unknown_rows=drop_unknown,
                drop_unknown_cols=drop_unknown,
                sort_rows_by_total=True,
            )

            banner_levels = [c for c in pct_tbl.columns.tolist()]  # includes Total
            ws.append([None, None] + banner_levels)

            r_header = ws.max_row
            for col in range(1, ws.max_column + 1):
                cell = ws.cell(row=r_header, column=col)
                cell.fill = header_fill
                cell.alignment = Alignment(vertical="top", wrap_text=True)

            data_start = 3

            sig_tbl = None
            if show_sig:
                sig_tbl = sig_letter_matrix(df, row_var=q, col_var=banner, alpha95=alpha95, alpha99=alpha99, drop_unknown=drop_unknown)
                # Ensure same col order
                if not sig_tbl.empty:
                    sig_tbl = sig_tbl.reindex(columns=[c for c in cnt_tbl.columns if c in sig_tbl.columns], fill_value="")

            options = pct_tbl.index.tolist()

            for opt in options:
                ws.append([None, opt])
                r_val = ws.max_row
                if first_val_row is None:
                    first_val_row = r_val
                last_val_row = r_val

                for j, lvl in enumerate(banner_levels):
                    val = pct_tbl.loc[opt, lvl]
                    cell = ws.cell(row=r_val, column=data_start + j)
                    if pd.isna(val):
                        cell.value = None
                    else:
                        cell.value = float(val)
                        if percent:
                            cell.number_format = "0.0\"%\""
                        else:
                            cell.number_format = "0.0"
                    cell.alignment = Alignment(vertical="top", wrap_text=True)

                # significance row (letters)
                if show_sig and sig_tbl is not None and (not sig_tbl.empty) and (opt in sig_tbl.index):
                    ws.append([None, "sig"])
                    r_sig = ws.max_row
                    for j, lvl in enumerate(banner_levels):
                        if lvl in sig_tbl.columns:
                            ws.cell(row=r_sig, column=data_start + j).value = str(sig_tbl.loc[opt, lvl])
                            ws.cell(row=r_sig, column=data_start + j).alignment = Alignment(vertical="top", wrap_text=True)

            # heatmap-style conditional formatting (Excel) for this question block
            if first_val_row is not None and last_val_row is not None and last_val_row >= first_val_row:
                start_data_col = data_start  # first numeric col for this question
                end_data_col = data_start + len(banner_levels) - 1  # includes Total
                rng = f"{get_column_letter(start_data_col)}{first_val_row}:{get_column_letter(end_data_col)}{last_val_row}"
                rule = ColorScaleRule(
                    start_type="min", start_color="FFFFFF",
                    mid_type="percentile", mid_value=50, mid_color="FFEB9C",
                    end_type="max", end_color="C6EFCE"
                )
                ws.conditional_formatting.add(rng, rule)

            ws.append([])

        ws.append([])  # separator between questions

    # widths
    for col in range(1, ws.max_column + 1):
        ws.column_dimensions[get_column_letter(col)].width = 14
    ws.column_dimensions["A"].width = 4
    ws.column_dimensions["B"].width = 52

    bio = BytesIO()
    wb.save(bio)
    return bio.getvalue()


# -----------------------------
# App UI
# -----------------------------
st.title("Survey EDA Dashboard (Interactive)")

st.sidebar.header("Data Source")
uploaded = st.sidebar.file_uploader("Upload CSV (required)", type=["csv"])

if uploaded is None:
    st.warning("⬅️ Please upload a CSV using the sidebar to start the dashboard.")
    st.stop()

df = pd.read_csv(uploaded)
df = strip_strings(df)

st.sidebar.header("Filters")
sel_region = select_filter(REGION_COL, df)
sel_ai = select_filter(AI_USE_COL, df)
sel_training = select_filter(TRAINING_COL, df)
sel_position = select_filter(POSITION_COL, df)
sel_years = select_filter(YEARS_COL, df)

df_f = df.copy()
for col, sel in [
    (REGION_COL, sel_region),
    (AI_USE_COL, sel_ai),
    (TRAINING_COL, sel_training),
    (POSITION_COL, sel_position),
    (YEARS_COL, sel_years),
]:
    if sel is not None and col in df_f.columns:
        df_f = df_f[df_f[col].replace({np.nan: "Unknown"}).astype(str).isin(sel)]

df_f = add_group_vars(df_f)

st.caption(f"Rows after filters: {len(df_f)} / {len(df)}")

with st.expander("Preview data"):
    st.dataframe(df_f.head(50), use_container_width=True)


# -----------------------------
# Tabs
# -----------------------------
tab1, tab2, tab3, tab4, tab5, tab6 = st.tabs([
    "Univariate", "Bivariate", "Correlations", "Geo / Map",
    "XTab All Qs (Excel-style)", "Visual XTab Matrix"
])

# -----------------------------
# Tab 1: Univariate
# -----------------------------
with tab1:
    st.subheader("Univariate distributions")

    num_cols = df_f.select_dtypes(include=["number"]).columns.tolist()
    if len(num_cols) == 0:
        st.info("No numeric columns detected.")
    else:
        col = st.selectbox(
            "Numeric variable (histogram)",
            num_cols,
            index=num_cols.index("Score") if "Score" in num_cols else 0
        )
        bins = st.slider("Bins", 5, 60, 20, key="bins_univ")

        fig = plt.figure(figsize=(8, 4))
        x = df_f[col].dropna()
        plt.hist(x, bins=bins)
        plt.title(f"Histogram: {col}")
        plt.xlabel(col)
        plt.ylabel("Count")
        plt.tight_layout()
        st.pyplot(fig)

        fig2 = plt.figure(figsize=(6, 2.5))
        plt.boxplot(x, vert=False)
        plt.title(f"Boxplot: {col}")
        plt.tight_layout()
        st.pyplot(fig2)

    st.subheader("Categorical frequency")
    cat_cols = df_f.select_dtypes(include=["object"]).columns.tolist()
    if len(cat_cols) == 0:
        st.info("No categorical columns detected.")
    else:
        cat = st.selectbox(
            "Categorical variable",
            cat_cols,
            index=cat_cols.index(REGION_COL) if REGION_COL in cat_cols else 0
        )
        vc = df_f[cat].replace({np.nan: "Unknown"}).astype(str).value_counts(dropna=False).reset_index()
        vc.columns = ["Category", "Count"]
        st.dataframe(vc, use_container_width=True)
        robust_barh(vc["Category"], vc["Count"], title=f"Counts: {cat}", xlabel="Count")


# -----------------------------
# Tab 2: Bivariate
# -----------------------------
with tab2:
    st.subheader("Bivariate exploration")

    num_cols = df_f.select_dtypes(include=["number"]).columns.tolist()
    cat_cols = df_f.select_dtypes(include=["object"]).columns.tolist()

    if len(num_cols) == 0 or len(cat_cols) == 0:
        st.info("Need at least one numeric and one categorical column.")
    else:
        y = st.selectbox(
            "Numeric Y",
            num_cols,
            index=num_cols.index("Score") if "Score" in num_cols else 0
        )
        x = st.selectbox(
            "Categorical X",
            cat_cols,
            index=cat_cols.index(REGION_COL) if REGION_COL in cat_cols else 0
        )

        tmp = df_f[[x, y]].copy()
        tmp[x] = tmp[x].replace({np.nan: "Unknown"}).astype(str)
        tmp[y] = pd.to_numeric(tmp[y], errors="coerce")
        tmp = tmp.dropna(subset=[y])

        top_k = st.slider("Max categories to show (top-k)", 3, 25, 12, key="topk_biv")
        topcats = tmp[x].value_counts().head(top_k).index
        tmp2 = tmp[tmp[x].isin(topcats)]

        fig = plt.figure(figsize=(10, 4))
        cats_sorted = tmp2[x].value_counts().index.tolist()
        data = [tmp2[tmp2[x] == c][y].values for c in cats_sorted]
        plt.boxplot(data, labels=cats_sorted, vert=True)
        plt.xticks(rotation=30, ha="right")
        plt.title(f"{y} by {x} (top {top_k})")
        plt.tight_layout()
        st.pyplot(fig)


# -----------------------------
# Tab 3: Correlations
# -----------------------------
with tab3:
    st.subheader("Correlation Matrix (Spearman) + Export")

    likert_cols = detect_likert_cols(df_f)
    st.write(f"Detected Likert/ordinal (1–5) columns: {len(likert_cols)}")

    min_nonnull = st.slider("Min non-null pairs required", 5, 50, 10, key="minnonnull_corr")
    cols_use = []
    for c in likert_cols:
        if pd.to_numeric(df_f[c], errors="coerce").notna().sum() >= min_nonnull:
            cols_use.append(c)

    if len(cols_use) < 2:
        st.info("Not enough Likert/ordinal columns with sufficient data.")
    else:
        # simple spearman matrix
        X = df_f[cols_use].apply(pd.to_numeric, errors="coerce")
        r_mat = X.corr(method="spearman")

        st.dataframe(r_mat.round(3), use_container_width=True)


# -----------------------------
# Tab 4: Geo / Map
# -----------------------------
with tab4:
    st.subheader("Geographic view (no pycountry)")

    iso3_col = next((c for c in ISO3_COL_CANDIDATES if c in df_f.columns), None)

    if iso3_col:
        tmp = df_f[[iso3_col]].copy()
        tmp[iso3_col] = tmp[iso3_col].replace({np.nan: "Unknown"}).astype(str).str.strip()
        tmp = tmp[tmp[iso3_col].str.len() == 3]

        counts = tmp[iso3_col].value_counts().reset_index()
        counts.columns = ["iso3", "respondents"]

        fig = px.choropleth(
            counts,
            locations="iso3",
            color="respondents",
            title="Respondents by Country (ISO3)",
        )
        st.plotly_chart(fig, use_container_width=True)
    else:
        st.info("No ISO3 column detected. Add ISO3 codes to enable a world map.")


# -----------------------------
# Tab 5: XTab All Qs (Excel-style)
# -----------------------------
with tab5:
    st.subheader("Excel-style XTab (All Questions) — Preview + Export")

    cat_cols = df_f.select_dtypes(include=["object"]).columns.tolist()
    exclude = {ROLE_GROUP_COL, TENURE_GROUP_COL, REGION_GROUP_COL, TRAINING_BIN_COL}
    cat_cols = [c for c in cat_cols if c not in exclude]

    if len(cat_cols) < 2:
        st.info("Need at least 2 categorical columns.")
    else:
        default_banners = [c for c in [REGION_COL, POSITION_COL, AI_USE_COL, TRAINING_COL, YEARS_COL] if c in cat_cols]
        banners = st.multiselect("Banners (grouping variables)", options=cat_cols,
                                 default=default_banners[:3] if default_banners else cat_cols[:2],
                                 key="xtab_allqs_banners")

        questions_all = [c for c in cat_cols if c not in set(banners) and c not in exclude]

        st.markdown("### Preview (Excel-style table)")
        preview_q = st.selectbox("Question to preview", options=questions_all, key="xtab_allqs_preview_q")
        preview_banner = st.selectbox("Banner to preview", options=banners, key="xtab_allqs_preview_banner")

        drop_unknown = st.checkbox("Drop Unknown", value=True, key="drop_unknown_xtab_allqs")
        show_sig = st.checkbox("Show sig letters (a/b/c, A/B/C)", value=True, key="xtab_allqs_show_sig")
        alpha95 = st.slider("Alpha (95% letters)", 0.001, 0.10, 0.05, step=0.001, key="xtab_allqs_alpha95")
        alpha99 = st.slider("Alpha (99% letters)", 0.001, 0.10, 0.01, step=0.001, key="xtab_allqs_alpha99")

        if preview_q and preview_banner:
            _pct = xtab_percent_table(
                df_f,
                row_var=preview_q,
                col_var=preview_banner,
                drop_unknown_rows=drop_unknown,
                drop_unknown_cols=drop_unknown,
                sort_rows_by_total=True,
            )
            st.dataframe(
                _pct.style.format("{:.1f}%").background_gradient(axis=None),
                use_container_width=True
            )
            if show_sig:
                _sig = sig_letter_matrix(df_f, row_var=preview_q, col_var=preview_banner,
                                         alpha95=alpha95, alpha99=alpha99, drop_unknown=drop_unknown)
                st.caption("Letters show significance within each banner column (a/b/c = 95%, A/B/C = 99%).")
                st.dataframe(_sig, use_container_width=True)

        st.divider()
        st.markdown("### Build full Excel report (All Questions × All Banners)")

        if st.button("Build XTab All Qs Excel", key="build_xtab_allqs_btn"):
            if not banners or not questions_all:
                st.warning("Select at least 1 banner and ensure there are questions remaining.")
            else:
                xls_bytes = build_xtab_allqs_excel(
                    df=df_f,
                    banners=banners,
                    questions=questions_all,
                    drop_unknown=drop_unknown,
                    percent=True,
                    show_sig=show_sig,
                    alpha95=alpha95,
                    alpha99=alpha99
                )
                st.download_button(
                    "Download XTab All Qs (Excel heatmap + sig letters)",
                    data=xls_bytes,
                    file_name="XTab_AllQs_ExcelStyle.xlsx",
                    mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
                    key="dl_xtab_allqs"
                )


# -----------------------------
# Tab 6: Visual XTab Matrix (Plotly heatmaps)
# -----------------------------
with tab6:
    st.subheader("Visual XTab Matrix (colored % heatmaps)")

    cat_cols = df_f.select_dtypes(include=["object"]).columns.tolist()
    if len(cat_cols) < 2:
        st.info("Need at least two categorical columns.")
    else:
        default_banners = [c for c in [REGION_COL, POSITION_COL, AI_USE_COL, TRAINING_COL, YEARS_COL] if c in cat_cols]
        banners = st.multiselect(
            "Banners (grouping variables)",
            options=cat_cols,
            default=default_banners[:3] if len(default_banners) else cat_cols[:2],
            key="xtab_matrix_banners"
        )

        candidate_questions = [c for c in cat_cols if c not in set(banners)]
        questions = st.multiselect(
            "Questions (to cross vs each banner)",
            options=candidate_questions,
            default=candidate_questions[:3] if len(candidate_questions) >= 3 else candidate_questions,
            key="xtab_matrix_questions"
        )

        drop_unknown = st.checkbox("Drop Unknown (recommended)", value=True, key="xtab_matrix_drop_unknown")
        max_rows = st.slider("Max response options (rows) per heatmap", 5, 60, 25, key="xtab_matrix_max_rows")
        max_cols = st.slider("Max banner categories (cols) per heatmap", 3, 30, 12, key="xtab_matrix_max_cols")

        if len(banners) == 0 or len(questions) == 0:
            st.info("Select at least 1 banner and 1 question.")
        else:
            for q in questions:
                st.markdown(f"## {q}")
                for b in banners:
                    try:
                        pct = xtab_percent_table(
                            df_f,
                            row_var=q,
                            col_var=b,
                            drop_unknown_rows=drop_unknown,
                            drop_unknown_cols=drop_unknown,
                            sort_rows_by_total=True
                        )

                        # trim for readability
                        pct2 = pct.copy()
                        if pct2.shape[1] > max_cols:
                            col_order = pct2.sum(axis=0).sort_values(ascending=False).index[:max_cols]
                            pct2 = pct2[col_order]
                        if pct2.shape[0] > max_rows:
                            row_order = pct2.sum(axis=1).sort_values(ascending=False).index[:max_rows]
                            pct2 = pct2.loc[row_order]

                        z = pct2.values
                        text = np.vectorize(lambda x: "" if np.isnan(x) else f"{x:.1f}%")(z)

                        fig = go.Figure(
                            data=go.Heatmap(
                                z=z,
                                x=[str(c) for c in pct2.columns],
                                y=[str(r) for r in pct2.index],
                                text=text,
                                texttemplate="%{text}",
                                hovertemplate="Row: %{y}<br>Col: %{x}<br>%: %{z:.1f}<extra></extra>",
                                colorbar={"title": "%"},
                            )
                        )
                        fig.update_layout(
                            title=f"{q} × {b} (column %)",
                            height=max(420, 28 * (pct2.shape[0] + 6)),
                            margin=dict(l=10, r=10, t=50, b=10),
                        )
                        st.plotly_chart(fig, use_container_width=True)

                    except Exception as e:
                        st.warning(f"Could not plot {q} × {b}: {e}")
