import pandas as pd
import numpy as np

import streamlit as st
import matplotlib.pyplot as plt
import plotly.express as px

from scipy.stats import chi2_contingency

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

# If you want a world map WITHOUT pycountry, you need ISO3 codes already in your data
ISO3_COL_CANDIDATES = ["iso3", "ISO3", "ISO_3", "country_iso3"]
COUNTRY_NAME_CANDIDATES = ["Country", "country", "Primary Country", "Jurisdiction Country"]

# -----------------------------
# Helpers
# -----------------------------
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

def cramers_v(x: pd.Series, y: pd.Series) -> float:
    confusion = pd.crosstab(x, y)
    if confusion.shape[0] < 2 or confusion.shape[1] < 2:
        return np.nan
    chi2, _, _, _ = chi2_contingency(confusion)
    n = confusion.to_numpy().sum()
    r, k = confusion.shape
    phi2 = chi2 / n
    phi2corr = max(0, phi2 - ((k - 1) * (r - 1)) / (n - 1))
    rcorr = r - ((r - 1) ** 2) / (n - 1)
    kcorr = k - ((k - 1) ** 2) / (n - 1)
    denom = min((kcorr - 1), (rcorr - 1))
    if denom <= 0:
        return np.nan
    return np.sqrt(phi2corr / denom)

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

@st.cache_data
def load_data(path: str) -> pd.DataFrame:
    df = pd.read_csv(path)
    df = strip_strings(df)

    # Common numeric fields (if present)
    df = coerce_numeric(df, [
        "Duration (in seconds)", "Progress", "Q_RecaptchaScore", "Score",
        "Q_UnansweredPercentage", "Q_UnansweredQuestions"
    ])

    # Fix known typo(s) in region if present
    if REGION_COL in df.columns:
        df[REGION_COL] = (
            df[REGION_COL]
            .replace({np.nan: "Unknown"})
            .astype(str)
            .str.strip()
            .replace({"Asia-Paciifc": "Asia-Pacific", "nan": "Unknown", "": "Unknown"})
        )

    return df

def select_filter(colname: str, df: pd.DataFrame):
    if colname not in df.columns:
        return None
    vals = df[colname].replace({np.nan: "Unknown"}).astype(str)
    options = sorted(vals.unique().tolist())
    selected = st.sidebar.multiselect(colname, options, default=options)
    return selected

import plotly.graph_objects as go

def xtab_pct_df(df: pd.DataFrame, q: str, banner: str, drop_unknown=True) -> pd.DataFrame:
    tmp = df[[q, banner]].copy()
    tmp[q] = clean_cat_series(tmp[q])
    tmp[banner] = clean_cat_series(tmp[banner])

    if drop_unknown:
        tmp = tmp[(tmp[q] != "Unknown") & (tmp[banner] != "Unknown")]

    ct = pd.crosstab(tmp[q], tmp[banner], dropna=False)

    # Column % (each banner col sums to 100)
    col_sums = ct.sum(axis=0).replace(0, np.nan)
    pct = (ct.div(col_sums, axis=1) * 100).round(1)

    return pct  # rows = responses, cols = banner categories

def plot_pct_heatmap(pct: pd.DataFrame, title: str, max_rows=25, max_cols=12):
    # keep top rows/cols by mass to avoid unreadable plots
    pct2 = pct.copy()

    # reduce columns
    if pct2.shape[1] > max_cols:
        col_order = pct2.sum(axis=0).sort_values(ascending=False).index[:max_cols]
        pct2 = pct2[col_order]

    # reduce rows
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
            textfont={"size": 12},
            hovertemplate="Row: %{y}<br>Col: %{x}<br>%: %{z:.1f}<extra></extra>",
            colorbar={"title": "%"},
        )
    )
    fig.update_layout(
        title=title,
        height=max(420, 28 * (pct2.shape[0] + 6)),
        margin=dict(l=10, r=10, t=50, b=10),
    )
    st.plotly_chart(fig, use_container_width=True)


# -----------------------------
# UI: Sidebar
# -----------------------------
st.title("Survey EDA Dashboard (Interactive)")

st.sidebar.header("Data Source")
path = st.sidebar.text_input("CSV path", value=DEFAULT_PATH)
uploaded = st.sidebar.file_uploader("...or upload a CSV", type=["csv"])

if uploaded is not None:
    df = pd.read_csv(uploaded)
    df = strip_strings(df)
else:
    df = load_data(path)

st.sidebar.header("Filters")
sel_region = select_filter(REGION_COL, df)
sel_ai = select_filter(AI_USE_COL, df)
sel_training = select_filter(TRAINING_COL, df)
sel_position = select_filter(POSITION_COL, df)
sel_years = select_filter(YEARS_COL, df)

# Apply filters
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

st.caption(f"Rows after filters: {len(df_f)} / {len(df)}")

with st.expander("Preview data"):
    st.dataframe(df_f.head(50), use_container_width=True)


# -----------------------------
# Tabs
# -----------------------------
tab1, tab2, tab3, tab4, tab5, tab6 = st.tabs([
    "Univariate", "Bivariate", "Correlations", "Geo / Map", "XTab Report", "Visual XTab Matrix"
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
        bins = st.slider("Bins", 5, 60, 20)

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

        top_k = st.slider("Max categories to show (top-k)", 3, 25, 12)
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
    st.subheader("Correlations / Associations")

    st.markdown("**Numeric–Numeric:** Spearman correlation (good for Likert / non-normal).")
    num_cols = df_f.select_dtypes(include=["number"]).columns.tolist()

    if len(num_cols) >= 2:
        corr = df_f[num_cols].corr(method="spearman")
        st.dataframe(corr, use_container_width=True)

        corr_pairs = (
            corr.abs()
                .where(np.triu(np.ones(corr.shape), k=1).astype(bool))
                .stack()
                .sort_values(ascending=False)
                .head(25)
        )
        st.write("Top absolute Spearman correlations:")
        st.dataframe(
            corr_pairs.rename("abs_corr").reset_index()
                      .rename(columns={"level_0": "var1", "level_1": "var2"}),
            use_container_width=True
        )
    else:
        st.info("Not enough numeric columns to compute correlations.")

    st.markdown("**Categorical–Categorical:** Cramér’s V (top pairs).")
    cat_cols = df_f.select_dtypes(include=["object"]).columns.tolist()
    cat_cols = [c for c in cat_cols if 2 <= df_f[c].nunique(dropna=True) <= 25]

    if len(cat_cols) >= 2:
        pairs = []
        for i in range(len(cat_cols)):
            for j in range(i + 1, len(cat_cols)):
                a, b = cat_cols[i], cat_cols[j]
                v = cramers_v(
                    df_f[a].replace({np.nan: "Unknown"}),
                    df_f[b].replace({np.nan: "Unknown"})
                )
                if not np.isnan(v):
                    pairs.append((a, b, v))

        pairs = sorted(pairs, key=lambda t: t[2], reverse=True)[:25]
        st.dataframe(pd.DataFrame(pairs, columns=["var1", "var2", "cramers_v"]), use_container_width=True)
    else:
        st.info("Not enough categorical columns to compute Cramér’s V.")


# -----------------------------
# Tab 4: Geo / Map (NO pycountry)
# -----------------------------
with tab4:
    st.subheader("Geographic view (no pycountry)")

    iso3_col = next((c for c in ISO3_COL_CANDIDATES if c in df_f.columns), None)
    country_name_col = next((c for c in COUNTRY_NAME_CANDIDATES if c in df_f.columns), None)

    if iso3_col:
        # If you already have ISO3 codes, we can draw a true world map
        tmp = df_f[[iso3_col]].copy()
        tmp[iso3_col] = tmp[iso3_col].replace({np.nan: "Unknown"}).astype(str).str.strip()
        tmp = tmp[tmp[iso3_col].str.len() == 3]  # keep valid ISO3-like

        counts = tmp[iso3_col].value_counts().reset_index()
        counts.columns = ["iso3", "respondents"]

        fig = px.choropleth(
            counts,
            locations="iso3",
            color="respondents",
            title="Respondents by Country (ISO3)",
            color_continuous_scale="Blues",
        )
        st.plotly_chart(fig, use_container_width=True)

    else:
        st.info(
            "No ISO3 column detected. Without pycountry we cannot reliably map free-text country names.\n\n"
            "Options:\n"
            "1) Add an ISO3 column to your CSV (recommended)\n"
            "2) Use Region counts (fallback below)."
        )

        if REGION_COL in df_f.columns:
            rc = df_f[REGION_COL].replace({np.nan: "Unknown"}).astype(str).value_counts(dropna=False).reset_index()
            rc.columns = ["Region", "respondents"]
            st.dataframe(rc, use_container_width=True)
            robust_barh(rc["Region"], rc["respondents"], title="Respondents by Region", xlabel="Respondents")

def clean_cat_series(s: pd.Series) -> pd.Series:
    s = s.replace({np.nan: "Unknown"}).astype(str).str.strip()
    return s.replace({"": "Unknown", "nan": "Unknown", "None": "Unknown"})

def xtab_percent_table(
    df: pd.DataFrame,
    row_var: str,
    col_var: str,
    drop_unknown_rows: bool = False,
    drop_unknown_cols: bool = False,
    sort_rows_by_total: bool = False,
) -> pd.DataFrame:
    """
    Returns a table like:
      - index: response options of row_var
      - columns: banner categories of col_var + "Total"
      - values: column percentages (each column sums to 100%)
    """
    tmp = df[[row_var, col_var]].copy()
    tmp[row_var] = clean_cat_series(tmp[row_var])
    tmp[col_var] = clean_cat_series(tmp[col_var])

    if drop_unknown_rows:
        tmp = tmp[tmp[row_var] != "Unknown"]
    if drop_unknown_cols:
        tmp = tmp[tmp[col_var] != "Unknown"]

    # counts by banner
    counts = pd.crosstab(tmp[row_var], tmp[col_var], dropna=False)

    # add Total column (overall)
    counts["Total"] = counts.sum(axis=1)

    # column percentages (banner-wise)
    col_sums = counts.sum(axis=0).replace(0, np.nan)
    pct = (counts / col_sums) * 100
    pct = pct.round(1)

    # optional sort
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
# Tab 5: XTab Report (readable % table like your example)
# -----------------------------
with tab5:
    st.subheader("XTab Report (Percent tables by banner)")

    cat_cols = df_f.select_dtypes(include=["object"]).columns.tolist()
    if len(cat_cols) < 2:
        st.info("Need at least two categorical columns to build an XTab.")
    else:
        # Suggested banners (edit this list to match your questionnaire)
        default_banners = [c for c in [POSITION_COL, REGION_COL, AI_USE_COL, TRAINING_COL, YEARS_COL] if c in cat_cols]
        banner = st.selectbox(
            "Banner column (columns in the table)",
            options=cat_cols,
            index=cat_cols.index(default_banners[0]) if len(default_banners) else 0
        )

        question = st.selectbox(
            "Question column (rows in the table)",
            options=cat_cols,
            index=cat_cols.index(REGION_COL) if REGION_COL in cat_cols else 0
        )

        c1, c2, c3 = st.columns(3)
        with c1:
            drop_unknown_rows = st.checkbox("Drop Unknown (rows)", value=False)
        with c2:
            drop_unknown_cols = st.checkbox("Drop Unknown (banner cols)", value=False)
        with c3:
            sort_rows = st.checkbox("Sort rows by Total", value=False)

        show_counts = st.checkbox("Show counts table too", value=True)

        pct = xtab_percent_table(
            df_f,
            row_var=question,
            col_var=banner,
            drop_unknown_rows=drop_unknown_rows,
            drop_unknown_cols=drop_unknown_cols,
            sort_rows_by_total=sort_rows,
        )

        st.markdown("### Percent table (column %)")
        st.dataframe(pct, use_container_width=True)

        if show_counts:
            cnt = xtab_count_table(
                df_f,
                row_var=question,
                col_var=banner,
                drop_unknown_rows=drop_unknown_rows,
                drop_unknown_cols=drop_unknown_cols,
                sort_rows_by_total=sort_rows,
            )
            st.markdown("### Count table")
            st.dataframe(cnt, use_container_width=True)

        st.caption("Interpretation: Each banner column sums to ~100% (distribution of the question within each subgroup).")

# -----------------------------
# Tab 6: Visual XTab Matrix (many-by-many, colored + %)
# -----------------------------
with tab6:
    st.subheader("Visual XTab Matrix (colored % heatmaps)")

    cat_cols = df_f.select_dtypes(include=["object"]).columns.tolist()
    if len(cat_cols) < 2:
        st.info("Need at least two categorical columns.")
    else:
        st.markdown("Pick a set of **Questions** (rows) and **Banners** (columns). The plot shows **column %**.")

        # Good defaults for banners (edit to your dataset)
        default_banners = [c for c in [REGION_COL, POSITION_COL, AI_USE_COL, TRAINING_COL, YEARS_COL] if c in cat_cols]
        banners = st.multiselect(
            "Banners (grouping variables)",
            options=cat_cols,
            default=default_banners[:3] if len(default_banners) else cat_cols[:2],
        )

        # Questions: you typically want multiple survey questions (exclude banners)
        candidate_questions = [c for c in cat_cols if c not in set(banners)]
        questions = st.multiselect(
            "Questions (to cross vs each banner)",
            options=candidate_questions,
            default=candidate_questions[:3] if len(candidate_questions) >= 3 else candidate_questions,
        )

        drop_unknown = st.checkbox("Drop Unknown (recommended)", value=True)
        max_rows = st.slider("Max response options (rows) per heatmap", 5, 60, 25)
        max_cols = st.slider("Max banner categories (cols) per heatmap", 3, 30, 12)

        if len(banners) == 0 or len(questions) == 0:
            st.info("Select at least 1 banner and 1 question.")
        else:
            st.caption("Each heatmap: rows = response options, columns = banner categories, values = % within each banner column.")

            for q in questions:
                st.markdown(f"## {q}")
                for b in banners:
                    try:
                        pct = xtab_pct_df(df_f, q=q, banner=b, drop_unknown=drop_unknown)
                        plot_pct_heatmap(
                            pct,
                            title=f"{q}  ×  {b}  (column %)",
                            max_rows=max_rows,
                            max_cols=max_cols,
                        )
                    except Exception as e:
                        st.warning(f"Could not plot {q} × {b}: {e}")
