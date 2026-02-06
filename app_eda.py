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
tab1, tab2, tab3, tab4 = st.tabs(["Univariate", "Bivariate", "Correlations", "Geo / Map"])


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
