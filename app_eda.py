import pandas as pd
import numpy as np

import streamlit as st
import matplotlib.pyplot as plt
import plotly.express as px

from scipy.stats import chi2_contingency
from scipy.stats import spearmanr
from io import BytesIO

from openpyxl import Workbook
from openpyxl.styles import PatternFill


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
# Grouping variables (Phase 1)
# -----------------------------
ROLE_GROUP_COL = "role_group"
TENURE_GROUP_COL = "tenure_group"
REGION_GROUP_COL = "region_group"  # e.g., US vs Other
TRAINING_BIN_COL = "training_bin"  # Yes vs No

GREEN = PatternFill(start_color="C6EFCE", end_color="C6EFCE", fill_type="solid")   # p<0.05
YELLOW = PatternFill(start_color="FFEB9C", end_color="FFEB9C", fill_type="solid") # p<0.10


def build_corr_excel(r_mat: pd.DataFrame, p_mat: pd.DataFrame, n_mat: pd.DataFrame) -> bytes:
    """
    Excel with 3 sheets: Spearman_r (colored by p), Spearman_p, Pairwise_n
    """
    wb = Workbook()

    # r sheet
    ws = wb.active
    ws.title = "Spearman_r"
    ws.append([""] + list(r_mat.columns))
    for idx in r_mat.index:
        ws.append([idx] + [None if pd.isna(v) else float(v) for v in r_mat.loc[idx].values])

    # fill based on p
    for i in range(len(r_mat.index)):
        for j in range(len(r_mat.columns)):
            p = p_mat.iat[i, j]
            cell = ws.cell(row=2 + i, column=2 + j)
            if pd.isna(p):
                continue
            if p < 0.05:
                cell.fill = GREEN
            elif p < 0.10:
                cell.fill = YELLOW

    # p sheet
    ws2 = wb.create_sheet("Spearman_p")
    ws2.append([""] + list(p_mat.columns))
    for idx in p_mat.index:
        ws2.append([idx] + [None if pd.isna(v) else float(v) for v in p_mat.loc[idx].values])

    # n sheet
    ws3 = wb.create_sheet("Pairwise_n")
    ws3.append([""] + list(n_mat.columns))
    for idx in n_mat.index:
        ws3.append([idx] + [None if pd.isna(v) else float(v) for v in n_mat.loc[idx].values])

    bio = BytesIO()
    wb.save(bio)
    return bio.getvalue()

def corr_heatmap_plotly(r_mat: pd.DataFrame, p_mat: pd.DataFrame, title: str):
    """
    Plotly heatmap with r text, and border-like significance encoding via text symbols.
    - Green marker for p<0.05, Yellow marker for p<0.10
    """
    import plotly.graph_objects as go

    r = r_mat.copy()
    p = p_mat.copy()

    z = r.values
    sig = np.full(z.shape, "", dtype=object)
    sig[p.values < 0.05] = "✓"      # significant
    sig[(p.values >= 0.05) & (p.values < 0.10)] = "•"  # directional

    text = np.empty(z.shape, dtype=object)
    for i in range(z.shape[0]):
        for j in range(z.shape[1]):
            if np.isnan(z[i, j]):
                text[i, j] = ""
            else:
                text[i, j] = f"{z[i, j]:.2f}{sig[i, j]}"

    fig = go.Figure(
        data=go.Heatmap(
            z=z,
            x=[str(c) for c in r.columns],
            y=[str(i) for i in r.index],
            text=text,
            texttemplate="%{text}",
            hovertemplate="Var1: %{y}<br>Var2: %{x}<br>r: %{z:.3f}<extra></extra>",
            colorbar={"title": "Spearman r"},
        )
    )
    fig.update_layout(
        title=title,
        height=max(520, 26 * (len(r.index) + 6)),
        margin=dict(l=10, r=10, t=60, b=10),
    )
    return fig


def clean_cat_series(s: pd.Series) -> pd.Series:
    s = s.replace({np.nan: "Unknown"}).astype(str).str.strip()
    return s.replace({"": "Unknown", "nan": "Unknown", "None": "Unknown"})

def recode_role(series: pd.Series) -> pd.Series:
    s = clean_cat_series(series)
    # Ajusta estos mapeos a tus labels reales
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
    # Ajusta según cómo venga “United States” en tu data
    return s.map(lambda x: "US" if ("united states" in x or x == "us" or "u.s." in x) else ("Other" if x != "unknown" else "Unknown"))

def recode_tenure_5yrs(series: pd.Series) -> pd.Series:
    s = clean_cat_series(series).str.lower()
    # Ajusta según tus categorías reales (ej: "0-2 years", "3-5 years", "6-10 years", "10+")
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

import re
from collections import Counter

def is_multiselect_col(s: pd.Series, sep=";") -> bool:
    s2 = s.dropna().astype(str)
    if s2.empty:
        return False
    # Heurística: si muchas filas contienen separador, es multi-select
    return (s2.str.contains(re.escape(sep)).mean() >= 0.25)

def multiselect_freq_table(s: pd.Series, sep=";") -> pd.DataFrame:
    s2 = s.dropna().astype(str)
    items = []
    for cell in s2:
        parts = [p.strip() for p in cell.split(sep) if p.strip()]
        items.extend(parts)
    if not items:
        return pd.DataFrame(columns=["Option", "n", "%"])
    c = Counter(items)
    df = pd.DataFrame({"Option": list(c.keys()), "n": list(c.values())})
    df = df.sort_values("n", ascending=False).reset_index(drop=True)
    df["%"] = (df["n"] / df["n"].sum() * 100).round(1)
    return df

def single_select_freq_table(s: pd.Series) -> pd.DataFrame:
    s2 = clean_cat_series(s)
    vc = s2.value_counts(dropna=False).reset_index()
    vc.columns = ["Option", "n"]
    vc["%"] = (vc["n"] / vc["n"].sum() * 100).round(1)
    return vc

def open_end_theme_bullets(texts: list[str], top_k=5) -> list[str]:
    # resumen simple (sin LLM): top términos (filtra stopwords básicas)
    stop = set(["the","and","to","of","in","for","a","an","is","it","on","that","this","with","as","are","be","at","by","or","from"])
    tokens = []
    for t in texts:
        t = re.sub(r"[^a-zA-Z0-9\s]", " ", t.lower())
        toks = [w for w in t.split() if len(w) >= 3 and w not in stop]
        tokens.extend(toks)
    if not tokens:
        return []
    common = [w for w,_ in Counter(tokens).most_common(top_k)]
    return [f"Frequent theme/term: **{w}**" for w in common]

def is_open_end_col(colname: str, s: pd.Series) -> bool:
    name = colname.lower()
    if ("other" in name and "specify" in name) or ("please specify" in name) or ("open" in name) or ("comments" in name):
        return True
    # fallback: muchas categorías únicas y textos largos
    s2 = s.dropna().astype(str)
    if s2.empty:
        return False
    return (s2.nunique() / max(1, len(s2)) > 0.6) and (s2.str.len().median() >= 20)


from scipy.stats import fisher_exact, mannwhitneyu, kruskal, spearmanr

def fisher_or_chi2(ct: pd.DataFrame) -> dict:
    """
    Fisher exact SOLO 2x2 con scipy.
    Para RxC, devolvemos chi2 como fallback + nota (porque exact RxC no está en scipy).
    """
    out = {"test": None, "p": np.nan, "note": "", "effect": {}}
    if ct.shape == (2, 2):
        oddsratio, p = fisher_exact(ct.values)
        out["test"] = "Fisher exact (2x2)"
        out["p"] = p
        out["effect"] = {"odds_ratio": float(oddsratio)}
        return out
    else:
        chi2, p, dof, _ = chi2_contingency(ct.values)
        out["test"] = "Chi-square (fallback; Fisher exact RxC not in scipy)"
        out["p"] = p
        out["note"] = "If you require Fisher-Freeman-Halton exact (RxC), we can add a dedicated library."
        out["effect"] = {"chi2": float(chi2), "dof": int(dof)}
        return out

def likert_group_test(values: pd.Series, groups: pd.Series) -> dict:
    """
    Mann-Whitney U for 2 groups, Kruskal-Wallis for 3+.
    """
    tmp = pd.DataFrame({"y": pd.to_numeric(values, errors="coerce"), "g": clean_cat_series(groups)})
    tmp = tmp.dropna(subset=["y", "g"])
    tmp = tmp[tmp["g"] != "Unknown"]
    levels = tmp["g"].unique().tolist()

    out = {"test": None, "p": np.nan, "k": len(levels), "effect": {}}
    if len(levels) < 2:
        return out

    samples = [tmp[tmp["g"] == lv]["y"].values for lv in levels]
    if len(levels) == 2:
        u, p = mannwhitneyu(samples[0], samples[1], alternative="two-sided")
        out["test"] = "Mann-Whitney U"
        out["p"] = p
        out["effect"] = {"U": float(u), "median_diff": float(np.median(samples[0]) - np.median(samples[1]))}
        return out
    else:
        h, p = kruskal(*samples)
        out["test"] = "Kruskal-Wallis"
        out["p"] = p
        out["effect"] = {"H": float(h)}
        return out

def spearman_matrix_with_p_n(df: pd.DataFrame, cols: list[str], min_pairwise_n: int = 8):
    """
    Returns:
      r_mat: Spearman r
      p_mat: p-values
      n_mat: pairwise n used
    """
    X = df[cols].apply(pd.to_numeric, errors="coerce")

    r_mat = pd.DataFrame(index=cols, columns=cols, dtype=float)
    p_mat = pd.DataFrame(index=cols, columns=cols, dtype=float)
    n_mat = pd.DataFrame(index=cols, columns=cols, dtype=float)

    for i, a in enumerate(cols):
        for j, b in enumerate(cols):
            if i == j:
                r_mat.loc[a, b] = 1.0
                p_mat.loc[a, b] = 0.0
                n_mat.loc[a, b] = X[a].notna().sum()
            elif j < i:
                # mirror lower triangle
                r_mat.loc[a, b] = r_mat.loc[b, a]
                p_mat.loc[a, b] = p_mat.loc[b, a]
                n_mat.loc[a, b] = n_mat.loc[b, a]
            else:
                tmp = X[[a, b]].dropna()
                n = len(tmp)
                n_mat.loc[a, b] = n

                if n < min_pairwise_n:
                    r_mat.loc[a, b] = np.nan
                    p_mat.loc[a, b] = np.nan
                else:
                    r, p = spearmanr(tmp[a], tmp[b])
                    r_mat.loc[a, b] = r
                    p_mat.loc[a, b] = p

    return r_mat, p_mat, n_mat

def build_frequency_frames(df: pd.DataFrame, sep=";"):
    cat_cols = df.select_dtypes(include=["object"]).columns.tolist()
    # excluye grouping vars si existen
    exclude = {ROLE_GROUP_COL, TENURE_GROUP_COL, REGION_GROUP_COL, TRAINING_BIN_COL}
    cat_cols = [c for c in cat_cols if c not in exclude]

    singles = []
    multis = []
    openraw = []

    for c in cat_cols:
        s = df[c]
        if is_open_end_col(c, s):
            texts = s.dropna().astype(str).str.strip()
            texts = texts[texts != ""].tolist()
            for t in texts:
                openraw.append({"question": c, "response": t})
            continue

        if is_multiselect_col(s, sep=sep):
            ft = multiselect_freq_table(s, sep=sep)
            for _, r in ft.iterrows():
                multis.append({"question": c, "option": r["Option"], "n": int(r["n"]), "%": float(r["%"])})
        else:
            ft = single_select_freq_table(s)
            for _, r in ft.iterrows():
                singles.append({"question": c, "option": r["Option"], "n": int(r["n"]), "%": float(r["%"])})

    return (
        pd.DataFrame(singles),
        pd.DataFrame(multis),
        pd.DataFrame(openraw),
    )
from openpyxl.utils.dataframe import dataframe_to_rows

def write_df(ws, df: pd.DataFrame, title: str = None):
    if title:
        ws.append([title])
        ws.append([])
    for r in dataframe_to_rows(df, index=False, header=True):
        ws.append(r)

def build_master_report_excel(
    df: pd.DataFrame,
    top_findings_df: pd.DataFrame,
    sep: str,
    likert_cols: list[str],
    min_pairwise: int
) -> bytes:
    wb = Workbook()
    wb.remove(wb.active)

    # Frequencies
    single_df, multi_df, open_df = build_frequency_frames(df, sep=sep)

    ws1 = wb.create_sheet("Frequencies_Single")
    write_df(ws1, single_df)

    ws2 = wb.create_sheet("Frequencies_Multi")
    write_df(ws2, multi_df)

    ws3 = wb.create_sheet("OpenEnds_Raw")
    write_df(ws3, open_df)

    # Top10
    ws4 = wb.create_sheet("Top10_Findings")
    write_df(ws4, top_findings_df if top_findings_df is not None else pd.DataFrame())

    # Correlation
    if len(likert_cols) >= 2:
        r_mat, p_mat, n_mat = spearman_matrix_with_p_n(df_f, cols_use, min_pairwise_n=min_pairwise)

        # Use existing corr excel builder logic but write into this workbook
        ws_r = wb.create_sheet("Spearman_r")
        ws_r.append([""] + list(r_mat.columns))
        for idx in r_mat.index:
            ws_r.append([idx] + [None if pd.isna(v) else float(v) for v in r_mat.loc[idx].values])

        for i in range(len(r_mat.index)):
            for j in range(len(r_mat.columns)):
                p = p_mat.iat[i, j]
                cell = ws_r.cell(row=2 + i, column=2 + j)
                if pd.isna(p):
                    continue
                if p < 0.05:
                    cell.fill = GREEN
                elif p < 0.10:
                    cell.fill = YELLOW

        ws_p = wb.create_sheet("Spearman_p")
        ws_p.append([""] + list(p_mat.columns))
        for idx in p_mat.index:
            ws_p.append([idx] + [None if pd.isna(v) else float(v) for v in p_mat.loc[idx].values])

        ws_n = wb.create_sheet("Pairwise_n")
        ws_n.append([""] + list(n_mat.columns))
        for idx in n_mat.index:
            ws_n.append([idx] + [None if pd.isna(v) else float(v) for v in n_mat.loc[idx].values])

    # Appendix
    wsA = wb.create_sheet("Appendix")
    wsA.append(["Technical Appendix"])
    wsA.append(["Software", "Python + pandas + scipy + streamlit + openpyxl"])
    wsA.append(["Cleaning", "Trim strings; blank/nan/None -> Unknown; optional drop Unknown in tests"])
    wsA.append(["Tests", "Spearman rho (Likert/ordinal); Fisher exact (2x2) or Chi-square fallback; Mann-Whitney (2 groups); Kruskal-Wallis (3+)"])
    wsA.append(["Significance", "Green: p<0.05; Yellow: p<0.10"])

    bio = BytesIO()
    wb.save(bio)
    return bio.getvalue()


from io import BytesIO
from openpyxl import Workbook
from openpyxl.styles import PatternFill
from openpyxl.utils.dataframe import dataframe_to_rows

GREEN = PatternFill(start_color="C6EFCE", end_color="C6EFCE", fill_type="solid")  # p<0.05
YELLOW = PatternFill(start_color="FFEB9C", end_color="FFEB9C", fill_type="solid") # p<0.10

def build_corr_excel(r_mat: pd.DataFrame, p_mat: pd.DataFrame, n_mat: pd.DataFrame) -> bytes:
    wb = Workbook()

    ws = wb.active
    ws.title = "Spearman_r"
    ws.append([""] + list(r_mat.columns))
    for idx in r_mat.index:
        ws.append([idx] + [None if pd.isna(v) else float(v) for v in r_mat.loc[idx].values])

    for i in range(len(r_mat.index)):
        for j in range(len(r_mat.columns)):
            p = p_mat.iat[i, j]
            cell = ws.cell(row=2 + i, column=2 + j)
            if pd.isna(p):
                continue
            if p < 0.05:
                cell.fill = GREEN
            elif p < 0.10:
                cell.fill = YELLOW

    ws2 = wb.create_sheet("Spearman_p")
    ws2.append([""] + list(p_mat.columns))
    for idx in p_mat.index:
        ws2.append([idx] + [None if pd.isna(v) else float(v) for v in p_mat.loc[idx].values])

    ws3 = wb.create_sheet("Pairwise_n")
    ws3.append([""] + list(n_mat.columns))
    for idx in n_mat.index:
        ws3.append([idx] + [None if pd.isna(v) else float(v) for v in n_mat.loc[idx].values])

    bio = BytesIO()
    wb.save(bio)
    return bio.getvalue()

# -----------------------------
# UI: Sidebar
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

df_f = add_group_vars(df_f)

st.caption(f"Rows after filters: {len(df_f)} / {len(df)}")

with st.expander("Preview data"):
    st.dataframe(df_f.head(50), use_container_width=True)


# -----------------------------
# Tabs
# -----------------------------
tab1, tab2, tab3, tab4, tab5, tab6, tab7, tab8, tab9 = st.tabs([
    "Univariate", "Bivariate", "Correlations", "Geo / Map",
    "XTab Report", "Visual XTab Matrix",
    "Baseline Report", "Group Tests + Top Findings",
    "Correlation Matrix"
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
    st.subheader("Correlation Matrix (Spearman) + Export")

    likert_cols = detect_likert_cols(df_f)
    st.write(f"Detected Likert/ordinal (1–5) columns: {len(likert_cols)}")

    min_nonnull = st.slider("Min non-null pairs required", 5, 50, 10)
    cols_use = []
    for c in likert_cols:
        if pd.to_numeric(df_f[c], errors="coerce").notna().sum() >= min_nonnull:
            cols_use.append(c)

    if len(cols_use) < 2:
        st.info("Not enough Likert/ordinal columns with sufficient data.")
    else:
        r_mat, p_mat, n_mat = spearman_matrix_with_p_n(df_f, cols_use)

        st.markdown("### Spearman r (colored by p-value thresholds)")
        # display r matrix; you already have Plotly/Go if you want heatmap
        st.dataframe(r_mat.round(3), use_container_width=True)

        xls = build_corr_excel(r_mat, p_mat, n_mat)
        st.download_button(
            "Download correlation matrix (Excel)",
            data=xls,
            file_name="spearman_matrix_with_pvalues.xlsx",
            mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
        )

        st.caption("Color rule: green p<0.05, yellow p<0.10. Values are Spearman r.")


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

with tab7:
    st.subheader("Baseline Report (all questions)")

    cat_cols = df_f.select_dtypes(include=["object"]).columns.tolist()

    # Excluir columnas internas / grouping
    exclude = {ROLE_GROUP_COL, TENURE_GROUP_COL, REGION_GROUP_COL, TRAINING_BIN_COL}
    cat_cols = [c for c in cat_cols if c not in exclude]

    sep = st.text_input("Multi-select separator", value=";")
    show_charts = st.checkbox("Show charts for single-select", value=True)

    for c in cat_cols:
        st.markdown(f"### {c}")

        s = df_f[c]
        if is_open_end_col(c, s):
            st.markdown("**Open-end / Other specify**")
            texts = s.dropna().astype(str).str.strip()
            texts = texts[texts != ""].tolist()

            if len(texts) == 0:
                st.info("No responses.")
                continue

            st.write(f"Responses (raw): {len(texts)}")
            st.dataframe(pd.DataFrame({"response": texts}), use_container_width=True, height=220)

            bullets = open_end_theme_bullets(texts, top_k=5)
            if bullets:
                st.markdown("**Common themes (quick scan):**")
                for b in bullets[:5]:
                    st.write(f"- {b}")
            continue

        if is_multiselect_col(s, sep=sep):
            st.markdown("**Multi-select frequency table**")
            ms = multiselect_freq_table(s, sep=sep)
            st.dataframe(ms, use_container_width=True)
        else:
            ft = single_select_freq_table(s)
            st.dataframe(ft, use_container_width=True)
            if show_charts:
                # bar chart
                top_k = min(20, len(ft))
                ft2 = ft.head(top_k)
                fig = plt.figure(figsize=(10, 4))
                plt.bar(ft2["Option"].astype(str), ft2["n"].astype(int))
                plt.xticks(rotation=30, ha="right")
                plt.title(f"Counts: {c} (top {top_k})")
                plt.tight_layout()
                st.pyplot(fig)

with tab8:
    st.subheader("Segmented Group Comparisons + Top Findings")

    grouping_vars = {
        "Role (3 groups)": ROLE_GROUP_COL,
        "Tenure (<5 vs >=5)": TENURE_GROUP_COL,
        "AI Training (Yes/No)": TRAINING_BIN_COL,
        "Region (US vs Other)": REGION_GROUP_COL,
    }

    g_label = st.selectbox("Grouping variable", list(grouping_vars.keys()), index=0)
    gcol = grouping_vars[g_label]

    # Detect Likert cols (1-5) across all columns, not only numeric
    likert_cols = detect_likert_cols(df_f)
    cat_cols = df_f.select_dtypes(include=["object"]).columns.tolist()

    # Candidate outcomes:
    outcome_type = st.radio("Outcome type", ["Categorical outcome (Fisher/Chi2)", "Likert/Ordinal outcome (MWU/KW)"], index=0)

    findings = []

    if outcome_type.startswith("Categorical"):
        # Excluir grouping vars del outcome
        exclude = {ROLE_GROUP_COL, TENURE_GROUP_COL, REGION_GROUP_COL, TRAINING_BIN_COL}
        candidates = [c for c in cat_cols if c not in exclude and c != gcol]

        outcome = st.selectbox("Categorical outcome variable", candidates, index=0 if candidates else None)
        if outcome:
            tmp = df_f[[outcome, gcol]].copy()
            tmp[outcome] = clean_cat_series(tmp[outcome])
            tmp[gcol] = clean_cat_series(tmp[gcol])
            tmp = tmp[(tmp[outcome] != "Unknown") & (tmp[gcol] != "Unknown")]

            ct = pd.crosstab(tmp[outcome], tmp[gcol])
            st.markdown("### Contingency table (counts)")
            st.dataframe(ct, use_container_width=True)

            res = fisher_or_chi2(ct)
            st.write(f"Test: **{res['test']}**, p = **{res['p']:.4f}**")
            if res.get("note"):
                st.caption(res["note"])
            if res["effect"]:
                st.json(res["effect"])

    else:
        # Likert outcome
        outcome = st.selectbox("Likert/Ordinal outcome variable", likert_cols, index=0 if likert_cols else None)
        if outcome:
            res = likert_group_test(df_f[outcome], df_f[gcol])
            st.write(f"Test: **{res['test']}**, p = **{res['p']:.4f}**, k={res['k']}")
            if res["effect"]:
                st.json(res["effect"])

    st.divider()
    st.markdown("### Top 10 significant findings (auto sweep)")

    run_sweep = st.button("Run sweep across all outcomes")
    alpha = st.selectbox("Significance threshold", [0.05, 0.10], index=0)

    if run_sweep:
        # Sweep: categorical outcomes
        exclude = {ROLE_GROUP_COL, TENURE_GROUP_COL, REGION_GROUP_COL, TRAINING_BIN_COL}
        candidates_cat = [c for c in cat_cols if c not in exclude]

        for outc in candidates_cat:
            if outc == gcol:
                continue
            tmp = df_f[[outc, gcol]].copy()
            tmp[outc] = clean_cat_series(tmp[outc])
            tmp[gcol] = clean_cat_series(tmp[gcol])
            tmp = tmp[(tmp[outc] != "Unknown") & (tmp[gcol] != "Unknown")]
            if tmp.empty:
                continue
            ct = pd.crosstab(tmp[outc], tmp[gcol])
            if ct.shape[0] < 2 or ct.shape[1] < 2:
                continue
            r = fisher_or_chi2(ct)
            if not np.isnan(r["p"]):
                findings.append({
                    "grouping": g_label,
                    "outcome": outc,
                    "type": "categorical",
                    "test": r["test"],
                    "p": r["p"],
                })

        # Sweep: likert outcomes
        for outc in likert_cols:
            r = likert_group_test(df_f[outc], df_f[gcol])
            if r["test"] and not np.isnan(r["p"]):
                findings.append({
                    "grouping": g_label,
                    "outcome": outc,
                    "type": "likert",
                    "test": r["test"],
                    "p": r["p"],
                })

        #fdf = pd.DataFrame(findings).sort_values("p", ascending=True)


        if len(findings) == 0:
            st.info("No test results were generated (insufficient data after filtering / too many Unknown / no valid tables).")
        else:
            fdf = pd.DataFrame(findings)

            # ensure column exists
            if "p" not in fdf.columns:
                st.warning(f"Unexpected findings schema. Columns found: {list(fdf.columns)}")
                st.dataframe(fdf, use_container_width=True)
            else:
                fdf["p"] = pd.to_numeric(fdf["p"], errors="coerce")
                fdf = fdf.dropna(subset=["p"]).sort_values("p", ascending=True)
        
                #fdf_sig = fdf[fdf["p"] < alpha].head(10)

        if len(findings) == 0:
            st.info("No test results were generated (insufficient data after filtering / too many Unknown / no valid tables).")
            top10_df = pd.DataFrame()
        else:
            fdf = pd.DataFrame(findings)

            if "p" not in fdf.columns:
                st.warning(f"Unexpected findings schema. Columns found: {list(fdf.columns)}")
                st.dataframe(fdf, use_container_width=True)
                top10_df = pd.DataFrame()
            else:
                fdf["p"] = pd.to_numeric(fdf["p"], errors="coerce")
                fdf = fdf.dropna(subset=["p"]).sort_values("p", ascending=True)

                fdf_sig = fdf[fdf["p"] < alpha].head(10)

                if fdf_sig.empty:
                    st.info(f"No findings below p<{alpha}.")
                    top10_df = pd.DataFrame()
                else:
                    st.dataframe(fdf_sig, use_container_width=True)
                    top10_df = fdf_sig.copy()

                with st.expander("Show top 50 (by p-value)"):
                    st.dataframe(fdf.head(50), use_container_width=True)

with tab9:
    st.subheader("Spearman Correlation Matrix (Likert/Ordinal)")

    likert_cols = detect_likert_cols(df_f)
    st.write(f"Detected Likert/ordinal columns: **{len(likert_cols)}**")

    min_nonnull = st.slider("Min non-null responses per variable", 5, 100, 10)
    min_pairwise = st.slider("Min pairwise N to compute correlation", 5, 50, 8)

    cols_use = []
    for c in likert_cols:
        nn = pd.to_numeric(df_f[c], errors="coerce").notna().sum()
        if nn >= min_nonnull:
            cols_use.append(c)

    if len(cols_use) < 2:
        st.info("Not enough Likert/ordinal columns after thresholds.")
    else:
        r_mat, p_mat, n_mat = spearman_matrix_with_p_n(df_f, cols_use, min_pairwise_n=min_pairwise)

        xls = build_corr_excel(r_mat, p_mat, n_mat)


        st.caption("Legend in cells: r value + ✓ (p<0.05) or • (p<0.10).")
        fig = corr_heatmap_plotly(r_mat, p_mat, title="Spearman r (colored by magnitude) + significance markers")
        st.plotly_chart(fig, use_container_width=True)

        st.markdown("### Tables")
        c1, c2 = st.columns(2)
        with c1:
            st.markdown("**r (Spearman)**")
            st.dataframe(r_mat.round(3), use_container_width=True)
        with c2:
            st.markdown("**p-values**")
            st.dataframe(p_mat.round(4), use_container_width=True)

        xls = build_corr_excel(r_mat, p_mat, n_mat)
        st.download_button(
            "Download Correlation Matrix (Excel)",
            data=xls,
            file_name="spearman_matrix_with_pvalues.xlsx",
            mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
        )

    sep = ";"  # o el que uses en UI
    min_pairwise = 8

    # Si ya calculas top10 en tab8, guarda el DF en una variable (ver nota abajo)
    # top10_df = ...

    if st.button("Build & Download Master Spreadsheet"):

        st.divider()
        st.subheader("Deliverables (Phase 1)")

        sep_deliv = st.text_input("Multi-select separator for deliverables", value=";")
        min_pairwise_master = st.slider("Min pairwise N (correlations) for deliverables", 5, 50, 8)

        if st.button("Build Deliverables (Excel)"):
            likert_master = detect_likert_cols(df_f)
            likert_master = [c for c in likert_master if pd.to_numeric(df_f[c], errors="coerce").notna().sum() >= 10]

            try:
                _top10 = top10_df
            except NameError:
                _top10 = pd.DataFrame()

            xls_bytes = build_master_report_excel(
                df=df_f,
                top_findings_df=_top10,
                sep=sep_deliv,
                likert_cols=likert_master,
                min_pairwise=min_pairwise_master
            )

            st.download_button(
                "Download Deliverables – AiRHUB Phase 1 (Excel)",
                data=xls_bytes,
                file_name="AiRHUB_Phase1_Deliverables.xlsx",
                mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
            )
