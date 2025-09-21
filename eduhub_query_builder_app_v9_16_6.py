
import streamlit as st
import pandas as pd
import numpy as np
from datetime import date, datetime
import io
from pathlib import Path
import re

st.set_page_config(page_title="EduHub Query Builder (v9.16)", layout="wide")
st.title("EduHub Query Builder (v9.16)")
st.caption("Categorical counts grouped under Measures + precise button alignment.")

# ------------------------------
# CSS: slim buttons + green Download
# ------------------------------
st.markdown("""
<style>
/* Global slim buttons */
button[kind], .stButton > button, div[data-testid="stButton"] > button,
div[data-testid="stDownloadButton"] > button {
  padding: 0.2rem 0.7rem;
  line-height: 1.1;
  font-size: 0.9rem;
  border-radius: 6px;
  white-space: nowrap;
  text-overflow: ellipsis;
  overflow: hidden;
}

/* Right-align containers for header rows */
.header-right { display: flex; justify-content: flex-end; }

/* Download results: green/white */
div[data-testid="stDownloadButton"] > button#download_results_only {
  background-color: #22c55e !important;
  color: #ffffff !important;
  border: 1px solid #16a34a !important;
}
div[data-testid="stDownloadButton"] > button#download_results_only:hover {
  background-color: #16a34a !important;
  border-color: #15803d !important;
}
</style>
""", unsafe_allow_html=True)

# ------------------------------
# Utilities
# ------------------------------
JS_SAFE_INT = (1 << 53) - 1

EPOCH_THRESHOLD_S = 10**9
EPOCH_THRESHOLD_MS = 10**12
EPOCH_THRESHOLD_NS = 10**15

def detect_epoch_unit(s: pd.Series):
    s_num = pd.to_numeric(s, errors="coerce").dropna()
    if s_num.empty:
        return None
    p75 = np.percentile(s_num, 75)
    if p75 >= EPOCH_THRESHOLD_NS and p75 < 1e20:
        return "ns"
    if p75 >= EPOCH_THRESHOLD_MS and p75 < EPOCH_THRESHOLD_NS:
        return "ms"
    if p75 >= EPOCH_THRESHOLD_S and p75 < EPOCH_THRESHOLD_MS:
        return "s"
    return None

def coerce_epoch_to_datetime(series, unit):
    s = pd.to_numeric(series, errors="coerce")
    if unit == "ns":
        return pd.to_datetime(s, unit="ns", errors="coerce")
    if unit == "ms":
        return pd.to_datetime(s, unit="ms", errors="coerce")
    if unit == "s":
        return pd.to_datetime(s, unit="s", errors="coerce")
    return pd.to_datetime(series, errors="coerce")

def load_csv(path_or_buf):
    try:
        df = pd.read_csv(path_or_buf)
    except UnicodeDecodeError:
        df = pd.read_csv(path_or_buf, encoding="utf-8-sig")
    df.columns = [c.strip() for c in df.columns]

    # 1) Name-based date parsing
    for c in df.columns:
        lc = c.lower()
        if lc in {"date","dob","created_at","updated_at"} or "date" in lc:
            unit = detect_epoch_unit(df[c])
            if unit:
                df[c] = coerce_epoch_to_datetime(df[c], unit)
            else:
                df[c] = pd.to_datetime(df[c], errors="coerce")

    # 2) Generic epoch detection
    for c in df.columns:
        if pd.api.types.is_datetime64_any_dtype(df[c]):
            continue
        s_num = pd.to_numeric(df[c], errors="coerce")
        nonnull = s_num.notna().sum()
        total = len(df[c])
        if total > 0 and nonnull / total >= 0.7:
            unit = detect_epoch_unit(s_num)
            if unit:
                df[c] = coerce_epoch_to_datetime(s_num, unit)

    # 3) Integer-like cast (e.g., Year)
    for c in df.columns:
        if pd.api.types.is_datetime64_any_dtype(df[c]):
            continue
        s_coerced = pd.to_numeric(df[c], errors="coerce")
        if s_coerced.notna().any():
            if s_coerced.dropna().apply(lambda x: float(x).is_integer()).all():
                df[c] = s_coerced.astype("Int64")
        else:
            if pd.api.types.is_float_dtype(df[c]):
                if df[c].dropna().apply(lambda x: float(x).is_integer()).all():
                    df[c] = df[c].astype("Int64")
    return df

def detect_column_types(df: pd.DataFrame):
    date_cols = [c for c in df.columns if pd.api.types.is_datetime64_any_dtype(df[c])]
    num_cols = [c for c in df.columns if pd.api.types.is_numeric_dtype(df[c]) and c not in date_cols]
    cat_cols = [c for c in df.columns if c not in num_cols and c not in date_cols]
    return cat_cols, num_cols, date_cols

def is_integer_like(series: pd.Series) -> bool:
    s = pd.to_numeric(series, errors="coerce")
    if s.notna().any():
        return s.dropna().apply(lambda x: float(x).is_integer()).all()
    return False

def sql_identifier(name: str) -> str:
    return name.replace(" ", "_")

def safe_alias(text: str) -> str:
    s = re.sub(r"[^0-9a-zA-Z_]+", "_", str(text).strip())
    s = re.sub(r"_+", "_", s).strip("_")
    return s or "value"

def sql_literal(v):
    if isinstance(v, (int, float, np.integer, np.floating)) and not isinstance(v, bool):
        try:
            return str(int(v) if float(v).is_integer() else v)
        except Exception:
            pass
    if isinstance(v, (pd.Timestamp, datetime, date)):
        return "'" + pd.to_datetime(v).date().isoformat() + "'"
    s = str(v).replace("'", "''")
    return "'" + s + "'"

def apply_filters(df: pd.DataFrame, filter_state: dict, skip_col: str = None) -> pd.DataFrame:
    out = df.copy()
    for col, f in (filter_state or {}).items():
        if col == skip_col:
            continue
        if col not in out.columns:
            continue
        if not isinstance(f, dict):
            continue
        ftype = f.get("type")
        if ftype == "categorical":
            vals = f.get("values", [])
            if vals:
                out = out[out[col].astype(str).isin([str(x) for x in vals])]
        elif ftype == "date_range":
            start = f.get("start")
            end = f.get("end")
            if start is not None and end is not None:
                start = pd.to_datetime(start)
                end = pd.to_datetime(end) + pd.Timedelta(days=1) - pd.Timedelta(milliseconds=1)
                series = pd.to_datetime(out[col], errors="coerce")
                out = out[(series >= start) & (series <= end)]
        elif ftype == "num_range":
            vmin = f.get("min")
            vmax = f.get("max")
            series = pd.to_numeric(out[col], errors="coerce")
            mask = pd.Series(True, index=out.index)
            if vmin is not None:
                mask &= series >= vmin
            if vmax is not None:
                mask &= series <= vmax
            out = out[mask]
    return out

def build_where_clause(filter_state: dict, valid_columns) -> str:
    parts = []
    for col, f in (filter_state or {}).items():
        if col not in valid_columns:
            continue
        if not isinstance(f, dict):
            continue
        key = sql_identifier(col)
        ftype = f.get("type")
        if ftype == "categorical":
            vals = f.get("values", [])
            if vals:
                lits = ", ".join([sql_literal(v) for v in vals])
                parts.append(f"{key} IN ({lits})")
        elif ftype == "date_range":
            start = f.get("start")
            end = f.get("end")
            if start is not None and end is not None:
                parts.append(f"{key} BETWEEN {sql_literal(start)} AND {sql_literal(end)}")
        elif ftype == "num_range":
            vmin = f.get("min")
            vmax = f.get("max")
            if vmin is not None and vmax is not None:
                parts.append(f"{key} BETWEEN {sql_literal(vmin)} AND {sql_literal(vmax)}")
            elif vmin is not None:
                parts.append(f"{key} >= {sql_literal(vmin)}")
            elif vmax is not None:
                parts.append(f"{key} <= {sql_literal(vmax)}")
    if parts:
        return "WHERE " + " AND ".join(parts)
    return ""

# ------------------------------
# Data sources (script dir first; else upload)
# ------------------------------
try:
    base_dir = Path(__file__).parent
except NameError:
    base_dir = Path.cwd()

default_files = {
    "Attendance": base_dir / "sample_attendance_sg_v2.csv",
    "PSLE Scores": base_dir / "sample_psle_scores_v2.csv",
}

available = {name: path for name, path in default_files.items() if path.exists()}

st.sidebar.header("1) Dataset")

# Reset helpers
def _reset_defaults():
    for k in list(st.session_state.keys()):
        if k.startswith("cat_") or k.startswith("date_") or k.startswith("num_min_") or k.startswith("num_max_") or k.startswith("drv_"):
            st.session_state.pop(k, None)
    for k in ["filter_state","dims","measures","user_limit","sort_field","sort_dir"]:
        st.session_state.pop(k, None)
    for k in list(st.session_state.keys()):
        if k.startswith("aggs_"):
            st.session_state.pop(k, None)

def _trigger_reset_to_defaults():
    st.session_state["_force_defaults"] = True
    _reset_defaults()
    st.rerun()

mode = "local" if available else "upload"

if mode == "local":
    ds_name = st.sidebar.selectbox("Choose dataset", options=list(available.keys()), key="ds_name")
    df_raw = load_csv(available[ds_name])
else:
    st.sidebar.info("Sample CSVs not found in the script folder. Upload one or more CSVs below.")
    if "uploaded_datasets" not in st.session_state:
        st.session_state["uploaded_datasets"] = {}
    uploaded = st.sidebar.file_uploader("Upload CSV files", type=["csv"], accept_multiple_files=True, key="uploader")
    for f in uploaded or []:
        df = load_csv(f)
        st.session_state["uploaded_datasets"][f.name] = df
    if not st.session_state["uploaded_datasets"]:
        st.warning("Please upload at least one CSV to proceed.")
        st.stop()
    ds_name = st.sidebar.selectbox("Choose dataset", options=list(st.session_state["uploaded_datasets"].keys()), key="ds_name_upload")
    df_raw = st.session_state["uploaded_datasets"][ds_name]

if st.session_state.get("_last_ds") != ds_name:
    st.session_state["_force_defaults"] = True
    _reset_defaults()
    st.session_state["_last_ds"] = ds_name
    st.rerun()

# Column types
cat_cols, num_cols, date_cols = detect_column_types(df_raw)

# Dimensions pool (include low-cardinality numeric like Year)
dim_num = [c for c in num_cols if df_raw[c].nunique(dropna=True) <= 200 or c.strip().lower() in {"year","term","week","month"}]
dimension_candidates = [c for c in (cat_cols + date_cols + dim_num) if c in df_raw.columns]

# Measures pool and defaults
measure_options = [c for c in num_cols if c in df_raw.columns]
default_measures = [m for m in ["Score","Average Score"] if m in measure_options]

# Low-cardinality categorical columns → derived count groups
LOW_CARD_MAX = 6
derived_groups = {}  # col -> list of values
for c in cat_cols:
    uniq = [str(x) for x in pd.Series(df_raw[c].dropna().astype(str).unique()).tolist()]
    if 1 < len(uniq) <= LOW_CARD_MAX:
        derived_groups[c] = sorted(uniq)

# Auto-defaults for classic attendance Status
auto_defaults = {}
for c, vals in derived_groups.items():
    if c.strip().lower() == "status":
        present_like = [v for v in vals if v.lower() == "present"]
        absent_like = [v for v in vals if v.lower() == "absent"]
        if present_like and absent_like:
            auto_defaults[c] = present_like + absent_like

# If a reset was requested, prepopulate defaults BEFORE widgets render
if st.session_state.pop("_force_defaults", False):
    st.session_state["dims"] = dimension_candidates[:]
    st.session_state["measures"] = default_measures
    for m in default_measures:
        st.session_state[f"aggs_{m}"] = ["avg"] if m.lower() == "score" else ["sum"]
    # derived defaults stored as drv_<col>
    for c, vals in derived_groups.items():
        st.session_state[f"drv_{c}"] = auto_defaults.get(c, [])


# ------------------------------
# Data Fields + Reset (adjacent, minimal gap)
# ------------------------------
col_label, col_reset = st.sidebar.columns([0.7, 0.3])
with col_label:
    st.markdown("## 2) Data Fields")
with col_reset:
    if st.button("Reset", key="reset_inline_btn"):
        _trigger_reset_to_defaults()

# Render widgets

default_dims = dimension_candidates[:]
dims = st.sidebar.multiselect("Choose data fields", options=dimension_candidates, default=default_dims, key="dims")

# ------------------------------
# Measures (grouped: numeric + categorical counts)
# ------------------------------
st.sidebar.subheader("Measures")

# Numeric measures
agg_options = ["count", "sum", "avg", "min", "max"]
chosen_measures = st.sidebar.multiselect("Numeric measures", options=measure_options, default=default_measures, key="measures")

agg_by_measure = {}
for m in chosen_measures:
    default_aggs = st.session_state.get(f"aggs_{m}", ["avg"] if m.lower() == "score" else ["sum"])
    agg_by_measure[m] = st.sidebar.multiselect(f"Aggregations for {m}", options=agg_options, default=default_aggs, key=f"aggs_{m}")

# Categorical counts grouped under Measures
if derived_groups:
    for c, vals in derived_groups.items():
        exp = st.sidebar.expander(f"{c} (counts)", expanded=False)
        with exp:
            key = f"drv_{c}"
            default_vals = st.session_state.get(key, auto_defaults.get(c, []))
            st.multiselect("Values to count", options=vals, default=default_vals, key=key)

# ------------------------------
# Filters
# ------------------------------
st.sidebar.header("3) Filters")

def current_cat_state():
    state = {}
    for c in cat_cols:
        key = f"cat_{c}"
        vals = st.session_state.get(key, [])
        if vals:
            state[c] = {"type": "categorical", "values": vals}
    return state

def current_date_state():
    state = {}
    for c in date_cols:
        key = f"date_{c}"
        v = st.session_state.get(key)
        if v is None:
            continue
        if isinstance(v, (list, tuple)) and len(v) == 2:
            d1, d2 = v
        else:
            d1 = d2 = v
        state[c] = {"type": "date_range", "start": d1, "end": d2}
    return state

def current_num_state():
    state = {}
    for c in num_cols:
        vmin = st.session_state.get(f"num_min_{c}")
        vmax = st.session_state.get(f"num_max_{c}")
        if vmin is None and vmax is None:
            continue
        state[c] = {"type": "num_range", "min": vmin, "max": vmax}
    return state

changed_any = False

# --- Categorical filters (sanitize BEFORE rendering)
for c in cat_cols:
    other_filters = {}
    for prev_c in cat_cols:
        if prev_c == c:
            continue
        vals = st.session_state.get(f"cat_{prev_c}", [])
        if vals:
            other_filters[prev_c] = {"type": "categorical", "values": vals}
    other_filters.update(current_date_state())
    other_filters.update(current_num_state())

    df_for_options = apply_filters(df_raw, other_filters)
    avail_vals = sorted([str(x) for x in df_for_options[c].dropna().astype(str).unique()]) if c in df_for_options.columns else []

    key = f"cat_{c}"
    cur = st.session_state.get(key, [])
    sanitized = [v for v in cur if v in avail_vals]
    if sanitized != cur:
        st.session_state[key] = sanitized
        changed_any = True

    st.sidebar.multiselect(f"{c}", options=avail_vals, default=sanitized, key=key)

    all_vals = sorted([str(x) for x in df_raw[c].dropna().astype(str).unique()]) if c in df_raw.columns else []
    unavailable = [v for v in all_vals if v not in avail_vals]
    if unavailable:
        st.sidebar.caption(f"Filtered out by other selections: {', '.join(unavailable)}")

# --- Date filters (clamp BEFORE rendering)
for c in date_cols:
    other_filters = {}
    other_filters.update(current_cat_state())
    for col, f in current_date_state().items():
        if col != c:
            other_filters[col] = f
    other_filters.update(current_num_state())

    df_for_range = apply_filters(df_raw, other_filters)
    series = pd.to_datetime(df_for_range[c], errors="coerce") if c in df_for_range.columns else pd.Series([], dtype="datetime64[ns]")
    if series.notna().any():
        cmin, cmax = series.min().date(), series.max().date()
    else:
        today = date.today()
        cmin = cmax = today

    key = f"date_{c}"
    v = st.session_state.get(key, (cmin, cmax))
    if isinstance(v, (list, tuple)) and len(v) == 2:
        d1, d2 = v
    else:
        d1 = d2 = v
    nd1 = max(cmin, min(d1, cmax))
    nd2 = max(cmin, min(d2, cmax))
    if (nd1, nd2) != (d1, d2):
        st.session_state[key] = (nd1, nd2)
        changed_any = True
    st.sidebar.date_input(f"{c} range", value=(nd1, nd2), key=key)

# --- Numeric filters (int-friendly UI + JS-safe bounds)
for c in num_cols:
    other_filters = {}
    other_filters.update(current_cat_state())
    other_filters.update(current_date_state())
    for col, f in current_num_state().items():
        if col != c:
            other_filters[col] = f

    df_for_range = apply_filters(df_raw, other_filters)
    series = pd.to_numeric(df_for_range[c], errors="coerce") if c in df_for_range.columns else pd.Series([], dtype="float64")
    if series.notna().any():
        nmin = float(series.min(skipna=True))
        nmax = float(series.max(skipna=True))
    else:
        nmin = 0.0
        nmax = 0.0

    min_key = f"num_min_{c}"
    max_key = f"num_max_{c}"
    cur_min = st.session_state.get(min_key, nmin)
    cur_max = st.session_state.get(max_key, nmax)

    integer_like = is_integer_like(series)
    within_js_bounds = (abs(nmin) <= JS_SAFE_INT) and (abs(nmax) <= JS_SAFE_INT)

    if integer_like and within_js_bounds:
        nmin_i, nmax_i = int(np.floor(nmin)), int(np.ceil(nmax))
        cur_min = int(max(nmin_i, min(int(cur_min), nmax_i))) if cur_min is not None else nmin_i
        cur_max = int(max(nmin_i, min(int(cur_max), nmax_i))) if cur_max is not None else nmax_i
        lcol, rcol = st.sidebar.columns(2)
        with lcol:
            st.number_input(f"{c} min", value=cur_min, step=1, format="%d", key=min_key)
        with rcol:
            st.number_input(f"{c} max", value=cur_max, step=1, format="%d", key=max_key)
    else:
        cur_min = max(nmin, min(float(cur_min), nmax)) if cur_min is not None else nmin
        cur_max = max(nmin, min(float(cur_max), nmax)) if cur_max is not None else nmax
        lcol, rcol = st.sidebar.columns(2)
        with lcol:
            st.number_input(f"{c} min", value=float(cur_min), key=min_key)
        with rcol:
            st.number_input(f"{c} max", value=float(cur_max), key=max_key)

if changed_any:
    st.rerun()

# Build flat filter_state
flat_filters = {}
for c in cat_cols:
    vals = st.session_state.get(f"cat_{c}", [])
    if vals:
        flat_filters[c] = {"type": "categorical", "values": vals}
for c in date_cols:
    v = st.session_state.get(f"date_{c}")
    if v is not None:
        if isinstance(v, (list, tuple)) and len(v) == 2:
            d1, d2 = v
        else:
            d1 = d2 = v
        flat_filters[c] = {"type": "date_range", "start": d1, "end": d2}
for c in num_cols:
    vmin = st.session_state.get(f"num_min_{c}")
    vmax = st.session_state.get(f"num_max_{c}")
    if vmin is not None or vmax is not None:
        flat_filters[c] = {"type": "num_range", "min": vmin, "max": vmax}

# ------------------------------
# Execute with pandas BEFORE LIMIT (with derived)
# ------------------------------
df = apply_filters(df_raw, flat_filters)

# Prepare derived specs (from grouped measure UI)
derived_specs = []  # (col, value, temp_col_name, alias_out)
for c, vals in derived_groups.items():
    chosen_vals = st.session_state.get(f"drv_{c}", [])
    for v in chosen_vals:
        temp = f"__drv__{safe_alias(c)}__{safe_alias(v)}"
        alias = f"{safe_alias(c)}_{safe_alias(v)}_count"
        derived_specs.append((c, v, temp, alias))

df_work = df.copy()
for (c, v, temp, alias) in derived_specs:
    df_work[temp] = (df_work[c].astype(str) == str(v)).astype(int)

# Measures presence
have_dims = len(st.session_state.get("dims", [])) > 0
dims = [d for d in st.session_state.get("dims", []) if d in df_work.columns]
have_num_measures = any(len(v)>0 for v in st.session_state.get("measures", []))
have_derived = len(derived_specs) > 0
have_measures = have_num_measures or have_derived

if have_dims and have_measures:
    grp_dims = dims
    if grp_dims:
        grp = df_work.groupby(grp_dims, dropna=False)
        agg_dict = {}
        # numeric measures
        for m in st.session_state.get("measures", []):
            if m not in df_work.columns:
                continue
            for how in st.session_state.get(f"aggs_{m}", []):
                key = f"{m}_{how}"
                if how == "avg":
                    agg_dict[key] = (m, "mean")
                elif how == "sum":
                    agg_dict[key] = (m, "sum")
                elif how == "min":
                    agg_dict[key] = (m, "min")
                elif how == "max":
                    agg_dict[key] = (m, "max")
                elif how == "count":
                    agg_dict[key] = (m, "count")
        # derived
        for (c, v, temp, alias) in derived_specs:
            if temp in df_work.columns:
                agg_dict[alias] = (temp, "sum")
        out_full = grp.agg(**agg_dict).reset_index() if agg_dict else df_work[grp_dims].drop_duplicates().reset_index(drop=True)
    else:
        out_full = df_work.copy()
elif have_dims and not have_measures:
    grp_dims = dims
    out_full = df_work[grp_dims].drop_duplicates().reset_index(drop=True) if grp_dims else df_work.copy()
elif not have_dims and have_measures:
    rows = {}
    # numeric measures
    for m in st.session_state.get("measures", []):
        if m not in df_work.columns:
            continue
        series = pd.to_numeric(df_work[m], errors="coerce")
        for how in st.session_state.get(f"aggs_{m}", []):
            if how == "avg":
                rows[f"{m}_avg"] = [series.mean()]
            elif how == "sum":
                rows[f"{m}_sum"] = [series.sum()]
            elif how == "min":
                rows[f"{m}_min"] = [series.min()]
            elif how == "max":
                rows[f"{m}_max"] = [series.max()]
            elif how == "count":
                rows[f"{m}_count"] = [series.count()]
    # derived
    for (c, v, temp, alias) in derived_specs:
        if temp in df_work.columns:
            rows[alias] = [int(pd.to_numeric(df_work[temp], errors="coerce").sum())]
    out_full = pd.DataFrame(rows) if rows else pd.DataFrame()
else:
    out_full = df_work.copy()

# Sort BEFORE limiting
sort_candidates = (dims + [c for c in out_full.columns if c not in dims]) if len(out_full.columns) else list(out_full.columns)
sort_field = st.sidebar.selectbox("Sort by", options=[c for c in sort_candidates if c in out_full.columns] or list(out_full.columns), index=0, key="sort_field")
sort_dir = st.sidebar.radio("Direction", ["Ascending", "Descending"], index=0, horizontal=True, key="sort_dir")
ascending = (sort_dir == "Ascending")
if sort_field in out_full.columns:
    out_full = out_full.sort_values(by=sort_field, ascending=ascending)


# Dynamic max rows + auto-synced LIMIT
dynamic_max = int(out_full.shape[0])
st.sidebar.header("4) Row limit")

# Persist and auto-sync user_limit to the current max whenever it changes
if dynamic_max == 0:
    st.sidebar.info("No rows match the current filters.")
    st.session_state["user_limit"] = 0
    st.session_state["_last_dynamic_max"] = 0
    user_limit = 0
else:
    prev_max = st.session_state.get("_last_dynamic_max")
    if prev_max != dynamic_max:
        # size changed => reset LIMIT to new maximum
        st.session_state["user_limit"] = dynamic_max
        st.session_state["_last_dynamic_max"] = dynamic_max
    # Clamp any stale value to [1, dynamic_max]
    current_limit = st.session_state.get("user_limit", dynamic_max)
    current_limit = max(1, min(int(current_limit), dynamic_max))
    st.session_state["user_limit"] = current_limit
    user_limit = st.sidebar.number_input(
        "LIMIT (rows returned)",
        min_value=1,
        max_value=dynamic_max,
        value=current_limit,
        step=1,
        format="%d",
        key="user_limit"
    )

# Apply LIMIT

out = out_full.head(int(user_limit)) if dynamic_max > 0 else out_full.head(0)

# Tidy integer-like columns for display
for c in out.columns:
    if pd.api.types.is_float_dtype(out[c]):
        if out[c].dropna().apply(lambda x: float(x).is_integer()).all():
            out[c] = out[c].astype("Int64")

# Add S/N
out = out.reset_index(drop=True)
out.insert(0, "S/N", range(1, len(out)+1))

# ------------------------------
# Results header with flush-right Download (v9_16_1: align like v9_10)
# ------------------------------
res_left, res_mid, res_right = st.columns([0.68, 0.14, 0.18])
with res_left:
    st.subheader("Results")
with res_mid:
    st.write("")
with res_right:
    st.markdown('<div class="header-right">', unsafe_allow_html=True)
    csv_buf = io.StringIO()
    out.to_csv(csv_buf, index=False)
    st.download_button("Download results", data=csv_buf.getvalue(), file_name="eduhub_query_results.csv", mime="text/csv", key="download_results_only", use_container_width=True)
    st.markdown('</div>', unsafe_allow_html=True)

# Table

st.dataframe(out, use_container_width=True, hide_index=True)

# ------------------------------
# SQL preview (include derived SUM(CASE WHEN ...))
# ------------------------------
def agg_sql_expr(col, how):
    key = sql_identifier(col)
    if how == "count":
        return f"COUNT({key}) AS {key}_count"
    if how == "sum":
        return f"SUM({key}) AS {key}_sum"
    if how == "avg":
        return f"AVG({key}) AS {key}_avg"
    if how == "min":
        return f"MIN({key}) AS {key}_min"
    if how == "max":
        return f"MAX({key}) AS {key}_max"
    return key

select_parts = []
group_by_parts = []
if dims:
    group_by_parts = [sql_identifier(d) for d in dims if d in df_raw.columns]
    select_parts.extend(group_by_parts)
for m in st.session_state.get("measures", []):
    for how in st.session_state.get(f"aggs_{m}", []):
        if m in df_raw.columns:
            select_parts.append(agg_sql_expr(m, how))

for (c, v, temp, alias) in derived_specs:
    col_id = sql_identifier(c)
    alias_id = safe_alias(alias)
    select_parts.append(f"SUM(CASE WHEN {col_id} = {sql_literal(v)} THEN 1 ELSE 0 END) AS {alias_id}")

where_clause = build_where_clause(flat_filters, df_raw.columns)
order_col = sql_identifier(sort_field) if sort_field else None
order_clause = f"ORDER BY {order_col} {'DESC' if sort_dir=='Descending' else 'ASC'}" if order_col else ""
select_clause = ",\n       ".join(select_parts) if select_parts else "*"
group_by_clause = f"GROUP BY {', '.join(group_by_parts)}" if group_by_parts else ""
limit_value = int(user_limit) if dynamic_max > 0 else 0
generated_sql = f"""
SELECT
       {select_clause}
FROM {{approved_curated_view_for_{ds_name.replace(' ', '_')}}}
{where_clause}
{group_by_clause}
{order_clause}
LIMIT {limit_value};
""".strip()

# ------------------------------
# Generated SQL
# ------------------------------
st.subheader("Generated Structured Query Language")
st.code(generated_sql, language="sql")


