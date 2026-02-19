# app.py
import json
import re
from pathlib import Path
from urllib.parse import urlparse, parse_qs

import pandas as pd
import plotly.express as px
import streamlit as st


# -----------------------------
# Config
# -----------------------------
# Use relative paths from the script location for Streamlit Cloud compatibility
SCRIPT_DIR = Path(__file__).parent
DEFAULT_CSV = SCRIPT_DIR / "webapp submisssions.csv"
DEFAULT_PROPOSALS_JSON = SCRIPT_DIR / "proposals_by_category.json"

BUCKET_SUPPORT = {"Critical to Me & Others", "Good Idea"}
BUCKET_UNSURE = {"Not Sure"}
BUCKET_UNSUPPORT = {"Dislike It", "Hate It"}

CATEGORY_ORDER = [
    "Economic Mobility and Growth",
    "K-12 Education",
    "Healthcare",
    "Energy Policy",
    "Taxes",
    "Federal Spending & Debt",
    "Congressional Reform",
]

BUCKET_ORDER = [
    "Critical to Me & Others",
    "Good Idea",
    "Not Sure",
    "Dislike It",
    "Hate It",
]


# -----------------------------
# Helpers
# -----------------------------
def clean_df(df: pd.DataFrame) -> pd.DataFrame:
    """Force a DataFrame to have a clean, contiguous index starting from 0."""
    if df.empty:
        return df.copy()
    # Use deep copy to ensure no references to original index remain
    result = df.copy(deep=True)
    result.index = pd.RangeIndex(len(result))
    return result


def normalize_blank(x) -> str:
    if x is None:
        return ""
    s = str(x).strip()
    return "" if s.lower() in {"nan", "none", "null"} else s


def parse_boolish(x):
    s = normalize_blank(x)
    if not s:
        return None
    s2 = s.strip().lower()
    if s2 in {"true", "t", "yes", "y", "1"}:
        return True
    if s2 in {"false", "f", "no", "n", "0"}:
        return False
    return None


def safe_json_loads(x):
    s = normalize_blank(x)
    if not s:
        return None
    try:
        return json.loads(s)
    except Exception:
        return None


def extract_utm_source(origin: str):
    s = normalize_blank(origin)
    if not s:
        return ""
    try:
        q = parse_qs(urlparse(s).query)
        return (q.get("utm_source") or [""])[0]
    except Exception:
        return ""


def read_proposals_by_category(fp: Path) -> pd.DataFrame:
    raw = json.loads(fp.read_text(encoding="utf-8"))

    rows = []
    # Expected structure:
    # { "Category": [ { "id": ..., "title": "..." }, ... ], ... }
    for category, items in raw.items():
        if not isinstance(items, list):
            continue
        for item in items:
            if not isinstance(item, dict):
                continue

            reform_key = item.get("id")
            title = item.get("title") or item.get("proposal")

            if reform_key is None or title is None:
                continue

            rows.append(
                {
                    "category": str(category).strip(),
                    "title": str(title).strip(),
                    "reform_key": str(reform_key).strip(),
                }
            )

    df = pd.DataFrame(rows)
    if df.empty:
        return pd.DataFrame(columns=["category", "title", "reform_key", "lookup_key"])

    df["lookup_key"] = (df["category"] + "||" + df["title"]).str.strip().values
    return df.drop_duplicates(subset=["lookup_key"]).reset_index(drop=True)


def build_rating_items(submissions: pd.DataFrame, reforms_ref: pd.DataFrame) -> pd.DataFrame:
    reforms_map = dict(zip(reforms_ref["lookup_key"], reforms_ref["reform_key"]))

    out = []
    for _, row in submissions.iterrows():
        session_id = normalize_blank(row.get("session_id"))
        if not session_id:
            continue

        rating_mode = normalize_blank(row.get("rating_mode")).lower()  # 'overall' / 'individual'
        assignment = safe_json_loads(row.get("assignment"))

        if not isinstance(assignment, list):
            continue

        origin = normalize_blank(row.get("origin"))
        utm_source = extract_utm_source(origin)
        version = normalize_blank(row.get("version"))
        affiliation = normalize_blank(row.get("affiliation")) or ""
        email = normalize_blank(row.get("email"))
        email_present = bool(email)
        reflection_q2_raw = row.get("reflection_q2")
        reflection_q2_value = parse_boolish(reflection_q2_raw)
        completed = normalize_blank(reflection_q2_raw) != ""

        # completion gate: only count completed downstream
        if not completed:
            continue

        for item in assignment:
            if not isinstance(item, dict):
                continue

            category = normalize_blank(item.get("category"))
            bucket = normalize_blank(item.get("bucket"))

            # timestamp (prefer item_submitted_at; fall back to submitted_at)
            ts = normalize_blank(item.get("item_submitted_at")) or normalize_blank(item.get("submitted_at"))
            # parse later; keep raw here
            proposal_title = ""
            reform_key = ""

            if rating_mode == "individual":
                proposal_title = normalize_blank(item.get("proposal") or item.get("title"))
                lk = f"{category}||{proposal_title}".strip()
                reform_key = reforms_map.get(lk, "")

            out.append(
                {
                    "session_id": session_id,
                    "completed": completed,
                    "reflection_q2_value": reflection_q2_value,
                    "rating_mode": rating_mode,
                    "category": category,
                    "bucket": bucket,
                    "item_submitted_at_raw": ts,
                    "version": version,
                    "origin": origin,
                    "utm_source": utm_source,
                    "email_present": email_present,
                    "affiliation": affiliation,
                    "proposal_title": proposal_title,
                    "reform_key": reform_key,
                }
            )

    df = pd.DataFrame(out)
    if df.empty:
        return clean_df(df)

    df = clean_df(df)
    # Convert to datetime - using .values ensures we get an array, not a Series with index alignment issues
    dt_array = pd.to_datetime(df["item_submitted_at_raw"].values, errors="coerce", utc=True)
    # Assign as array - df already has clean index from clean_df()
    df.loc[:, "item_submitted_at"] = dt_array
    df["is_support"] = df["bucket"].isin(BUCKET_SUPPORT).astype(int).values
    df["is_unsure"] = df["bucket"].isin(BUCKET_UNSURE).astype(int).values
    df["is_unsupport"] = df["bucket"].isin(BUCKET_UNSUPPORT).astype(int).values

    # Make categories consistent ordering where possible
    df["category"] = pd.Categorical(df["category"].values, categories=CATEGORY_ORDER, ordered=True)

    return clean_df(df)


def build_area_rollups(rating_items: pd.DataFrame) -> pd.DataFrame:
    if rating_items.empty:
        return clean_df(rating_items)

    # Ensure rating_items has a completely clean index first
    rating_items = clean_df(rating_items)
    
    # Overall mode: already 1 row per (session, category)
    overall_mask = rating_items["rating_mode"] == "overall"
    overall = clean_df(rating_items.loc[overall_mask])
    
    # Extract grouping columns as arrays to avoid index alignment issues
    group_cols = ["session_id", "category", "rating_mode", "version", "origin", "utm_source", "email_present", "affiliation", "reflection_q2_value"]
    # Convert to regular Python arrays, especially for categorical columns
    group_data = {}
    for col in group_cols:
        col_data = overall[col]
        if isinstance(col_data.dtype, pd.CategoricalDtype):
            # Convert categorical to regular array
            group_data[col] = col_data.astype(str).values
        else:
            group_data[col] = col_data.values
    
    # Create a temporary DataFrame with clean index for grouping
    overall_for_groupby = pd.DataFrame(group_data)
    overall_for_groupby["is_support"] = overall["is_support"].values
    overall_for_groupby["is_unsure"] = overall["is_unsure"].values
    overall_for_groupby["is_unsupport"] = overall["is_unsupport"].values
    overall_for_groupby["bucket"] = overall["bucket"].values
    overall_for_groupby["item_submitted_at"] = overall["item_submitted_at"].values
    
    overall_roll = (
        overall_for_groupby.groupby(
            group_cols,
            dropna=False,
            as_index=False,
        )
        .agg(
            support_n=("is_support", "sum"),
            unsure_n=("is_unsure", "sum"),
            unsupport_n=("is_unsupport", "sum"),
            total_n=("bucket", "count"),
            min_ts=("item_submitted_at", "min"),
            max_ts=("item_submitted_at", "max"),
        )
        .reset_index(drop=True)
    )

    # Individual mode: multiple reforms per (session, category)
    indiv_mask = rating_items["rating_mode"] == "individual"
    indiv = clean_df(rating_items.loc[indiv_mask])
    
    # Extract grouping columns as arrays to avoid index alignment issues
    group_data_indiv = {}
    for col in group_cols:
        col_data = indiv[col]
        if isinstance(col_data.dtype, pd.CategoricalDtype):
            # Convert categorical to regular array
            group_data_indiv[col] = col_data.astype(str).values
        else:
            group_data_indiv[col] = col_data.values
    
    # Create a temporary DataFrame with clean index for grouping
    indiv_for_groupby = pd.DataFrame(group_data_indiv)
    indiv_for_groupby["is_support"] = indiv["is_support"].values
    indiv_for_groupby["is_unsure"] = indiv["is_unsure"].values
    indiv_for_groupby["is_unsupport"] = indiv["is_unsupport"].values
    indiv_for_groupby["bucket"] = indiv["bucket"].values
    indiv_for_groupby["item_submitted_at"] = indiv["item_submitted_at"].values
    
    indiv_roll = (
        indiv_for_groupby.groupby(
            group_cols,
            dropna=False,
            as_index=False,
        )
        .agg(
            support_n=("is_support", "sum"),
            unsure_n=("is_unsure", "sum"),
            unsupport_n=("is_unsupport", "sum"),
            total_n=("bucket", "count"),
            min_ts=("item_submitted_at", "min"),
            max_ts=("item_submitted_at", "max"),
        )
        .reset_index(drop=True)
    )

    roll = clean_df(pd.concat([overall_roll, indiv_roll], ignore_index=True))
    roll["pct_support"] = (roll["support_n"] / roll["total_n"]).values
    roll["pct_unsure"] = (roll["unsure_n"] / roll["total_n"]).values
    roll["pct_unsupport"] = (roll["unsupport_n"] / roll["total_n"]).values
    roll["category"] = pd.Categorical(roll["category"].values, categories=CATEGORY_ORDER, ordered=True)
    return clean_df(roll)


def pct_fmt(x):
    if pd.isna(x):
        return ""
    return f"{x*100:.1f}%"


def compute_policy_area_summary(area_rollups: pd.DataFrame) -> pd.DataFrame:
    if area_rollups.empty:
        return clean_df(area_rollups)

    agg = (
        area_rollups.groupby("category", dropna=False, as_index=False)
        .agg(
            support_n=("support_n", "sum"),
            unsure_n=("unsure_n", "sum"),
            unsupport_n=("unsupport_n", "sum"),
            total_n=("total_n", "sum"),
            n_overall=("rating_mode", lambda s: (s == "overall").sum()),
            n_individual=("rating_mode", lambda s: (s == "individual").sum()),
        )
        .reset_index(drop=True)
    )
    agg["pct_support"] = (agg["support_n"] / agg["total_n"]).values
    agg["pct_unsure"] = (agg["unsure_n"] / agg["total_n"]).values
    agg["pct_unsupport"] = (agg["unsupport_n"] / agg["total_n"]).values
    agg["category"] = pd.Categorical(agg["category"].values, categories=CATEGORY_ORDER, ordered=True)
    return clean_df(agg.sort_values("category"))


def compute_reform_leaderboard(rating_items: pd.DataFrame, reforms_ref: pd.DataFrame) -> pd.DataFrame:
    if rating_items.empty:
        return clean_df(rating_items)

    indiv_mask = rating_items["rating_mode"] == "individual"
    indiv = clean_df(rating_items.loc[indiv_mask])
    if indiv.empty:
        return clean_df(indiv)

    # Extract grouping columns as arrays to avoid index alignment issues
    group_cols = ["reform_key", "proposal_title", "category"]
    group_data = {}
    for col in group_cols:
        col_data = indiv[col]
        if isinstance(col_data.dtype, pd.CategoricalDtype):
            # Convert categorical to regular array
            group_data[col] = col_data.astype(str).values
        else:
            group_data[col] = col_data.values
    
    # Create a temporary DataFrame with clean index for grouping
    indiv_for_groupby = pd.DataFrame(group_data)
    indiv_for_groupby["is_support"] = indiv["is_support"].values
    indiv_for_groupby["is_unsure"] = indiv["is_unsure"].values
    indiv_for_groupby["is_unsupport"] = indiv["is_unsupport"].values
    indiv_for_groupby["bucket"] = indiv["bucket"].values
    indiv_for_groupby["session_id"] = indiv["session_id"].values

    grp = (
        indiv_for_groupby.groupby(group_cols, dropna=False, as_index=False)
        .agg(
            support_n=("is_support", "sum"),
            unsure_n=("is_unsure", "sum"),
            unsupport_n=("is_unsupport", "sum"),
            total_n=("bucket", "count"),
            n_sessions=("session_id", "nunique"),
        )
        .reset_index(drop=True)
    )
    grp["pct_support"] = (grp["support_n"] / grp["total_n"]).values
    grp["pct_unsure"] = (grp["unsure_n"] / grp["total_n"]).values
    grp["pct_unsupport"] = (grp["unsupport_n"] / grp["total_n"]).values

    # add mapping quality signal
    grp["mapped"] = (grp["reform_key"].astype(str).str.strip() != "").values
    grp["category"] = pd.Categorical(grp["category"].values, categories=CATEGORY_ORDER, ordered=True)

    grp = grp.sort_values(["pct_support", "n_sessions"], ascending=[False, False])
    return clean_df(grp)


# -----------------------------
# UI
# -----------------------------
st.set_page_config(page_title="GBP Live Dashboard (Local)", layout="wide")

st.title("GBP Live Dashboard (Local)")

with st.sidebar:
    st.header("Data source")

    csv_path = st.text_input("Submissions CSV path", value=str(DEFAULT_CSV))
    json_path = st.text_input("proposals_by_category.json path", value=str(DEFAULT_PROPOSALS_JSON))

    st.divider()
    st.header("Filters")

    # placeholders; actual date bounds set after load
    date_from = st.date_input("Date from (UTC)", value=None)
    date_to = st.date_input("Date to (UTC)", value=None)

    category_filter = st.multiselect("Category", options=CATEGORY_ORDER, default=[])
    mode_filter = st.multiselect("Rating mode", options=["overall", "individual"], default=[])
    version_filter = st.multiselect("Version", options=[], default=[])
    utm_filter = st.multiselect("UTM source", options=[], default=[])
    affiliation_filter = st.multiselect("Affiliation", options=[], default=[])


@st.cache_data(show_spinner=False)
def load_inputs(csv_fp: str, json_fp: str):
    csv_file = Path(csv_fp)
    json_file = Path(json_fp)

    if not csv_file.exists():
        raise FileNotFoundError(f"CSV not found: {csv_file}")
    if not json_file.exists():
        raise FileNotFoundError(f"JSON not found: {json_file}")

    submissions = pd.read_csv(csv_file, index_col=False).reset_index(drop=True)
    reforms_ref = read_proposals_by_category(json_file)
    rating_items = build_rating_items(submissions, reforms_ref)
    # Ensure rating_items has clean index before passing to build_area_rollups
    rating_items = clean_df(rating_items)
    area_rollups = build_area_rollups(rating_items)

    return submissions, reforms_ref, rating_items, area_rollups


try:
    submissions_df, reforms_ref_df, rating_items_df, area_rollups_df = load_inputs(csv_path, json_path)
except Exception as e:
    import traceback
    error_msg = f"Error loading data: {str(e)}\n\nFull traceback:\n{traceback.format_exc()}"
    st.error(error_msg)
    st.stop()

# Update sidebar filter options dynamically
with st.sidebar:
    st.caption(f"Loaded submissions rows: {len(submissions_df):,}")
    st.caption(f"Flattened rating items (completed only): {len(rating_items_df):,}")

    versions = sorted([v for v in rating_items_df["version"].dropna().unique().tolist() if str(v).strip() != ""])
    utms = sorted([u for u in rating_items_df["utm_source"].dropna().unique().tolist() if str(u).strip() != ""])
    affs = sorted([a for a in rating_items_df["affiliation"].dropna().unique().tolist() if str(a).strip() != ""])

    # re-render multiselects with populated options
    st.session_state.setdefault("version_filter", [])
    st.session_state.setdefault("utm_filter", [])
    st.session_state.setdefault("affiliation_filter", [])

# Apply filters (date range uses rating_items timestamp)
try:
    df_items = clean_df(rating_items_df)
except Exception as e:
    import traceback
    st.error(f"Error in clean_df(rating_items_df): {str(e)}\n\nTraceback:\n{traceback.format_exc()}")
    st.stop()

# Date bounds default to all time if user didn't pick
if not df_items.empty:
    min_dt = df_items["item_submitted_at"].min()
    max_dt = df_items["item_submitted_at"].max()
else:
    min_dt = None
    max_dt = None

# Interpret date inputs
if min_dt is not None and max_dt is not None:
    if date_from is not None:
        df_items = clean_df(df_items[df_items["item_submitted_at"] >= pd.Timestamp(date_from).tz_localize("UTC")])
    if date_to is not None:
        # inclusive end date
        df_items = clean_df(df_items[df_items["item_submitted_at"] <= (pd.Timestamp(date_to) + pd.Timedelta(days=1) - pd.Timedelta(seconds=1)).tz_localize("UTC")])

if category_filter:
    df_items = clean_df(df_items[df_items["category"].isin(category_filter)])
if mode_filter:
    df_items = clean_df(df_items[df_items["rating_mode"].isin(mode_filter)])
if version_filter:
    df_items = clean_df(df_items[df_items["version"].isin(version_filter)])
if utm_filter:
    df_items = clean_df(df_items[df_items["utm_source"].isin(utm_filter)])
if affiliation_filter:
    df_items = clean_df(df_items[df_items["affiliation"].isin(affiliation_filter)])

# Session universe for this filtered view
session_ids_in_view = set(df_items["session_id"].unique().tolist())

# Rebuild rollups from filtered items
try:
    df_roll = build_area_rollups(df_items)
except Exception as e:
    import traceback
    st.error(f"Error in build_area_rollups: {str(e)}\n\nTraceback:\n{traceback.format_exc()}")
    st.stop()

# -----------------------------
# Tabs
# -----------------------------
tab_overview, tab_policy, tab_reforms, tab_demo, tab_qa = st.tabs(
    ["Overview", "Policy Areas (Combined)", "Reforms (Individual)", "Email + Affiliation", "QA / Mapping"]
)

with tab_overview:
    st.subheader("Overview (Completed only)")

    # Completed submissions within filter window = sessions with any rating item in view
    # (rating_items already completion-gated)
    completed_sessions_in_view = len(session_ids_in_view)

    # Mode split based on submissions present in view
    if completed_sessions_in_view == 0:
        st.info("No completed sessions in the current filter range.")
    else:
        # Extract columns as arrays to avoid index alignment issues
        sess_for_groupby = pd.DataFrame({
            "session_id": df_items["session_id"].values,
            "rating_mode": df_items["rating_mode"].values,
            "reflection_q2_value": df_items["reflection_q2_value"].values,
            "email_present": df_items["email_present"].values
        })
        
        sess_mode = (
            sess_for_groupby.groupby("session_id", as_index=False)
            .agg(rating_mode=("rating_mode", "first"), reflection_q2_value=("reflection_q2_value", "first"), email_present=("email_present", "first"))
            .reset_index(drop=True)
        )

        overall_n = int((sess_mode["rating_mode"] == "overall").sum())
        indiv_n = int((sess_mode["rating_mode"] == "individual").sum())

        q2_true = int((sess_mode["reflection_q2_value"] == True).sum())  # noqa: E712
        q2_false = int((sess_mode["reflection_q2_value"] == False).sum())  # noqa: E712
        q2_known = q2_true + q2_false

        email_n = int(sess_mode["email_present"].sum())

        c1, c2, c3, c4, c5 = st.columns(5)
        c1.metric("Completed submissions", f"{completed_sessions_in_view:,}")
        c2.metric("Overall mode", f"{overall_n:,}")
        c3.metric("Individual mode", f"{indiv_n:,}")
        c4.metric("Emails provided", f"{email_n:,}", f"{(email_n/completed_sessions_in_view)*100:.1f}%")
        c5.metric("reflection_q2 known", f"{q2_known:,}", f"{(q2_known/completed_sessions_in_view)*100:.1f}%")

        st.divider()

        # reflection_q2 true/false percentages
        if q2_known > 0:
            st.subheader("Reflection Q2: True vs False")
            q2_true_pct = (q2_true / q2_known) * 100
            q2_false_pct = (q2_false / q2_known) * 100
            
            col1, col2 = st.columns(2)
            col1.metric("True", f"{q2_true:,}", f"{q2_true_pct:.1f}%")
            col2.metric("False", f"{q2_false:,}", f"{q2_false_pct:.1f}%")
            
            st.divider()

        # reflection_q2 true/false by mode
        q2_filtered = sess_mode[sess_mode["reflection_q2_value"].notna()]
        q2_filtered = clean_df(q2_filtered)
        
        # Extract grouping columns as arrays to avoid index alignment issues
        group_cols_q2 = ["rating_mode", "reflection_q2_value"]
        group_data_q2 = {}
        for col in group_cols_q2:
            col_data = q2_filtered[col]
            if isinstance(col_data.dtype, pd.CategoricalDtype):
                group_data_q2[col] = col_data.astype(str).values
            else:
                group_data_q2[col] = col_data.values
        
        q2_for_groupby = pd.DataFrame(group_data_q2)
        q2_for_groupby["session_id"] = q2_filtered["session_id"].values
        
        q2_by_mode = (
            q2_for_groupby.groupby(group_cols_q2, as_index=False)
            .agg(n=("session_id", "count"))
            .reset_index(drop=True)
        )
        if not q2_by_mode.empty:
            # Calculate percentages by mode
            q2_by_mode["reflection_q2_value"] = q2_by_mode["reflection_q2_value"].map({True: "True", False: "False"}).values
            
            # Add percentage calculations
            mode_totals = q2_by_mode.groupby("rating_mode")["n"].sum()
            q2_by_mode["pct"] = q2_by_mode.apply(
                lambda row: (row["n"] / mode_totals[row["rating_mode"]]) * 100, axis=1
            )
            
            st.subheader("Reflection Q2: True vs False by Mode")
            
            # Display percentages in a table
            display_df = q2_by_mode.copy()
            display_df["Count"] = display_df["n"]
            display_df["Percentage"] = display_df["pct"].map(lambda x: f"{x:.1f}%")
            display_df = display_df[["rating_mode", "reflection_q2_value", "Count", "Percentage"]]
            display_df = display_df.rename(columns={
                "rating_mode": "Mode",
                "reflection_q2_value": "Q2 Value"
            })
            st.dataframe(display_df, use_container_width=True, hide_index=True)
            
            # Chart
            fig = px.bar(q2_by_mode, x="rating_mode", y="n", color="reflection_q2_value", barmode="stack",
                        text=q2_by_mode["pct"].map(lambda x: f"{x:.1f}%"),
                        labels={"rating_mode": "Rating Mode", "n": "Count", "reflection_q2_value": "Q2 Value"})
            fig.update_traces(textposition="inside")
            st.plotly_chart(fig, use_container_width=True)

        # submissions over time (by day)
        ts = clean_df(df_items.dropna(subset=["item_submitted_at"]))
        if not ts.empty:
            # Extract columns as arrays to avoid index alignment issues
            day_vals = ts["item_submitted_at"].dt.date.values
            session_id_vals = ts["session_id"].values
            
            ts_for_groupby = pd.DataFrame({
                "day": day_vals,
                "session_id": session_id_vals
            })
            
            by_day = ts_for_groupby.groupby(["day"], as_index=False).agg(sessions=("session_id", "nunique")).reset_index(drop=True)
            fig2 = px.line(by_day, x="day", y="sessions")
            st.plotly_chart(fig2, use_container_width=True)

        # Top UTM sources
        # Extract columns as arrays to avoid index alignment issues
        utm_for_groupby = pd.DataFrame({
            "utm_source": df_items["utm_source"].values,
            "session_id": df_items["session_id"].values
        })
        
        top_utm = (
            utm_for_groupby.groupby("utm_source", as_index=False)
            .agg(sessions=("session_id", "nunique"))
            .reset_index(drop=True)
            .sort_values("sessions", ascending=False)
        )
        top_utm = top_utm[top_utm["utm_source"].astype(str).str.strip() != ""].head(20).reset_index(drop=True)
        if not top_utm.empty:
            fig3 = px.bar(top_utm, x="utm_source", y="sessions")
            st.plotly_chart(fig3, use_container_width=True)

with tab_policy:
    st.subheader("Policy Areas (Combined Overall + Individual Rollups)")

    if df_roll.empty:
        st.info("No policy-area data in the current filter range.")
    else:
        summary = compute_policy_area_summary(df_roll)

        # stacked 100% bars
        long = summary.melt(
            id_vars=["category", "n_overall", "n_individual", "total_n"],
            value_vars=["pct_support", "pct_unsure", "pct_unsupport"],
            var_name="metric",
            value_name="pct",
        ).reset_index(drop=True)
        metric_name = {
            "pct_support": "Support (Critical + Good)",
            "pct_unsure": "Not sure",
            "pct_unsupport": "Unsupport (Dislike + Hate)",
        }
        long["metric"] = long["metric"].map(metric_name).values

        fig = px.bar(long, x="pct", y="category", color="metric", orientation="h")
        fig.update_layout(xaxis_tickformat=".0%")
        st.plotly_chart(fig, use_container_width=True)

        # transparency table
        t = clean_df(summary)
        t["support%"] = t["pct_support"].map(pct_fmt).values
        t["not_sure%"] = t["pct_unsure"].map(pct_fmt).values
        t["unsupport%"] = t["pct_unsupport"].map(pct_fmt).values
        t = t[["category", "support%", "not_sure%", "unsupport%", "n_overall", "n_individual", "total_n"]]
        t = t.rename(columns={"total_n": "denominator_votes"})
        st.dataframe(t, use_container_width=True, hide_index=True)

with tab_reforms:
    st.subheader("Reforms (Individual mode only)")

    indiv_items = clean_df(df_items[df_items["rating_mode"] == "individual"])
    if indiv_items.empty:
        st.info("No individual-mode reform data in the current filter range.")
    else:
        leaderboard = compute_reform_leaderboard(indiv_items, reforms_ref_df)

        # filter out unmapped reform_key for the main leaderboard
        mapped_only = st.checkbox("Show only mapped reforms (recommended)", value=True)
        show = clean_df(leaderboard[leaderboard["mapped"]]) if mapped_only else clean_df(leaderboard)

        # chart top N by support%
        top_n = st.slider("Top N by Support%", min_value=10, max_value=50, value=25, step=5)
        chart_df = clean_df(show.head(top_n))
        chart_df["support_pct"] = chart_df["pct_support"].values
        fig = px.bar(chart_df, x="pct_support", y="proposal_title", orientation="h")
        fig.update_layout(xaxis_tickformat=".0%", yaxis_title="")
        st.plotly_chart(fig, use_container_width=True)

        # table
        table = clean_df(show)
        table["Support%"] = table["pct_support"].map(pct_fmt).values
        table["Not sure%"] = table["pct_unsure"].map(pct_fmt).values
        table["Unsupport%"] = table["pct_unsupport"].map(pct_fmt).values
        table = table[
            ["category", "proposal_title", "reform_key", "Support%", "Not sure%", "Unsupport%", "n_sessions", "total_n", "mapped"]
        ]
        st.dataframe(table, use_container_width=True, hide_index=True)

with tab_demo:
    st.subheader("Email + Affiliation (Completed only)")

    # session-level (within filter view)
    if len(session_ids_in_view) == 0:
        st.info("No completed sessions in the current filter range.")
    else:
        # Extract columns as arrays to avoid index alignment issues
        sess_for_groupby = pd.DataFrame({
            "session_id": df_items["session_id"].values,
            "email_present": df_items["email_present"].values,
            "affiliation": df_items["affiliation"].values,
            "rating_mode": df_items["rating_mode"].values,
            "utm_source": df_items["utm_source"].values
        })
        
        sess = (
            sess_for_groupby.groupby("session_id", as_index=False)
            .agg(
                email_present=("email_present", "first"),
                affiliation=("affiliation", "first"),
                rating_mode=("rating_mode", "first"),
                utm_source=("utm_source", "first"),
            )
            .reset_index(drop=True)
        )

        completed_n = len(sess)
        email_n = int(sess["email_present"].sum())
        st.metric("Emails provided", f"{email_n:,}", f"{(email_n/completed_n)*100:.1f}% of completed in view")

        st.divider()

        with_email = clean_df(sess[sess["email_present"]])
        if with_email.empty:
            st.info("No email-provided submissions in the current filter range.")
        else:
            # Extract columns as arrays to avoid index alignment issues
            affiliation_vals = with_email["affiliation"].replace("", "Unknown/Blank").values
            session_id_vals = with_email["session_id"].values
            
            aff_for_groupby = pd.DataFrame({
                "affiliation": affiliation_vals,
                "session_id": session_id_vals
            })
            
            aff = (
                aff_for_groupby.groupby("affiliation", as_index=False)
                .agg(n=("session_id", "count"))
                .reset_index(drop=True)
                .sort_values("n", ascending=False)
            )
            fig = px.bar(aff, x="affiliation", y="n")
            st.plotly_chart(fig, use_container_width=True)

            aff["pct"] = (aff["n"] / aff["n"].sum()).values
            aff["pct"] = aff["pct"].map(pct_fmt).values
            st.dataframe(aff.rename(columns={"n": "count"}), use_container_width=True, hide_index=True)

with tab_qa:
    st.subheader("QA / Mapping")
    if df_items.empty:
        st.info("No data in the current filter range.")
    else:
        indiv = clean_df(df_items[df_items["rating_mode"] == "individual"])
        if indiv.empty:
            st.info("No individual-mode items in the current filter range.")
        else:
            total = len(indiv)
            mapped = int((indiv["reform_key"].astype(str).str.strip() != "").sum())
            st.metric("Reform mapping success", f"{mapped:,} / {total:,}", f"{(mapped/total)*100:.1f}%")

            # show top unmapped titles
            unmapped_mask = indiv["reform_key"].astype(str).str.strip() == ""
            unmapped = clean_df(indiv.loc[unmapped_mask])
            if unmapped.empty:
                st.success("No unmapped reform items in the current filter range.")
            else:
                # Extract grouping columns as arrays to avoid index alignment issues
                group_cols = ["category", "proposal_title"]
                group_data = {}
                for col in group_cols:
                    col_data = unmapped[col]
                    if isinstance(col_data.dtype, pd.CategoricalDtype):
                        group_data[col] = col_data.astype(str).values
                    else:
                        group_data[col] = col_data.values
                
                # Create a temporary DataFrame with clean index for grouping
                unmapped_for_groupby = pd.DataFrame(group_data)
                unmapped_for_groupby["session_id"] = unmapped["session_id"].values
                
                u = (
                    unmapped_for_groupby.groupby(group_cols, as_index=False)
                    .agg(n=("session_id", "count"))
                    .reset_index(drop=True)
                    .sort_values("n", ascending=False)
                )
                st.write("Unmapped reform titles (top 50):")
                st.dataframe(u.head(50), use_container_width=True, hide_index=True)

st.caption("Completion gate: only submissions with reflection_q2 filled are included.")