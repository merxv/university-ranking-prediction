"""Streamlit dashboard for URPS. Start with ``urps app`` or ``streamlit run src/urps/dashboard.py``.

The UI is a thin presentation layer: all computations live in ``urps.analysis``.
"""

from __future__ import annotations

import hashlib
import io

import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st

from urps.analysis import INDICATOR_LABELS, NOTE, Bundle, build_bundle, feature_label, latest_indicators, what_if
from urps.data import PILLARS, clean, load_raw
from urps.synthetic import generate_the_table

MAX_COMPARE = 8  # more lines than this make the trend chart unreadable


@st.cache_data(show_spinner="Cleaning and validating data...")
def load_from_bytes(content: bytes) -> pd.DataFrame:
    return clean(load_raw(io.BytesIO(content)))


@st.cache_data(show_spinner="Generating the synthetic sample...")
def load_sample() -> pd.DataFrame:
    return clean(generate_the_table())


@st.cache_resource(show_spinner="Training and evaluating models...")
def get_bundle(data_key: str, _df: pd.DataFrame) -> Bundle:
    return build_bundle(_df)


def sidebar() -> tuple[pd.DataFrame, str] | None:
    st.sidebar.header("Data")
    source = st.sidebar.radio("Data source", ["Synthetic sample", "Upload CSV"], index=0)
    st.sidebar.caption(
        "CSV in the Times Higher Education layout: world_rank, university_name, country, teaching, "
        "international, research, citations, income, total_score, num_students, student_staff_ratio, "
        "international_students, year. Extra columns are ignored."
    )
    if source == "Synthetic sample":
        return load_sample(), "sample"
    uploaded = st.sidebar.file_uploader("Ranking CSV", type="csv")
    if uploaded is None:
        st.info("Upload a CSV file in the sidebar, or switch back to the synthetic sample.")
        return None
    content = uploaded.getvalue()
    try:
        df = load_from_bytes(content)
    except Exception as exc:  # show validation problems to the user instead of a traceback
        st.error(f"The file could not be used: {exc}")
        return None
    return df, hashlib.sha256(content).hexdigest()


def tab_trends(df: pd.DataFrame) -> None:
    st.subheader("Historical ranking trends (US1)")
    latest_year = int(df["year"].max())
    latest = (
        df[df["year"] == latest_year]
        .sort_values("total_score", ascending=False)[["university", "country", "total_score"]]
        .rename(columns={"total_score": f"score {latest_year}"})
        .reset_index(drop=True)
    )

    # A table with row checkboxes instead of a multiselect: with hundreds of universities a dropdown
    # stays open while picking and is hard to dismiss; the table also shows who is at the top.
    left, right = st.columns([2, 3])
    with left:
        st.caption(f"Tick up to {MAX_COMPARE} universities. Use the search icon in the table toolbar to find one.")
        event = st.dataframe(
            latest,
            hide_index=True,
            height=420,
            on_select="rerun",
            selection_mode="multi-row",
            selection_default={"selection": {"rows": [0, 1, 2]}},
            key="trend_select",
        )
    rows = list(event.selection.rows)
    chosen = latest.loc[rows, "university"].tolist()
    with right:
        indicator = st.selectbox(
            "Indicator", list(INDICATOR_LABELS), format_func=lambda c: INDICATOR_LABELS[c], key="trend_indicator"
        )
        if not chosen:
            st.info("Tick at least one university in the table.")
            return
        if len(chosen) > MAX_COMPARE:
            st.warning(f"Showing the first {MAX_COMPARE} of {len(chosen)} selected universities.")
            chosen = chosen[:MAX_COMPARE]
        data = df[df["university"].isin(chosen)].sort_values("year")
        fig = px.line(
            data,
            x="year",
            y=indicator,
            color="university",
            markers=True,
            labels={indicator: INDICATOR_LABELS[indicator]},
        )
        fig.update_layout(xaxis=dict(dtick=1), legend=dict(orientation="h", y=-0.25, title_text=""))
        st.plotly_chart(fig, width="stretch")

    table = data.assign(
        world_rank=data.apply(
            lambda r: (
                str(r["rank_lower"]) if r["rank_lower"] == r["rank_upper"] else f"{r['rank_lower']}-{r['rank_upper']}"
            ),
            axis=1,
        )
    )[["university", "year", "world_rank", "total_score", *PILLARS]]
    st.dataframe(table, hide_index=True, width="stretch", placeholder="-")


def tab_forecast(bundle: Bundle) -> None:
    df = bundle.clean
    next_year = int(df["year"].max()) + 1
    st.subheader(f"Forecast for {next_year} and what-if scenarios (US2)")
    options = bundle.forecast["university"].tolist()
    uni = st.selectbox("University", options, key="forecast_uni")
    row = bundle.forecast.set_index("university").loc[uni]

    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Predicted score", f"{row['predicted_score']:.1f}")
    c2.metric("90% interval", f"{row['lower']:.1f} - {row['upper']:.1f}")
    c3.metric("Predicted rank", str(row["predicted_rank"]))
    c4.metric("Model", bundle.model.name)

    hist = df[df["university"] == uni].sort_values("year")
    fig = go.Figure()
    fig.add_scatter(x=hist["year"], y=hist["total_score"], mode="lines+markers", name="Published score")
    fig.add_scatter(
        x=[next_year],
        y=[row["predicted_score"]],
        mode="markers",
        marker=dict(size=12, symbol="diamond"),
        error_y=dict(
            type="data",
            symmetric=False,
            array=[row["upper"] - row["predicted_score"]],
            arrayminus=[row["predicted_score"] - row["lower"]],
        ),
        name="Forecast (90% interval)",
    )
    fig.update_layout(xaxis=dict(dtick=1), yaxis_title="Overall score")
    st.plotly_chart(fig, width="stretch")

    st.markdown("#### What-if: change this year's indicators")
    st.caption("Move the sliders to see how the forecast reacts if the latest published indicators had been different.")
    latest = latest_indicators(df, uni)
    changes: dict[str, float] = {}
    cols = st.columns(len(PILLARS))
    for col, pillar in zip(cols, PILLARS, strict=True):
        current = latest[pillar]
        if pd.isna(current):
            col.caption(f"{INDICATOR_LABELS[pillar]}: not published")
            continue
        value = col.slider(INDICATOR_LABELS[pillar], 0.0, 100.0, float(current), 0.5, key=f"wi_{uni}_{pillar}")
        if abs(value - float(current)) > 1e-9:
            changes[pillar] = value
    if changes:
        result = what_if(bundle, uni, changes)
        s1, s2 = st.columns(2)
        s1.metric("Scenario score", f"{result['scenario_score']:.1f}", f"{result['change']:+.2f}")
        s2.metric("Scenario rank", str(result["scenario_rank"]), f"was {result['base_rank']}", delta_color="off")
    else:
        st.caption("No changes yet: the scenario equals the base forecast.")
    st.info(NOTE)


def tab_influence(bundle: Bundle) -> None:
    st.subheader("Indicator influence (US3)")
    if bundle.model.name == "persistence":
        st.info(
            "No model beat the naive forecast on the held-out year, so the forecast simply repeats each "
            "university's position from last year and no indicator influence can be estimated."
        )
        return
    top = bundle.importance.head(12).rename(index=feature_label).sort_values()
    fig = px.bar(
        x=top.values,
        y=top.index,
        orientation="h",
        labels={"x": "Increase in MAE when the feature is shuffled", "y": ""},
    )
    st.plotly_chart(fig, width="stretch")
    st.caption(
        "Permutation importance on the held-out year: how much worse the prediction gets when one feature is "
        "randomly shuffled. Correlated features (e.g. last year's score and its historical mean) share influence, "
        "so read the chart as a ranking, not as effect sizes. It shows correlation, not causation."
    )


def tab_evaluation(bundle: Bundle) -> None:
    st.subheader(f"Model evaluation on the held-out year {bundle.test_year}")
    metrics = pd.DataFrame(bundle.metrics).T.rename(
        columns={
            "mae": "MAE",
            "rmse": "RMSE",
            "r2": "R²",
            "r2_change": "R² of yearly change",
            "rank_mae": "Mean rank error",
            "spearman": "Rank correlation",
        }
    )[["MAE", "RMSE", "Mean rank error", "Rank correlation", "R²", "R² of yearly change"]]
    metrics.index.name = "model"
    st.dataframe(metrics, width="stretch")
    best = bundle.model.name
    base = bundle.metrics["persistence"]["mae"]
    if best == "persistence":
        st.warning(
            "No model beat the naive forecast ('same position as last year') on the held-out year, so the "
            "dashboard uses the naive forecast. This is common with few ranking years: positions are very stable "
            "and the remaining changes are mostly noise or methodology changes."
        )
    else:
        st.caption(
            f"Selected model: **{best}**. It improves MAE by {100 * (base - bundle.metrics[best]['mae']) / base:.1f}% "
            f"over the naive forecast ('same position as last year'). Training uses only years before "
            f"{bundle.test_year}."
        )
    with st.expander("How to read these numbers"):
        st.markdown(
            "All scores are measured **relative to the average of their ranking year**. Ranking agencies rescale "
            "scores between editions, which moves the whole scale by several points; that shift is unpredictable "
            "and does not change anyone's position, so it is removed before training and evaluation.\n\n"
            "**Mean rank error** is how many places the predicted position is off on average; **rank "
            "correlation** (Spearman) compares the predicted and actual order.\n\n"
            "**R²** is close to 1 even for the naive forecast, because universities differ far more from each "
            "other than from one year to the next. **R² of yearly change** is the share of the actual change "
            "that a model explains: about 0 for the naive forecast, negative when a model is worse than it."
        )
    tp = bundle.test_predictions
    fig = px.scatter(
        tp,
        x="target",
        y="predicted_score",
        hover_name="university",
        labels={
            "target": "Actual score (relative to year average)",
            "predicted_score": "Predicted score (relative to year average)",
        },
    )
    lo, hi = float(tp["target"].min()), float(tp["target"].max())
    fig.add_shape(type="line", x0=lo, y0=lo, x1=hi, y1=hi, line=dict(dash="dash", color="grey"))
    st.plotly_chart(fig, width="stretch")
    st.download_button(
        "Download forecast (CSV)",
        bundle.forecast.to_csv(index=False).encode("utf-8"),
        file_name="urps_forecast.csv",
        mime="text/csv",
    )


def main() -> None:
    st.set_page_config(page_title="University Ranking Prediction", page_icon=":mortar_board:", layout="wide")
    st.title("University Ranking Prediction System")
    st.caption(NOTE)
    loaded = sidebar()
    if loaded is None:
        return
    df, key = loaded
    try:
        bundle = get_bundle(key, df)
    except ValueError as exc:
        st.error(f"Not enough history to train and evaluate: {exc}")
        return

    a, b, c = st.columns(3)
    a.metric("Universities", df["university"].nunique())
    b.metric("Ranking years", f"{df['year'].min()} - {df['year'].max()}")
    c.metric("Rows after cleaning", len(df))

    t1, t2, t3, t4 = st.tabs(["Trends", "Forecast & what-if", "Indicator influence", "Model evaluation"])
    with t1:
        tab_trends(df)
    with t2:
        tab_forecast(bundle)
    with t3:
        tab_influence(bundle)
    with t4:
        tab_evaluation(bundle)


main()
