"""
Exoplanet Detection — ML Model Comparison
Streamlit demo UI for COMP702 CA2.

Loads the trained pipelines saved by exoplanet_detection.ipynb (in ./models/)
and the results saved by the results-export cell (in ./results/). Nothing here
is hardcoded — every number and plot comes from your actual notebook run.

Run with:  streamlit run app.py
"""

import os
import joblib
import pandas as pd
import numpy as np
import streamlit as st
import matplotlib.pyplot as plt

st.set_page_config(page_title="Exoplanet Detection", layout="wide")

MODELS_DIR = "models"
RESULTS_DIR = "results"
CHART_FIGSIZE = (4.2, 2.6)   # small, fixed — matches the reference dashboard's compact charts

st.markdown("""
<style>
div[data-testid="stMetric"] { background-color: #F7FAFF; border-radius: 10px; padding: 8px; }
</style>
""", unsafe_allow_html=True)


# ---------------------------------------------------------------------------
# Load saved artefacts
# ---------------------------------------------------------------------------
@st.cache_resource
def load_pipelines():
    pipelines = {}
    name_map = {
        "Logistic Regression": "logistic_regression_tuned_pipeline.joblib",
        "Random Forest": "random_forest_tuned_pipeline.joblib",
        "MLP": "mlp_tuned_pipeline.joblib",
    }
    for display_name, fname in name_map.items():
        path = os.path.join(MODELS_DIR, fname)
        if os.path.exists(path):
            pipelines[display_name] = joblib.load(path)
    return pipelines


@st.cache_data
def load_results_table():
    path = os.path.join(RESULTS_DIR, "model_comparison.csv")
    if os.path.exists(path):
        df = pd.read_csv(path)
        return df[df["Version"] == "Tuned"].reset_index(drop=True)
    return None


def feature_names_for(pipeline):
    """Ground truth for what a pipeline was trained on, read straight off the
    fitted imputer — can't drift out of sync with a separately-saved column
    list, which was the earlier cause of the 69-vs-70 feature error."""
    return list(pipeline.named_steps["impute"].feature_names_in_)


pipelines = load_pipelines()
results_table = load_results_table()

if not pipelines:
    st.error(
        "No trained models found in ./models/. Copy the models/ folder produced by "
        "exoplanet_detection.ipynb into the same directory as this app.py, then rerun."
    )
    st.stop()

default_feature_cols = feature_names_for(next(iter(pipelines.values())))

# ---------------------------------------------------------------------------
# Sidebar
# ---------------------------------------------------------------------------
with st.sidebar:
    st.markdown("### 🪐 Exoplanet Detection")
    st.caption("MSc COMP702 project")
    st.radio("Navigation", ["🏠 Home", "🗄️ Data", "⚙️ Models", "📊 Results", "🔧 Settings"], index=0, label_visibility="collapsed")
    st.markdown("---")
    st.caption("Single-page demo — everything lives in the tabs on the right.")

# ---------------------------------------------------------------------------
# Header
# ---------------------------------------------------------------------------
header_left, header_right = st.columns([5, 1])
with header_left:
    st.markdown("# Exoplanet Detection")
    st.caption("ML Model Comparison")
with header_right:
    with st.popover("ℹ️ About"):
        st.write(
            "COMP702 MSc Project — Automated Exoplanet Detection from Space Telescope "
            "Data Using Machine Learning."
        )
st.markdown("---")

tab_predict, tab_compare = st.tabs(["Predict", "Model Comparison"])

# ---------------------------------------------------------------------------
# PREDICT TAB — 3-column dashboard, matching the reference layout
# ---------------------------------------------------------------------------
with tab_predict:
    left_col, center_col, right_col = st.columns([1, 1, 1.1], gap="medium")

    # ---- Left column: upload, preview, model selection --------------------
    with left_col:
        with st.container(border=True):
            st.markdown("**1. Upload KOI Feature Dataset**")
            uploaded_file = st.file_uploader("Drag and drop CSV file here", type=["csv"], label_visibility="collapsed")
            if uploaded_file is not None:
                st.success(f"{uploaded_file.name}")

        with st.container(border=True):
            st.markdown("**2. Dataset Preview**")
            if uploaded_file is not None:
                input_df = pd.read_csv(uploaded_file, comment="#")
            else:
                st.caption("No file uploaded — using a sample row.")
                sample = {c: np.nan for c in default_feature_cols}
                for c, v in {
                    "koi_period": 10.487, "koi_depth": 512.3, "koi_duration": 2.45,
                    "koi_prad": 2.31, "koi_teq": 905.2,
                }.items():
                    if c in sample:
                        sample[c] = v
                input_df = pd.DataFrame([sample])

            preview_cols = [c for c in default_feature_cols if c in input_df.columns][:5]
            preview_df = input_df[preview_cols].head(5)
            # when using the placeholder sample row, only show columns that
            # actually have a value rather than a row of empty "None" cells
            if uploaded_file is None:
                preview_df = preview_df.dropna(axis=1, how="all")
            st.dataframe(preview_df, height=180)
            n_rows = min(5, len(input_df))
            st.caption(f"Showing first {n_rows} row{'s' if n_rows != 1 else ''} of {len(input_df)} total")

            has_target = "target" in input_df.columns

            if len(input_df) > 1:
                if has_target:
                    st.info(
                        "Evaluation dataset: 20% held-out test set, split by host star "
                        "(KIC ID). These observations were not used to train the final "
                        "models."
                    )
                selected_idx = st.slider(
                    "Test Observation", min_value=1, max_value=len(input_df), value=1
                ) - 1
            else:
                selected_idx = 0

        with st.container(border=True):
            st.markdown("**3. Select Model**")
            model_name = st.selectbox(
                "Model", list(pipelines.keys()),
                index=list(pipelines.keys()).index("Random Forest") if "Random Forest" in pipelines else 0,
                label_visibility="collapsed",
            )
            run_prediction = st.button("Predict", width="stretch", type="primary")

    # ---- Center column: prediction result and headline metrics -----------
    with center_col:
        with st.container(border=True):
            st.markdown("**4. Prediction Result**")

            if run_prediction:
                pipeline = pipelines[model_name]
                model_feature_cols = feature_names_for(pipeline)

                row = input_df.iloc[[selected_idx]]
                # target and kepid must never reach the model as input features —
                # target is the label itself, kepid is only a grouping identifier.
                # Dropping them explicitly here (rather than relying on them simply
                # not being in model_feature_cols) makes that guarantee visible.
                feature_row = row.drop(columns=[c for c in ["target", "kepid"] if c in row.columns])

                aligned = pd.DataFrame(columns=model_feature_cols)
                aligned = pd.concat([aligned, feature_row[[c for c in model_feature_cols if c in feature_row.columns]]])
                for c in model_feature_cols:
                    if c not in aligned.columns:
                        aligned[c] = np.nan
                aligned = aligned[model_feature_cols]

                pred = pipeline.predict(aligned)[0]

                # predict_proba can fail on a model pickled with a newer
                # scikit-learn than the one currently installed (this
                # notebook used 1.9.0; if your local sklearn is older,
                # LogisticRegression.predict_proba() may reference an
                # attribute that no longer exists). Fall back to
                # decision_function + a sigmoid transform for the binary
                # case so the app still works either way — but note when
                # that fallback was used, since it's an approximation.
                confidence_is_approximate = False
                try:
                    proba = pipeline.predict_proba(aligned)[0]
                    confidence = proba[pred] * 100
                except AttributeError:
                    confidence_is_approximate = True
                    decision_score = pipeline.decision_function(aligned)[0]
                    proba_positive = 1 / (1 + np.exp(-decision_score))
                    confidence = (proba_positive if pred == 1 else 1 - proba_positive) * 100

                if pred == 1:
                    st.success("🪐 **Planet Candidate**  \n(Exoplanet)")
                else:
                    st.error("❌ **False Positive**  \n(Not an exoplanet)")

                c1, c2 = st.columns(2)
                if confidence_is_approximate:
                    c1.metric("Confidence (approx.)", f"{confidence:.1f}%")
                else:
                    c1.metric("Confidence", f"{confidence:.1f}%")
                c2.metric("Model Used", model_name)

                if confidence_is_approximate:
                    st.caption(
                        "⚠️ Estimated from decision_function — predict_proba was "
                        "unavailable due to a scikit-learn version mismatch between "
                        "training and this environment. Run `pip install --upgrade "
                        "scikit-learn` to resolve."
                    )

                if len(input_df) > 1:
                    st.caption(f"Observation {selected_idx + 1} of {len(input_df)}")

                if has_target:
                    actual = int(row["target"].iloc[0])
                    actual_label = (
                        "🪐 Planet Candidate (Exoplanet)" if actual == 1
                        else "❌ False Positive (Not an exoplanet)"
                    )
                    st.markdown(f"**Actual label:** {actual_label}")
                    if actual == pred:
                        st.success("✅ Correct prediction")
                    else:
                        st.error("❌ Incorrect prediction")
            else:
                st.info("Upload data (or use the sample) and click Predict.")

        if results_table is not None:
            with st.container(border=True):
                st.markdown("**Evaluation (on test set)**")
                row = results_table[results_table["Model"] == model_name]
                if not row.empty:
                    r = row.iloc[0]
                    m1, m2, m3, m4 = st.columns(4)
                    m1.metric("Precision", f"{r['Precision']:.2f}")
                    m2.metric("Recall", f"{r['Recall']:.2f}")
                    m3.metric("F1", f"{r['F1-score']:.2f}")
                    m4.metric("ROC-AUC", f"{r['ROC-AUC']:.2f}")

    # ---- Right column: model comparison table ----------------------------
    with right_col:
        if results_table is not None:
            with st.container(border=True):
                st.markdown("**5. Model Comparison Results**")
                display_table = results_table[["Model", "Precision", "Recall", "F1-score", "ROC-AUC"]]
                st.dataframe(display_table, hide_index=True, height=140)

                best_row = results_table.sort_values("F1-score", ascending=False).iloc[0]
                st.success(
                    f"🏆 **Best Performing Model**  \n"
                    f"**{best_row['Model']}**  \n"
                    f"Highest F1-score ({best_row['F1-score']:.2f})"
                )

    # ---- Chart row: full width so the three columns above end level ------
    st.write("")
    chart_left, chart_center, chart_right = st.columns(3, gap="medium")

    with chart_left:
        with st.container(border=True):
            st.markdown("**Top Important Features**")
            chosen_pipeline = pipelines.get(model_name)
            if chosen_pipeline is not None:
                chosen_feature_cols = feature_names_for(chosen_pipeline)

                if model_name == "Random Forest":
                    importances = pd.Series(
                        chosen_pipeline.named_steps["model"].feature_importances_,
                        index=chosen_feature_cols,
                    ).sort_values(ascending=False).head(5)
                    fig, ax = plt.subplots(figsize=CHART_FIGSIZE)
                    importances.iloc[::-1].plot(kind="barh", ax=ax, color="#2563EB")
                    ax.set_xlabel("Feature Importance", fontsize=8)
                    ax.tick_params(labelsize=7)
                    fig.tight_layout()
                    st.pyplot(fig, width="stretch")

                elif model_name == "Logistic Regression":
                    coef = pd.Series(
                        chosen_pipeline.named_steps["model"].coef_[0],
                        index=chosen_feature_cols,
                    ).sort_values(key=abs, ascending=False).head(5)
                    fig, ax = plt.subplots(figsize=CHART_FIGSIZE)
                    colors = ["#2563EB" if v > 0 else "#DC2626" for v in coef.iloc[::-1]]
                    coef.iloc[::-1].plot(kind="barh", ax=ax, color=colors)
                    ax.set_xlabel("Coefficient", fontsize=8)
                    ax.tick_params(labelsize=7)
                    fig.tight_layout()
                    st.pyplot(fig, width="stretch")
                else:
                    st.caption("Available for Random Forest and Logistic Regression.")

    with chart_center:
        roc_path = os.path.join(RESULTS_DIR, "outputs_roc_curves.png")
        if os.path.exists(roc_path):
            with st.container(border=True):
                st.markdown("**ROC Curves**")
                st.image(roc_path, width="stretch")

    with chart_right:
        if results_table is not None:
            with st.container(border=True):
                st.markdown("**F1-score Comparison**")
                fig, ax = plt.subplots(figsize=CHART_FIGSIZE)
                colors = ["#F59E0B", "#16A34A", "#2563EB"][:len(results_table)]
                ax.bar(results_table["Model"], results_table["F1-score"], color=colors)
                ax.set_ylim(0, 1)
                ax.set_ylabel("F1-score", fontsize=8)
                ax.tick_params(labelsize=7)
                plt.xticks(rotation=15)
                fig.tight_layout()
                st.pyplot(fig, width="stretch")

    st.caption(
        "Results are based on the held-out test set. Metrics may vary with "
        "different datasets and preprocessing choices."
    )

# ---------------------------------------------------------------------------
# MODEL COMPARISON TAB
# ---------------------------------------------------------------------------
with tab_compare:
    if results_table is not None:
        with st.container(border=True):
            st.markdown("**Full Model Comparison**")
            st.dataframe(results_table, hide_index=True)

        cm_path = os.path.join(RESULTS_DIR, "outputs_confusion_matrices.png")
        if os.path.exists(cm_path):
            with st.container(border=True):
                st.markdown("**Confusion Matrices**")
                st.image(cm_path, width=700)

        corr_path = os.path.join(RESULTS_DIR, "outputs_correlation.png")
        if os.path.exists(corr_path):
            with st.container(border=True):
                st.markdown("**Feature Correlation (training set)**")
                st.image(corr_path, width=500)
    else:
        st.warning("results/model_comparison.csv not found — copy your results/ folder here.")

st.markdown("---")
foot_left, foot_right = st.columns(2)
foot_left.caption("Data Source: NASA Exoplanet Archive (KOI Dataset)")
foot_right.caption("Exoplanet Detection ML Model Comparison")
