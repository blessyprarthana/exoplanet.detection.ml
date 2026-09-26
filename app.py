"""
Exoplanet Detection - ML Model Comparison
Streamlit demo UI for the COMP702 MSc project.

Reads the pipelines and results that exoplanet_detection.ipynb writes out into
models/ and results/, so nothing in here is hardcoded. If the numbers on screen
look wrong, rerun the notebook rather than editing this file.

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
CHART_FIGSIZE = (3.4, 1.9)   # small and fixed, so the dashboard fits one screen

# I wanted the whole dashboard to sit in one viewport, so this tightens
# Streamlit's default spacing: less padding, tighter cards, smaller gaps.
st.markdown("""
<style>
.block-container { padding-top: 1.2rem; padding-bottom: 0.5rem; }
div[data-testid="stMetric"] { background-color: #F7FAFF; border-radius: 8px; padding: 4px 8px; }
div[data-testid="stMetricValue"] { font-size: 1.25rem; }
div[data-testid="stMetricLabel"] p { font-size: 0.72rem; }
div[data-testid="stVerticalBlockBorderWrapper"] { padding: 0.35rem 0.6rem; }
div[data-testid="stVerticalBlock"] { gap: 0.5rem; }
div[data-testid="stFileUploaderDropzone"] { padding: 0.6rem 1rem; min-height: 0; }
div[data-testid="stFileUploaderDropzone"] small { display: none; }
h1 { font-size: 1.9rem; margin-bottom: 0; }

/* Charts and the ROC image are width-driven, so on a wide monitor they would
   grow tall enough to push the page past one screen. Cap their height and let
   the width follow, so the layout is stable at any window size. */
div[data-testid="stImage"] { display: flex; justify-content: center; }
div[data-testid="stImage"] img { max-height: 195px; width: auto !important; }

/* ---- Sidebar nav rail, styled after the CA1 interface mockup ---------- */
section[data-testid="stSidebar"] { background-color: #F7FAFF; border-right: 1px solid #E6EDF8; }
section[data-testid="stSidebar"] .block-container { padding-top: 1.6rem; }
.sidebar-brand { font-size: 1.05rem; font-weight: 700; color: #1E3A8A; margin-bottom: 0.1rem; }
.sidebar-sub { font-size: 0.78rem; color: #64748B; margin-bottom: 1.1rem; }
.sidebar-note { font-size: 0.72rem; color: #94A3B8; line-height: 1.35;
                margin-top: 1.1rem; padding-top: 0.9rem; border-top: 1px solid #E6EDF8; }

/* the radio becomes a stack of full-width nav pills */
section[data-testid="stSidebar"] div[role="radiogroup"] { gap: 0.15rem; width: 100%; }
section[data-testid="stSidebar"] div[data-testid="stRadio"] { width: 100%; }
/* the element wrapper shrink-wraps by default, which would leave the pills
   only as wide as their text */
section[data-testid="stSidebar"] div[data-testid="stElementContainer"] { width: 100%; align-self: stretch; }
section[data-testid="stSidebar"] div[role="radiogroup"] > label {
    display: flex; align-items: center; width: 100%;
    padding: 0.42rem 0.7rem; border-radius: 8px; cursor: pointer;
    transition: background-color 0.12s ease;
}
/* Structure is  label > div > div > [circle, textwrapper], so hide the circle
   and let the row fill the rail width. */
section[data-testid="stSidebar"] div[role="radiogroup"] > label > div { width: 100%; }
section[data-testid="stSidebar"] div[role="radiogroup"] > label > div > div { width: 100%; align-items: center; }
section[data-testid="stSidebar"] div[role="radiogroup"] > label > div > div > div:first-child { display: none; }
section[data-testid="stSidebar"] div[role="radiogroup"] > label p {
    font-size: 0.88rem; font-weight: 500; color: #334155; margin: 0;
}
section[data-testid="stSidebar"] div[role="radiogroup"] > label:hover { background-color: #E8F0FE; }
section[data-testid="stSidebar"] div[role="radiogroup"] > label[data-selected="true"],
section[data-testid="stSidebar"] div[role="radiogroup"] > label:has(input:checked) {
    background-color: #2563EB;
}
section[data-testid="stSidebar"] div[role="radiogroup"] > label[data-selected="true"] p,
section[data-testid="stSidebar"] div[role="radiogroup"] > label:has(input:checked) p {
    color: #FFFFFF; font-weight: 600;
}
</style>
""", unsafe_allow_html=True)


# ---- loading the saved models and results ----
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
    """What the pipeline was actually trained on, read off the fitted imputer.
    A separately-saved column list can drift out of sync, which is what caused
    my 69-vs-70 feature error earlier on."""
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

# ---- sidebar ----
with st.sidebar:
    st.markdown('<div class="sidebar-brand">🪐 Exoplanet Detection</div>', unsafe_allow_html=True)
    st.markdown('<div class="sidebar-sub">MSc COMP702 project</div>', unsafe_allow_html=True)
    # Still a radio underneath so the highlight follows the click, but the
    # circles are hidden in CSS so it reads as the nav rail from my mockup.
    # Note this is decorative at the moment: nothing reads st.session_state.nav,
    # the tabs below do the actual switching. Left in because the mockup has it.
    st.radio(
        "Navigation",
        ["Home", "Data", "Models", "Results", "Settings"],
        index=0,
        label_visibility="collapsed",
        captions=None,
        key="nav",
    )
    st.markdown('<div class="sidebar-note">Single-page demo, everything lives in the tabs on the right.</div>',
                unsafe_allow_html=True)

# ---- page header ----
header_left, header_right = st.columns([5, 1])
with header_left:
    st.markdown("# Exoplanet Detection")
    st.caption("ML Model Comparison")
with header_right:
    with st.popover("ℹ️ About"):
        st.write(
            "COMP702 MSc Project: Automated Exoplanet Detection from Space Telescope "
            "Data Using Machine Learning."
        )
st.markdown("---")

tab_predict, tab_compare = st.tabs(["Predict", "Model Comparison"])

# ---- Predict tab: 3 columns, laid out like the mockup ----
with tab_predict:
    left_col, center_col, right_col = st.columns([1, 1, 1.15], gap="small")

    # left column: upload, preview, pick a model
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
                st.caption("No file uploaded, so this is a sample row.")
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
            # otherwise the sample row shows as a line of empty None cells
            if uploaded_file is None:
                preview_df = preview_df.dropna(axis=1, how="all")
            st.dataframe(preview_df, height=130)
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

    # centre column: the prediction itself, plus metrics and importances
    with center_col:
        with st.container(border=True):
            st.markdown("**4. Prediction Result**")

            if run_prediction:
                pipeline = pipelines[model_name]
                model_feature_cols = feature_names_for(pipeline)

                row = input_df.iloc[[selected_idx]]
                # target and kepid must never reach the model: target is the
                # label itself and kepid is only a grouping key. I drop them
                # explicitly rather than trusting model_feature_cols to omit them.
                feature_row = row.drop(columns=[c for c in ["target", "kepid"] if c in row.columns])

                aligned = pd.DataFrame(columns=model_feature_cols)
                aligned = pd.concat([aligned, feature_row[[c for c in model_feature_cols if c in feature_row.columns]]])
                for c in model_feature_cols:
                    if c not in aligned.columns:
                        aligned[c] = np.nan
                aligned = aligned[model_feature_cols]

                pred = pipeline.predict(aligned)[0]

                # predict_proba can fail on a model pickled with a newer
                # scikit-learn than the one installed here (I trained
                # on 1.9.0). Falling back to decision_function + a sigmoid
                # keeps the app working, but it is an approximation, so I
                # flag it in the UI rather than passing it off as a real
                # probability.
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
                        "⚠️ Estimated from decision_function, because predict_proba was "
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

    # right column: comparison table, then ROC and F1 side by side
    with right_col:
        if results_table is not None:
            with st.container(border=True):
                st.markdown("**5. Model Comparison Results**")
                display_table = results_table[["Model", "Precision", "Recall", "F1-score", "ROC-AUC"]]
                st.dataframe(display_table, hide_index=True, height=145)

                best_row = results_table.sort_values("F1-score", ascending=False).iloc[0]
                st.success(
                    f"🏆 **Best Performing Model**  \n"
                    f"**{best_row['Model']}**  \n"
                    f"Highest F1-score ({best_row['F1-score']:.2f})"
                )

        roc_col, f1_col = st.columns(2, gap="small")

        with roc_col:
                roc_path = os.path.join(RESULTS_DIR, "outputs_roc_curves.png")
                if os.path.exists(roc_path):
                    with st.container(border=True):
                        st.markdown("**ROC Curves**")
                        st.image(roc_path, width="stretch")

        with f1_col:
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


# ---- Model comparison tab ----
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
        st.warning("results/model_comparison.csv not found. Copy the results/ folder here.")

st.markdown("---")
foot_left, foot_right = st.columns(2)
foot_left.caption("Data Source: NASA Exoplanet Archive (KOI Dataset)")
foot_right.caption("Exoplanet Detection ML Model Comparison")
