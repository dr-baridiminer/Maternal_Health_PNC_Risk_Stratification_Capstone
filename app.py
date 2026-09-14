"""Local, identifier-free PNC decision-support prototype."""

import json
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
import streamlit as st


ROOT = Path(__file__).resolve().parent
NUMERIC = {
    "maternal_age": ("Maternal age (years)", 15, 49),
    "children_ever_born": ("Children ever born", 0, None),
    "first_anc_month": ("Month of pregnancy at first antenatal care (ANC) visit", 1, 10),
    "anc_visits": ("Number of antenatal care (ANC) visits", 0, 20),
}
LABELS = {
    "county_name": "County",
    "residence": "Residence",
    "education_level": "Highest education level",
    "wealth_quintile": "Household wealth quintile",
    "marital_status": "Marital status",
    "currently_working": "Currently working",
    "pregnancy_intention": "Pregnancy intention",
    "delivery_place": "Place of delivery",
    "caesarean_delivery": "Caesarean delivery",
}
OPTIONAL = {"first_anc_month", "anc_visits"}


@st.cache_resource
def load_artifacts():
    # Load only the project's trusted, local serialized pipeline.
    metadata = json.loads((ROOT / "models/pnc_risk_model_metadata.json").read_text())
    pipeline = joblib.load(ROOT / "models/pnc_risk_pipeline.joblib")
    predictors = metadata["predictors"]
    if (len(predictors) != 13 or set(predictors) != set(NUMERIC) | set(LABELS)
            or list(pipeline.feature_names_in_) != predictors):
        raise ValueError("Incompatible predictor schema")
    threshold = float(metadata["decision_threshold"])
    if not 0 <= threshold <= 1 or list(pipeline.classes_) != [0, 1]:
        raise ValueError("Incompatible classification metadata")
    preprocessing = pipeline.named_steps["preprocessing"]
    encoder = preprocessing.named_transformers_["categorical"].named_steps["encoder"]
    columns = next(cols for name, _, cols in preprocessing.transformers_ if name == "categorical")
    categories = dict(zip(columns, [values.tolist() for values in encoder.categories_]))
    return pipeline, metadata, categories


def predict(pipeline, metadata, values):
    """Pass exactly the saved feature schema through the fitted preprocessing."""
    frame = pd.DataFrame([values], columns=metadata["predictors"])
    probability = float(pipeline.predict_proba(frame)[0, list(pipeline.classes_).index(1)])
    if not np.isfinite(probability) or not 0 <= probability <= 1:
        raise ValueError("Invalid model output")
    return probability, probability >= metadata["decision_threshold"]


def main():
    st.set_page_config(page_title="PNC decision support", page_icon="📋", layout="centered")
    st.title("Postnatal care decision support")
    st.write("Estimate the risk of missing timely postnatal care (PNC) within 48 hours after birth.")
    st.warning("Research prototype — decision support, not a clinical diagnosis. "
               "Use alongside professional judgment; do not use this score alone to make care decisions.")
    st.caption("Enter only the requested predictors. No names, DHS identifiers, household IDs, "
               "or contact details are requested. This app does not write form entries or predictions to disk.")
    try:
        pipeline, metadata, categories = load_artifacts()
    except Exception:
        st.error("The saved model could not be loaded. Check the two model files and install "
                 "the compatible dependencies listed in the README.")
        st.stop()

    st.write("Complete the form after delivery details are known. Maternal age must be 15–49 years, "
             "matching the project population. All fields are required except the two ANC fields.")
    st.caption("Leave an ANC field blank if unknown; leave first ANC month blank if no ANC was received. "
               "The saved pipeline imputes missing values. Enter 0 visits when no ANC was received. "
               "Wealth quintile is the survey-derived household wealth group, not an income amount; "
               "do not guess it if unavailable.")
    with st.form("pnc_predictors"):
        values = {}
        for field in metadata["predictors"]:
            if field in NUMERIC:
                label, minimum, maximum = NUMERIC[field]
                values[field] = st.number_input(
                    label + (" (optional)" if field in OPTIONAL else ""),
                    min_value=minimum, max_value=maximum, value=None, step=1, key=field,
                )
            else:
                values[field] = st.selectbox(LABELS[field], categories[field], index=None,
                                             placeholder="Select an option", key=field)
        submitted = st.form_submit_button("Estimate missed-timely-PNC risk")

    if submitted:
        missing = [NUMERIC[f][0] if f in NUMERIC else LABELS[f]
                   for f, v in values.items() if v is None and f not in OPTIONAL]
        if missing:
            st.error("Complete the required fields: " + "; ".join(missing) + ".")
            return
        if values["anc_visits"] == 0 and values["first_anc_month"] is not None:
            st.error("For zero ANC visits, leave the first ANC month blank.")
            return
        imputed = [NUMERIC[f][0] for f in OPTIONAL if values[f] is None]
        values = {f: np.nan if v is None else v for f, v in values.items()}
        try:
            probability, flagged = predict(pipeline, metadata, values)
        except Exception:
            st.error("A risk estimate could not be calculated. Check the model environment and inputs.")
            return
        st.subheader("Result for the last submitted form")
        st.metric("Predicted missed-timely-PNC risk", f"{probability:.2%}")
        st.write(f"**Threshold-based flag: {'Flagged for follow-up review' if flagged else 'Below the follow-up threshold'}**")
        st.caption(f"Flagged when the unrounded probability is greater than or equal to "
                   f"{metadata['decision_threshold']:.6%} (saved model threshold).")
        if imputed:
            st.info("The pipeline imputed missing values for: " + "; ".join(imputed) + ".")
        st.write("This is a model estimate of missed care, not a diagnosis or a prediction of a medical "
                 "complication. A flag can inform review of follow-up needs. A result below the threshold "
                 "does not guarantee timely care or justify withholding care. Estimates may be wrong "
                 "and have not been established here as suitable for clinical deployment.")
        st.caption("Results reflect the last submission. Submit again after editing any inputs.")


if __name__ == "__main__":
    main()
