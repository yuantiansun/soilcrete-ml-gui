# -*- coding: utf-8 -*-
"""

conda activate envML
streamlit run E:\Spyderdata\Spyderdata\gui_soilcrete.py

Soilcrete Shear Curve Prediction GUI
Upgraded version with:
 - English interface
 - Premium header & theme
 - CSV export
 - Peak strength identification
 - Refined layout
"""

import numpy as np
import pandas as pd
import joblib
import streamlit as st
import matplotlib.pyplot as plt

# =====================================================
# 0. Streamlit page config (premium theme)
# =====================================================
st.set_page_config(
    page_title="Soilcrete Shear Curve Prediction (ML Model)",
    layout="wide",
    page_icon="📈",
)

# Global style
st.markdown("""
<style>
/* Bigger title */
h1 {
    font-size: 32px !important;
    font-weight: 800 !important;
    color: #1a3e6b !important;
}

/* Section headers */
h2 {
    font-size: 24px !important;
    font-weight: 700 !important;
}

/* Better font for entire app */
html, body, [class*="css"]  {
    font-family: 'Segoe UI', sans-serif;
}
</style>
""", unsafe_allow_html=True)

# =====================================================
# 1. Load trained model
# =====================================================
MODEL_PATH = r"E:\Spyderdata\Spyderdata\soilcrete_gui_model.pkl"
gui_info = joblib.load(MODEL_PATH)

gbr  = gui_info["gbr"]
xgb  = gui_info["xgb"]
lgbm = gui_info["lgbm"]
cat  = gui_info["cat"]
feature_cols = gui_info["feature_cols"]
disp_min, disp_max = gui_info["displacement_range"]
n_points = gui_info["n_points"]

# =====================================================
# Main Title
# =====================================================
st.title("📈 Soilcrete Shear Stress–Displacement Curve Prediction")
st.markdown("""
This web GUI predicts the **shear stress–displacement (τ–δ) curve**  
for Soilcrete under specified curing age, normal stress and mix proportions  
using a **Machine Learning Ensemble Model (GBR + XGB + LGBM + CAT)**  
enhanced with physically-informed synthetic data (Eq.A curves).

---

### **Units**
- **Displacement δ:** mm  
- **Shear Stress τ:** kPa  
- **Normal Stress σₙ:** kPa  
- **Mix proportions:** % by mass  

---
""")

# =====================================================
# Layout
# =====================================================
col_left, col_right = st.columns([1, 2])

# =====================================================
# 2. Input panel
# =====================================================
with col_left:
    st.header("Input Parameters")

    age_d = st.number_input("Curing Age (days)", value=28, min_value=1, max_value=365)
    normal_stress = st.number_input("Normal Stress σₙ (kPa)", value=100, min_value=10, max_value=1000, step=10)

    st.subheader("Mix Proportions (%)")
    cement_pct = st.number_input("Cement (%)", value=40.0, min_value=0.0, max_value=100.0)
    mp_pct     = st.number_input("Mineral Powder (%)", value=40.0, min_value=0.0, max_value=100.0)
    fa_pct     = st.number_input("Fly Ash (%)", value=20.0, min_value=0.0, max_value=100.0)
    gy_pct     = st.number_input("Gypsum (%)", value=5.0, min_value=0.0, max_value=100.0)

    st.subheader("Displacement Settings")
    disp_min_input = st.number_input("Minimum δ (mm)", value=float(disp_min))
    disp_max_input = st.number_input("Maximum δ (mm)", value=float(disp_max))
    n_points_input = st.number_input("Number of points", min_value=20, max_value=400, value=int(n_points), step=10)

    predict_btn = st.button("🔮 Predict Shear Curve")

# =====================================================
# 3. Prediction and visualization
# =====================================================
with col_right:
    st.header("Predicted τ–δ Curve")

    if predict_btn:

        # ---- Build displacement vector ----
        disp = np.linspace(disp_min_input, disp_max_input, int(n_points_input))

        # ---- Build feature matrix ----
        data = pd.DataFrame({
            "age_d":              [age_d] * len(disp),
            "normal_stress_kPa":  [normal_stress] * len(disp),
            "cement_pct":         [cement_pct] * len(disp),
            "mineral_powder_pct": [mp_pct] * len(disp),
            "fly_ash_pct":        [fa_pct] * len(disp),
            "gypsum_pct":         [gy_pct] * len(disp),
            "displacement_mm":    disp,
        })

        X_new = data[feature_cols].values

        # ---- Ensemble prediction ----
        tau_pred = (
            gbr.predict(X_new)
            + xgb.predict(X_new)
            + lgbm.predict(X_new)
            + cat.predict(X_new)
        ) / 4.0

        # ---- Identify Peak Strength ----
        peak_idx = np.argmax(tau_pred)
        peak_disp = disp[peak_idx]
        peak_tau  = tau_pred[peak_idx]

        # ---- Plot Figure ----
        fig, ax = plt.subplots(figsize=(7, 4))
        ax.plot(disp, tau_pred, linewidth=2, label="Predicted Curve", color="#1a76d2")

        # Peak point
        ax.scatter([peak_disp], [peak_tau], color='red', s=60, label=f"Peak: ({peak_disp:.2f} mm, {peak_tau:.1f} kPa)")

        ax.set_xlabel("Displacement δ (mm)")
        ax.set_ylabel("Shear Stress τ (kPa)")
        ax.set_title(f"Predicted τ–δ Curve\nAge: {age_d} d, σₙ: {normal_stress} kPa")
        ax.grid(True)
        ax.legend()

        st.pyplot(fig)

        # ---- CSV Export ----
        export_df = pd.DataFrame({
            "displacement_mm": disp,
            "tau_pred_kPa": tau_pred,
        })

        csv_bytes = export_df.to_csv(index=False).encode("utf-8")
        st.download_button(
            label="📥 Download CSV",
            data=csv_bytes,
            file_name="soilcrete_predicted_curve.csv",
            mime="text/csv",
        )

        # Show preview of data
        st.subheader("Preview of First 10 Points")
        st.dataframe(export_df.head(10))

    else:
        st.info("Enter parameters on the left and click **Predict**.")

