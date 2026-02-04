import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from scipy.interpolate import UnivariateSpline
import pdfplumber

# --- 1. GEOTECHNICAL MATH ENGINE ---

def calculate_curvature(log_pressure, void_ratio, smoothing_factor=0.5):
    """
    Calculates the curvature (k) of the e-log(p) curve using a spline.
    Includes safety checks for short data arrays.
    """
    # Safety Check: Need at least 5 points for a cubic spline
    if len(log_pressure) < 5:
        return 2, None # Return a default index (middle of small set) if data is too short

    try:
        # Fit spline
        spline = UnivariateSpline(log_pressure, void_ratio, k=3, s=smoothing_factor)
        
        # Derivatives
        y_d1 = spline.derivative(n=1)(log_pressure)
        y_d2 = spline.derivative(n=2)(log_pressure)
        
        # Curvature formula
        curvature = np.abs(y_d2) / (1 + y_d1**2)**1.5
        
        # Limit search to the loading portion (usually first 70%)
        # Ensure we have a valid slice
        search_limit = int(len(curvature) * 0.7)
        if search_limit < 2: 
            search_limit = len(curvature) - 1
            
        # Find max curvature index (skipping the very first point to avoid noise)
        # Handle cases where slice might be empty
        search_window = curvature[1:search_limit]
        if len(search_window) == 0:
            return 2, spline
            
        max_k_idx = np.argmax(search_window) + 1
        return max_k_idx, spline
        
    except Exception:
        # Fallback if spline fails
        return int(len(log_pressure)/2), None

def find_linear_portion(log_pressure, void_ratio):
    """Identifies the Virgin Compression Line (VCL)."""
    # Use last 3 points if available, otherwise use all
    points_to_take = min(len(log_pressure), 4)
    x_vcl = log_pressure[-points_to_take:]
    y_vcl = void_ratio[-points_to_take:]
    
    if len(x_vcl) < 2:
        return 0, 0 # Cannot calculate slope
        
    slope, intercept = np.polyfit(x_vcl, y_vcl, 1)
    return slope, intercept

def get_intersection(m1, b1, m2, b2):
    """Finds intersection (x, y) of two lines."""
    if m1 == m2: return None, None
    x = (b2 - b1) / (m1 - m2)
    y = m1 * x + b1
    return x, y

def extract_from_pdf(uploaded_file):
    """Extract table from PDF with aggressive cleaning."""
    try:
        with pdfplumber.open(uploaded_file) as pdf:
            first_page = pdf.pages[0]
            table = first_page.extract_table()
            
            if table:
                df = pd.DataFrame(table[1:], columns=table[0])
                
                # Force columns to numeric, coercing errors to NaN
                # This fixes "dirty" cells that might have text
                df = df.apply(pd.to_numeric, errors='coerce')
                
                # Drop rows where ALL columns are NaN
                df = df.dropna(how='all')
                
                # Drop rows where ANY key column (Pressure/Void) is NaN
                df = df.dropna()
                
                return df
            else:
                return None
    except Exception:
        return None

# --- 2. STREAMLIT UI ---

st.set_page_config(page_title="GeoConsolidation AI", layout="wide")
st.title("1-D Consolidation Analysis")
st.markdown("Automated **Casagrande Construction** & **Schmertmann Correction**.")

# --- SIDEBAR: UNITS & INPUTS ---
st.sidebar.header("1. Project Settings")
unit_system = st.sidebar.selectbox("Unit System", ["Metric (SI)", "Imperial (tsf)", "Imperial (psf)"])

if unit_system == "Metric (SI)":
    p_unit = "kPa"
    default_sigma = 50.0
elif unit_system == "Imperial (tsf)":
    p_unit = "tsf"
    default_sigma = 1.0 
else:
    p_unit = "psf"
    default_sigma = 2000.0

st.sidebar.markdown(f"**Current Units:** Pressure in `{p_unit}`")

upload_type = st.sidebar.radio("Upload format:", ["CSV", "PDF"])
uploaded_file = st.sidebar.file_uploader(f"Upload {upload_type} file", type=[upload_type.lower()])

st.sidebar.header("2. Soil Parameters")
sigma_v0 = st.sidebar.number_input(f"In-Situ Overburden Pressure ($\sigma'_0$) [{p_unit}]", value=default_sigma)
e0_input = st.sidebar.number_input("Initial Void Ratio ($e_0$)", value=1.2, format="%.3f")

# --- MAIN LOGIC ---

df = None

# Mock Data Generator
if not uploaded_file:
    st.info("👋 Upload a file to begin. Using **Synthetic Demo Data** for visualization.")
    e_syn = np.array([1.2, 1.19, 1.17, 1.13, 0.98, 0.80, 0.62, 0.44, 0.46, 0.50])
    if unit_system == "Metric (SI)": 
        p_syn = np.array([10, 20, 40, 80, 160, 320, 640, 1280, 320, 80])
    elif unit_system == "Imperial (tsf)":
        p_syn = np.array([0.1, 0.2, 0.4, 0.8, 1.6, 3.2, 6.4, 12.8, 3.2, 0.8])
    else:
        p_syn = np.array([200, 400, 800, 1600, 3200, 6400, 12800, 25600, 6400, 1600])
    df = pd.DataFrame({'Pressure': p_syn, 'Void Ratio': e_syn})
else:
    if upload_type == "CSV":
        try:
            df = pd.read_csv(uploaded_file)
            df = df.apply(pd.to_numeric, errors='coerce').dropna()
        except:
            st.error("CSV format invalid.")
    else:
        df = extract_from_pdf(uploaded_file)
        if df is None:
            st.error("Could not extract table from PDF. Please ensure the PDF has a clear table structure or try CSV.")

if df is not None and not df.empty:
    try:
        # Column mapping
        pressure_col = df.columns[0]
        void_col = df.columns[1]
        
        # Sort by pressure just in case
        # But we must respect the load/unload order. 
        # So we generally trust the input order unless it's completely scrambled.
        
        # Identify Max Pressure Index to split Load/Unload
        max_p_idx = df[pressure_col].idxmax()
        loading_df = df.iloc[:max_p_idx+1].copy()
        unloading_df = df.iloc[max_p_idx:].copy()
        
        # Filter zeroes to avoid log errors
        loading_df = loading_df[loading_df[pressure_col] > 0]
        
        if len(loading_df) < 3:
            st.error("Error: Not enough data points in the loading curve (need at least 3).")
        else:
            loading_df['log_p'] = np.log10(loading_df[pressure_col])
            
            # --- CALCULATIONS ---
            
            # 1. Cc (Compression Index)
            slope_vcl, intercept_vcl = find_linear_portion(loading_df['log_p'].values, loading_df[void_col].values)
            Cc = -slope_vcl 
            
            # 2. Cr (Recompression Index)
            if len(unloading_df) > 1:
                unloading_df = unloading_df[unloading_df[pressure_col] > 0]
                unloading_df['log_p'] = np.log10(unloading_df[pressure_col])
                slope_url, _ = np.polyfit(unloading_df['log_p'], unloading_df[void_col], 1)
                Cr = -slope_url
            else:
                Cr = 0.0
                
            # 3. Max Curvature
            st.sidebar.header("3. Calibration")
            smooth_factor = st.sidebar.slider("Spline Smoothing", 0.0, 1.0, 0.3)
            
            max_k_idx, spline_fit = calculate_curvature(loading_df['log_p'], loading_df[void_col], smooth_factor)
            
            # Slider for manual adjust
            override_idx = st.sidebar.slider("Adjust Point of Max Curvature", 0, len(loading_df)-1, int(max_k_idx))
            
            # Point T coords
            log_p_T = loading_df['log_p'].iloc[override_idx]
            e_T = loading_df[void_col].iloc[override_idx]
            
            # Casagrande Math
            slope_tan = 0
            # If spline exists, use it for tangent. Else use finite difference.
            if spline_fit:
                slope_tan = spline_fit.derivative(n=1)(log_p_T)
            else:
                # Fallback: simple slope between neighboring points
                if override_idx < len(loading_df)-1:
                    dy = loading_df[void_col].iloc[override_idx+1] - loading_df[void_col].iloc[override_idx]
                    dx = loading_df['log_p'].iloc[override_idx+1] - loading_df['log_p'].iloc[override_idx]
                    slope_tan = dy/dx if dx != 0 else 0
            
            b_tan = e_T - slope_tan * log_p_T
            
            # Bisector Math
            angle_tan = np.arctan(slope_tan)
            angle_bi = angle_tan / 2
            slope_bi = np.tan(angle_bi)
            b_bi = e_T - slope_bi * log_p_T
            
            # Intersection (Pc')
            log_Pc, e_Pc = get_intersection(slope_bi, b_bi, slope_vcl, intercept_vcl)
            
            if log_Pc is not None:
                Pc = 10**log_Pc
            else:
                Pc = 0.0 # Error state
            
            # Schmertmann Math
            e_target = 0.42 * e0_input
            if slope_vcl != 0:
                log_p_schm = (e_target - intercept_vcl) / slope_vcl
            else:
                log_p_schm = 0
            
            # --- OUTPUTS ---
            col1, col2, col3 = st.columns(3)
            col1.metric(f"Pre-consolidation Pressure (Pc')", f"{Pc:.1f} {p_unit}")
            col2.metric("Compression Index (Cc)", f"{Cc:.3f}")
            col3.metric("Recompression Index (Cr)", f"{Cr:.3f}")

            # --- PLOTTING ---
            tab1, tab2 = st.tabs(["Casagrande Construction", "Schmertmann Correction"])
            
            with tab1:
                fig, ax = plt.subplots(figsize=(10, 6))
                
                # Plot Lines
                x_range = np.linspace(loading_df['log_p'].min(), loading_df['log_p'].max(), 100)
                x_range_linear = 10**x_range
                
                ax.plot(loading_df[pressure_col], loading_df[void_col], 'ko-', label='Lab Data', markersize=5)
                ax.plot(x_range_linear, [e_T]*100, 'b--', alpha=0.4, label='Horizontal')
                ax.plot(x_range_linear, slope_tan*x_range + b_tan, 'g--', alpha=0.4, label='Tangent')
                ax.plot(x_range_linear, slope_bi*x_range + b_bi, 'm--', alpha=0.8, linewidth=1.5, label='Bisector')
                ax.plot(x_range_linear, slope_vcl*x_range + intercept_vcl, 'r--', alpha=0.6, label='VCL')
                
                # Plot Pc Point
                if Pc > 0:
                    ax.plot(Pc, e_Pc, 'rx', markersize=12, markeredgewidth=3, label="Pc'")
                    ax.annotate(f"Pc' = {Pc:.1f} {p_unit}", (Pc, e_Pc), xytext=(Pc*1.3, e_Pc+0.05), 
                                arrowprops=dict(facecolor='black', shrink=0.05))
                
                ax.set_xscale('log')
                ax.set_xlabel(f"Effective Stress ({p_unit}) [Log Scale]")
                ax.set_ylabel("Void Ratio (e)")
                ax.set_ylim(min(loading_df[void_col])*0.9, max(loading_df[void_col])*1.1)
                ax.legend()
                ax.grid(True, which="both", ls="-", alpha=0.3)
                st.pyplot(fig)
                
            with tab2:
                fig2, ax2 = plt.subplots(figsize=(10, 6))
                ax2.plot(loading_df[pressure_col], loading_df[void_col], 'k--', alpha=0.4, label='Lab Data')
                
                # Schmertmann Field Curve
                if log_p_schm > 0:
                    p_schmertmann = [sigma_v0, 10**log_p_schm]
                    e_schmertmann = [e0_input, e_target]
                    ax2.plot(p_schmertmann, e_schmertmann, 'b-', linewidth=2, label='Field Corrected')
                
                ax2.plot(sigma_v0, e0_input, 'bo', label='In-Situ State')
                
                ax2.set_xscale('log')
                ax2.set_xlabel(f"Effective Stress ({p_unit}) [Log Scale]")
                ax2.set_ylabel("Void Ratio (e)")
                ax2.legend()
                ax2.grid(True, which="both", ls="-", alpha=0.3)
                st.pyplot(fig2)

    except Exception as e:
        st.error(f"Error in calculation logic: {e}. Try checking your data format.")
