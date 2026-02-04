import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from scipy.interpolate import UnivariateSpline
import pdfplumber
import io

# --- 1. GEOTECHNICAL MATH ENGINE ---

def calculate_curvature(log_pressure, void_ratio, smoothing_factor=0.5):
    """
    Calculates the curvature (k) of the e-log(p) curve using a spline.
    Returns the index of the point of maximum curvature.
    """
    # Fit a spline to smooth the data (order 3 for derivatives)
    # Weights can be adjusted if data is very noisy
    spline = UnivariateSpline(log_pressure, void_ratio, k=3, s=smoothing_factor)
    
    # Derivatives
    y_d1 = spline.derivative(n=1)(log_pressure)
    y_d2 = spline.derivative(n=2)(log_pressure)
    
    # Curvature formula: k = |y''| / (1 + (y')^2)^(3/2)
    curvature = np.abs(y_d2) / (1 + y_d1**2)**1.5
    
    # Find index of max curvature (limit search to first half of curve usually)
    # We ignore the very first few points to avoid noise at start
    search_limit = int(len(curvature) * 0.7) 
    max_k_idx = np.argmax(curvature[1:search_limit]) + 1
    
    return max_k_idx, spline

def find_linear_portion(log_pressure, void_ratio):
    """
    Identifies the Virgin Compression Line (VCL) - the steep straight linear portion.
    Usually the last few points of the loading curve.
    """
    # We take the last 3-4 points as the VCL anchor
    x_vcl = log_pressure[-4:]
    y_vcl = void_ratio[-4:]
    
    # Linear regression (Polyfit degree 1)
    slope, intercept = np.polyfit(x_vcl, y_vcl, 1)
    
    return slope, intercept

def get_intersection(m1, b1, m2, b2):
    """
    Finds intersection (x, y) of two lines: y = m1*x + b1 and y = m2*x + b2
    """
    if m1 == m2:
        return None
    x = (b2 - b1) / (m1 - m2)
    y = m1 * x + b1
    return x, y

# --- 2. DATA EXTRACTION ---

def extract_from_pdf(uploaded_file):
    """
    Attempt to extract a table from PDF. 
    Assumes a standard table structure.
    """
    try:
        with pdfplumber.open(uploaded_file) as pdf:
            first_page = pdf.pages[0]
            table = first_page.extract_table()
            
            if table:
                df = pd.DataFrame(table[1:], columns=table[0])
                # Attempt to clean numerical columns
                df = df.apply(pd.to_numeric, errors='coerce')
                df = df.dropna()
                return df
            else:
                return None
    except Exception as e:
        st.error(f"PDF Parsing Error: {e}")
        return None

# --- 3. STREAMLIT UI ---

st.set_page_config(page_title="GeoConsolidation AI", layout="wide")
st.title("1-D Consolidation Analysis: Casagrande & Schmertmann")
st.markdown("""
This app automates the **Casagrande Construction** to find Pre-consolidation Pressure ($P_c'$) 
and performs the **Schmertmann Correction** for sample disturbance.
""")

# --- SIDEBAR INPUTS ---
st.sidebar.header("1. Data Input")
upload_type = st.sidebar.radio("Upload format:", ["CSV", "PDF"])
uploaded_file = st.sidebar.file_uploader(f"Upload {upload_type} file", type=[upload_type.lower()])

st.sidebar.header("2. Soil Parameters")
sigma_v0 = st.sidebar.number_input("In-Situ Overburden Pressure ($\sigma'_0$) [kPa]", value=50.0)
e0_input = st.sidebar.number_input("Initial Void Ratio ($e_0$)", value=1.2, format="%.3f")

# --- MAIN LOGIC ---

df = None

# Mock Data for Demo if no file
if not uploaded_file:
    st.info("👋 Upload a file to begin. Using **Synthetic Demo Data** for visualization.")
    # Create synthetic consolidation data
    p_syn = np.array([10, 20, 40, 80, 160, 320, 640, 1280, 320, 80])
    e_syn = np.array([1.2, 1.19, 1.17, 1.13, 0.98, 0.80, 0.62, 0.44, 0.46, 0.50])
    df = pd.DataFrame({'Pressure (kPa)': p_syn, 'Void Ratio': e_syn})
else:
    if upload_type == "CSV":
        df = pd.read_csv(uploaded_file)
    else:
        df = extract_from_pdf(uploaded_file)
        if df is None:
            st.error("Could not extract table from PDF. Please try CSV.")

if df is not None:
    # Normalize column names for processing
    # We assume column 0 is Pressure, column 1 is Void Ratio (or Strain)
    try:
        pressure_col = df.columns[0]
        void_col = df.columns[1]
        
        # Ensure data is sorted by pressure for the loading curve
        # We need to separate Loading from Unloading
        max_p_idx = df[pressure_col].idxmax()
        loading_df = df.iloc[:max_p_idx+1].copy()
        unloading_df = df.iloc[max_p_idx:].copy()
        
        # Log Pressure
        loading_df['log_p'] = np.log10(loading_df[pressure_col])
        
        # --- CALCULATIONS ---
        
        # 1. Cc (Compression Index) - Slope of VCL
        slope_vcl, intercept_vcl = find_linear_portion(loading_df['log_p'].values, loading_df[void_col].values)
        Cc = -slope_vcl # Cc is positive convention
        
        # 2. Cr (Recompression Index) - Slope of Unload
        if len(unloading_df) > 1:
            unloading_df['log_p'] = np.log10(unloading_df[pressure_col])
            slope_url, _ = np.polyfit(unloading_df['log_p'], unloading_df[void_col], 1)
            Cr = -slope_url
        else:
            Cr = 0.0
            
        # 3. Max Curvature (Casagrande Point T)
        st.sidebar.header("3. Calibration")
        smooth_factor = st.sidebar.slider("Spline Smoothing", 0.0, 1.0, 0.3)
        max_k_idx, spline_fit = calculate_curvature(loading_df['log_p'], loading_df[void_col], smooth_factor)
        
        # Allow user to override the point of max curvature manually
        override_idx = st.sidebar.slider("Adjust Point of Max Curvature", 1, len(loading_df)-2, int(max_k_idx))
        
        # Get coordinates of Point T
        log_p_T = loading_df['log_p'].iloc[override_idx]
        e_T = loading_df[void_col].iloc[override_idx]
        
        # --- CASAGRANDE MATH ---
        
        # Tangent at T
        slope_tan = spline_fit.derivative(n=1)(log_p_T)
        b_tan = e_T - slope_tan * log_p_T
        
        # Horizontal line at T
        slope_horiz = 0
        b_horiz = e_T
        
        # Bisector
        # Angle of tangent: theta = arctan(slope)
        # Angle of horizontal: 0
        # Bisector angle: theta / 2
        # Note: Visual bisector depends on axis aspect ratio, but we calculate mathematical bisector here
        # For rigorous Casagrande, we bisect the angle.
        # Since log scale is dimensionless in x, we treat it as Cartesian for the construction.
        angle_tan = np.arctan(slope_tan)
        angle_bi = angle_tan / 2
        slope_bi = np.tan(angle_bi)
        b_bi = e_T - slope_bi * log_p_T
        
        # Intersection of Bisector and VCL
        log_Pc, e_Pc = get_intersection(slope_bi, b_bi, slope_vcl, intercept_vcl)
        Pc = 10**log_Pc
        
        # --- SCHMERTMANN MATH ---
        # 1. Point 1: (sigma_v0, e0)
        # 2. Point 2: Intersection of VCL with e = 0.42 * e0
        e_target = 0.42 * e0_input
        # VCL eq: e = -Cc * logp + b_vcl
        # logp = (e - b_vcl) / -Cc
        log_p_schm = (e_target - intercept_vcl) / slope_vcl
        
        # --- OUTPUTS ---
        
        col1, col2, col3 = st.columns(3)
        col1.metric("Pre-consolidation Pressure (Pc')", f"{Pc:.1f} kPa")
        col2.metric("Compression Index (Cc)", f"{Cc:.3f}")
        col3.metric("Recompression Index (Cr)", f"{Cr:.3f}")

        # --- PLOTTING ---
        
        tab1, tab2 = st.tabs(["Casagrande Construction", "Schmertmann Correction"])
        
        with tab1:
            fig, ax = plt.subplots(figsize=(10, 6))
            
            # Plot Raw Data
            ax.plot(loading_df[pressure_col], loading_df[void_col], 'ko-', label='Lab Data', markersize=4)
            ax.set_xscale('log')
            ax.set_xlabel("Effective Stress (kPa) [Log Scale]")
            ax.set_ylabel("Void Ratio (e)")
            ax.grid(True, which="both", ls="-", alpha=0.3)
            
            # Plot Casagrande Lines
            x_range = np.linspace(loading_df['log_p'].min(), loading_df['log_p'].max(), 100)
            x_range_linear = 10**x_range
            
            # 1. Horizontal
            ax.plot(x_range_linear, [e_T]*100, 'b--', alpha=0.5, label='Horizontal')
            
            # 2. Tangent
            ax.plot(x_range_linear, slope_tan*x_range + b_tan, 'g--', alpha=0.5, label='Tangent')
            
            # 3. Bisector
            ax.plot(x_range_linear, slope_bi*x_range + b_bi, 'm--', alpha=0.8, linewidth=2, label='Bisector')
            
            # 4. VCL Extension
            ax.plot(x_range_linear, slope_vcl*x_range + intercept_vcl, 'r--', alpha=0.8, label='Virgin Compression')
            
            # 5. Pc Point
            ax.plot(Pc, e_Pc, 'rx', markersize=12, markeredgewidth=3, label="Pc'")
            
            # Annotate
            ax.annotate(f"Pc' = {Pc:.1f}", (Pc, e_Pc), xytext=(Pc*1.2, e_Pc+0.05), arrowprops=dict(facecolor='black', shrink=0.05))
            
            ax.set_ylim(min(loading_df[void_col])*0.9, max(loading_df[void_col])*1.1)
            ax.legend()
            st.pyplot(fig)
            
        with tab2:
            fig2, ax2 = plt.subplots(figsize=(10, 6))
            
            # Plot Lab Data
            ax2.plot(loading_df[pressure_col], loading_df[void_col], 'k--', alpha=0.4, label='Lab Data')
            
            # Schmertmann Field Curve
            # Line from (sigma_v0, e0) to (10^log_p_schm, e_target)
            p_schmertmann = [sigma_v0, 10**log_p_schm]
            e_schmertmann = [e0_input, e_target]
            
            ax2.plot(p_schmertmann, e_schmertmann, 'b-', linewidth=2, label='Schmertmann Field Corrected')
            ax2.plot(sigma_v0, e0_input, 'bo', label='In-Situ State')
            
            ax2.set_xscale('log')
            ax2.set_xlabel("Effective Stress (kPa) [Log Scale]")
            ax2.set_ylabel("Void Ratio (e)")
            ax2.grid(True, which="both", ls="-", alpha=0.3)
            ax2.legend()
            
            st.pyplot(fig2)
            
            st.markdown(f"""
            **Schmertmann Logic:**
            1. Starts at In-Situ state: ($\sigma'_0={sigma_v0}, e_0={e0_input}$)
            2. Intersects Lab Virgin Line at $0.42 e_0$ ({e_target:.3f})
            """)

    except Exception as e:
        st.error(f"Error processing data: {e}. Please ensure your file has columns for Pressure and Void Ratio.")