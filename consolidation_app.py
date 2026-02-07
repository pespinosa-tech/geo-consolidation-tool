import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from scipy.interpolate import UnivariateSpline
import pdfplumber

# --- 1. GEOTECHNICAL MATH ENGINE ---

def calculate_curvature(log_pressure, void_ratio, smoothing_factor=0.5):
    """Calculates max curvature index with safety checks."""
    if len(log_pressure) < 5:
        return 2, None 

    try:
        spline = UnivariateSpline(log_pressure, void_ratio, k=3, s=smoothing_factor)
        y_d1 = spline.derivative(n=1)(log_pressure)
        y_d2 = spline.derivative(n=2)(log_pressure)
        curvature = np.abs(y_d2) / (1 + y_d1**2)**1.5
        
        search_limit = int(len(curvature) * 0.7)
        if search_limit < 2: search_limit = len(curvature) - 1
            
        search_window = curvature[1:search_limit]
        if len(search_window) == 0: return 2, spline
            
        max_k_idx = np.argmax(search_window) + 1
        return max_k_idx, spline
    except Exception:
        return int(len(log_pressure)/2), None

def find_linear_portion(log_pressure, void_ratio):
    """Finds slope of last 3-4 points (Virgin Compression Line)."""
    points = min(len(log_pressure), 4)
    if points < 2: return 0, 0
    
    x_vcl = log_pressure[-points:]
    y_vcl = void_ratio[-points:]
    slope, intercept = np.polyfit(x_vcl, y_vcl, 1)
    return slope, intercept

def get_intersection(m1, b1, m2, b2):
    if m1 == m2: return None, None
    x = (b2 - b1) / (m1 - m2)
    y = m1 * x + b1
    return x, y

def clean_data(df):
    """
    Cleans the dataframe:
    1. Renames columns to 'Pressure' and 'VoidRatio'
    2. Coerces to numeric
    3. Drops bad rows
    """
    # Normalize headers
    df.columns = df.columns.astype(str).str.strip()
    
    # Select first two columns (Pressure, Void Ratio)
    if len(df.columns) >= 2:
        df = df.iloc[:, [0, 1]]
        df.columns = ['Pressure', 'VoidRatio']
    else:
        return pd.DataFrame() # Return empty if not enough cols
    
    # Force Numeric
    df['Pressure'] = pd.to_numeric(df['Pressure'], errors='coerce')
    df['VoidRatio'] = pd.to_numeric(df['VoidRatio'], errors='coerce')
    
    # Drop NaNs
    df = df.dropna()
    
    # Sort by Pressure
    df = df.sort_values(by='Pressure').reset_index(drop=True)
    
    return df

def extract_from_pdf(uploaded_file):
    """Attempts to extract table from PDF."""
    try:
        with pdfplumber.open(uploaded_file) as pdf:
            for page in pdf.pages:
                table = page.extract_table()
                if table:
                    df = pd.DataFrame(table[1:], columns=table[0])
                    return df
        return None
    except Exception as e:
        return None

# --- 2. STREAMLIT UI ---

st.set_page_config(page_title="GeoConsolidation AI", layout="wide")
st.title("1-D Consolidation Analysis")
st.markdown("Automated **Casagrande Construction** & **Schmertmann Correction**.")

# --- SIDEBAR ---

# Initialize default values to prevent NameError
default_sigma = 50.0
p_unit = "kPa"

st.sidebar.header("1. Input Data")
upload_type = st.sidebar.radio("Upload format:", ["CSV", "PDF"])
uploaded_file = st.sidebar.file_uploader(f"Upload {upload_type} file", type=[upload_type.lower()])

debug_mode = st.sidebar.checkbox("Show Raw Data (Debug)", value=False)

st.sidebar.header("2. Project Settings")
unit_system = st.sidebar.selectbox("Unit System", ["Metric (SI)", "Imperial (tsf)", "Imperial (psf)"])

# Update defaults based on selection
if unit_system == "Metric (SI)":
    p_unit = "kPa"
    default_sigma = 50.0
elif unit_system == "Imperial (tsf)":
    p_unit = "tsf"
    default_sigma = 1.0 
else:
    p_unit = "psf"
    default_sigma = 2000.0

sigma_v0 = st.sidebar.number_input(f"In-Situ Pressure ($\sigma'_0$) [{p_unit}]", value=default_sigma)
e0_input = st.sidebar.number_input("Initial Void Ratio ($e_0$)", value=1.0, format="%.3f")

# --- MAIN LOGIC ---

df = None

# LOAD DATA
if uploaded_file:
    if upload_type == "CSV":
        try:
            raw_df = pd.read_csv(uploaded_file)
            df = clean_data(raw_df)
        except Exception as e:
            st.error(f"CSV Error: {e}")
    else:
        # PDF Handling
        raw_df = extract_from_pdf(uploaded_file)
        if raw_df is not None:
            if debug_mode:
                st.subheader("Raw Data from PDF (Before Cleaning)")
                st.write(raw_df)
            
            df = clean_data(raw_df)
            
            if df.empty:
                st.error("Found a table structure, but it contained no numbers. Please check 'Show Raw Data'.")
        else:
            st.error("No tables found in PDF. Please convert data to CSV.")
else:
    # SYNTHETIC DEMO DATA
    if unit_system == "Metric (SI)": 
        p_syn = [10, 20, 40, 80, 160, 320, 640, 1280]
    elif unit_system == "Imperial (tsf)":
        p_syn = [0.1, 0.2, 0.4, 0.8, 1.6, 3.2, 6.4, 12.8]
    else:
        p_syn = [200, 400, 800, 1600, 3200, 6400, 12800, 25600]
    
    e_syn = [1.2, 1.19, 1.17, 1.13, 0.98, 0.80, 0.62, 0.44]
    df = pd.DataFrame({'Pressure': p_syn, 'VoidRatio': e_syn})
    
    if not uploaded_file:
        st.info("👋 Upload a file to start. Using Demo Data.")

# PROCESS DATA
if df is not None and not df.empty:
    if debug_mode:
        st.subheader("Cleaned Data (Used for Calculation)")
        st.write(df)

    # Prepare Data
    df = df[df['Pressure'] > 0] # Remove zero pressures
    df['log_p'] = np
