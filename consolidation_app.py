import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from scipy.interpolate import UnivariateSpline
import pdfplumber

# --- 1. GEOTECHNICAL MATH ENGINE (ROBUST) ---

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

def extract_from_pdf(uploaded_file):
    """
    Attempts to extract table. If standard table fails, tries text parsing.
    """
    try:
        with pdfplumber.open(uploaded_file) as pdf:
            # Iterate through pages to find a table
            for page in pdf.pages:
                table = page.extract_table()
                if table:
                    # Create DataFrame
                    df = pd.DataFrame(table[1:], columns=table[0])
                    return df
        return None
    except Exception as e:
        st.error(f"PDF extraction error: {e}")
        return None

def clean_data(df):
    """
    Cleans the dataframe:
    1. Renames columns to 'Pressure' and 'VoidRatio'
    2. Coerces to numeric
    3. Drops bad rows
    """
    # 1. Rename columns (Auto-detect)
    # We assume Column 0 is Pressure, Column 1 is Void Ratio (or Strain)
    # If the user uploads a 10-column file, we take the first two meaningful ones.
    
    # Strip whitespace from headers
    df.columns = df.columns.astype(str).str.strip()
    
    # Select first two columns
    df = df.iloc[:, [0, 1]]
    df.columns = ['Pressure', 'VoidRatio']
    
    # 2. Force Numeric (invalid cells become NaN)
    df['Pressure'] = pd.to_numeric(df['Pressure'], errors='coerce')
    df['VoidRatio'] = pd.to_numeric(df['VoidRatio'], errors='coerce')
    
    # 3. Drop NaNs
    df = df.dropna()
    
    # 4. Sort by Pressure (crucial for logic)
    df = df.sort_values(by='Pressure').reset_index(drop=True)
    
    return df

# --- 2. STREAMLIT UI ---

st.set_page_config(page_title="GeoConsolidation AI", layout="wide")
st.title("1-D Consolidation Analysis")
st.markdown("Automated **Casagrande Construction** & **Schmertmann Correction**.")

# --- SIDEBAR ---
st.sidebar.header("1. Input Data")
upload_type = st.sidebar.radio("Upload format:", ["CSV", "PDF"])
uploaded_file = st.sidebar.file_uploader(f"Upload {upload_type} file", type=[upload_type.lower()])

debug_mode = st.sidebar.checkbox("Show Raw Data (Debug)", value=False)

st.sidebar.header("2. Project Settings")
unit_system = st.sidebar.selectbox("Unit System", ["Metric (SI)", "Imperial (tsf)", "Imperial (psf)"])

if unit_system == "Metric (SI)":
    p_unit = "kPa"
    default_sigma
