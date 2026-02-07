import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from scipy.interpolate import UnivariateSpline
from PIL import Image
from streamlit_image_coordinates import streamlit_image_coordinates

# --- 1. GEOTECHNICAL MATH ENGINE ---
# (This logic remains the same, but now processes clicked points)

def calculate_casagrande(df):
    try:
        # Sort and clean
        df = df.sort_values('Pressure')
        log_p = np.log10(df['Pressure'])
        e = df['VoidRatio']
        
        # Spline for curvature
        if len(df) < 4: return None, None, None, None # Not enough data
        
        spline = UnivariateSpline(log_p, e, k=3, s=0.1)
        
        # Find max curvature
        p_range = np.linspace(log_p.min(), log_p.max(), 100)
        y_d1 = spline.derivative(n=1)(p_range)
        y_d2 = spline.derivative(n=2)(p_range)
        k = np.abs(y_d2) / (1 + y_d1**2)**1.5
        
        # Max curvature index (ignore edges)
        idx_max = np.argmax(k[5:-5]) + 5
        log_p_T = p_range[idx_max]
        e_T = spline(log_p_T)
        
        # Tangent & Bisector
        slope_tan = spline.derivative(n=1)(log_p_T)
        angle_tan = np.arctan(slope_tan)
        angle_bi = angle_tan / 2
        slope_bi = np.tan(angle_bi)
        
        # VCL (last few points)
        slope_vcl, intercept_vcl = np.polyfit(log_p[-3:], e[-3:], 1)
        
        # Intersection
        # Line 1 (Bisector): y - e_T = m_bi * (x - log_p_T) -> y = m_bi*x + (e_T - m_bi*log_p_T)
        b_bi = e_T - slope_bi * log_p_T
        
        # Intersection X = (b2 - b1) / (m1 - m2)
        log_pc = (intercept_vcl - b_bi) / (slope_bi - slope_vcl)
        pc = 10**log_pc
        
        return pc, -slope_vcl, (log_p_T, e_T), (slope_vcl, intercept_vcl)
    except:
        return None, None, None, None

# --- 2. COORDINATE MAPPING ---

def map_pixels_to_data(clicks, p1_ref, p2_ref, is_log_x=True):
    """
    Maps pixel coordinates (x, y) to data values using 2 reference points.
    p1_ref = {'pixel': (x,y), 'value': (val_x, val_y)}
    """
    # 1. Calculate Scale Factors
    pix_x1, pix_y1 = p1_ref['pixel']
    pix_x2, pix_y2 = p2_ref['pixel']
    
    val_x1, val_y1 = p1_ref['value']
    val_x2, val_y2 = p2_ref['value']
    
    # Handle Log Scale for X
    if is_log_x:
        val_x1 = np.log10(val_x1)
        val_x2 = np.log10(val_x2)
        
    # Pixels per unit
    x_scale = (val_x2 - val_x1) / (pix_x2 - pix_x1)
    y_scale = (val_y2 - val_y1) / (pix_y2 - pix_y1)
    
    data_points = []
    for (px, py) in clicks:
        # Interpolate
        vx = val_x1 + (px - pix_x1) * x_scale
        vy = val_y1 + (py - pix_y1) * y_scale
        
        if is_log_x:
            vx = 10**vx
            
        data_points.append({'Pressure': vx, 'VoidRatio': vy})
        
    return pd.DataFrame(data_points)

# --- 3. UI MAIN ---

st.set_page_config(page_title="GeoConsolidation Digitizer", layout="wide")

if 'step' not in st.session_state: st.session_state.step = 1
if 'cal_points' not in st.session_state: st.session_state.cal_points = []
if 'data_points' not in st.session_state: st.session_state.data_points = []

st.title("Graphs to Data: Consolidation Digitizer")
st.markdown("Use this tool if you only have an **Image** of the consolidation curve.")

# SIDEBAR CONTROLS
st.sidebar.header("Step 1: Upload Image")
uploaded_file = st.sidebar.file_uploader("Upload Graph (PNG/JPG/Screenshot)", type=["png", "jpg", "jpeg"])

if uploaded_file:
    image = Image.open(uploaded_file)
    w, h = image.size
    
    # Layout
    col_img, col_data = st.columns([2, 1])
    
    with col_img:
        st.subheader("Interactive Graph")
        
        # INSTRUCTIONS BASED ON STEP
        if st.session_state.step == 1:
            st.info("👆 **Click on a GRID INTERSECTION on the LEFT side** (e.g., P=10, e=1.0).")
        elif st.session_state.step == 2:
            st.info("👆 **Click on a GRID INTERSECTION on the RIGHT side** (e.g., P=1000, e=0.5).")
        elif st.session_state.step == 3:
            st.success("👆 **Click along the CURVE** to digitize points. (Click 'Done' in sidebar when finished).")
            
        # THE MAGIC COMPONENT
        # This captures clicks on the image
        value = streamlit_image_coordinates(image, key="pil")
        
        # LOGIC HANDLER
        if value:
            coords = (value['x'], value['y'])
            
            # Prevent duplicate clicks registering instantly
            last_click = st.session_state.get('last_click', None)
            if coords != last_click:
                st.session_state.last_click = coords
                
                if st.session_state.step == 1:
                    st.session_state.cal_points.append({'pixel': coords})
                    st.session_state.step = 2
                    st.rerun()
                    
                elif st.session_state.step == 2:
                    st.session_state.cal_points.append({'pixel': coords})
                    st.session_state.step = 3
                    st.rerun()
                    
                elif st.session_state.step == 3:
                    st.session_state.data_points.append(coords)
                    st.toast(f"Point Added: {coords}")

    with col_data:
        st.subheader("Calibration & Results")
        
        # Calibration Inputs
        if len(st.session_state.cal_points) >= 1:
            st.write("**Reference Point 1 (Left)**")
            st.write(f"Pixel: {st.session_state.cal_points[0]['pixel']}")
            p1_val = st.number_input("Pressure @ Ref 1", value=10.0, key="p1")
            e1_val = st.number_input("Void Ratio @ Ref 1", value=1.5, key="e1")
            
        if len(st.session_state.cal_points) >= 2:
            st.write("---")
            st.write("**Reference Point 2 (Right)**")
            st.write(f"Pixel: {st.session_state.cal_points[1]['pixel']}")
            p2_val = st.number_input("Pressure @ Ref 2", value=1000.0, key="p2")
            e2_val = st.number_input("Void Ratio @ Ref 2", value=0.5, key="e2")
            
            st.write("---")
            if st.button("Reset / Start Over"):
                st.session_state.step = 1
                st.session_state.cal_points = []
                st.session_state.data_points = []
                st.rerun()

        # RESULTS AREA
        if len(st.session_state.data_points) > 3:
            st.write(f"**Digitized Points:** {len(st.session_state.data_points)}")
            
            # MAP PIXELS TO DATA
            p1 = {'pixel': st.session_state.cal_points[0]['pixel'], 'value': (p1_val, e1_val)}
            p2 = {'pixel': st.session_state.cal_points[1]['pixel'], 'value': (p2_val, e2_val)}
            
            df_digitized = map_pixels_to_data(st.session_state.data_points, p1, p2)
            
            # Show Table
            st.dataframe(df_digitized.head(), height=150)
            
            # RUN MATH
            Pc, Cc, T_pt, VCL_params = calculate_casagrande(df_digitized)
            
            if Pc:
                st.metric("Calculated Pc'", f"{Pc:.1f} (Units match input)")
                st.metric("Compression Index (Cc)", f"{Cc:.3f}")
                
                # Schmertmann inputs
                st.write("---")
                st.write("**Schmertmann Params**")
                sigma_0 = st.number_input("In-Situ Pressure", value=p1_val*5)
                e_0 = st.number_input("In-Situ Void Ratio", value=e1_val)
                
                # Quick plot
                fig, ax = plt.subplots(figsize=(4,3))
                ax.plot(df_digitized['Pressure'], df_digitized['VoidRatio'], 'ko-')
                ax.set_xscale('log')
                ax.set_title("Digitized Curve")
                st.pyplot(fig)
            else:
                st.warning("Need more points to calculate Pc.")

else:
    st.info("Please upload an image (Screenshot your PDF if needed).")
