import streamlit as st
import pandas as pd
import numpy as np
import joblib

# ----------------------------------
# Page configuration
# ----------------------------------
st.set_page_config(
    page_title="Kerala House Price Predictor",
    page_icon="🏠",
    layout="wide"
)

# ----------------------------------
# Load model
# ----------------------------------
model = joblib.load("Model_lr.joblib")

# ----------------------------------
# Header Section
# ----------------------------------
st.markdown(
    """
    <h1 style="text-align:center;">🏠 Kerala House Price Predictor</h1>
    <p style="text-align:center; color:grey;">
    AI-powered house price estimation using Machine Learning
    </p>
    <hr>
    """,
    unsafe_allow_html=True
)

# ----------------------------------
# Banner Image (Optional but Powerful)
# ----------------------------------
st.image(
    "https://images.unsplash.com/photo-1600585154340-be6161a56a0c",
     use_container_width=True
)

# ----------------------------------
# Sidebar Inputs
# ----------------------------------
st.sidebar.header("📋 Enter Property Details")

District = st.sidebar.selectbox(
    "📍 District",
    ['Thrissur', 'Kannur', 'Kozhikode', 'Kochi', 'Thiruvananthapuram']
)

Property_Type = st.sidebar.selectbox(
    "🏘️ Property Type",
    ['Villa', 'Apartment', 'Independent House']
)

BHK = st.sidebar.selectbox("🛏️ BHK", [1, 2, 3, 4, 5])
Bathrooms = st.sidebar.selectbox("🛁 Bathrooms", [1, 2, 3, 4])
Parking_Slots = st.sidebar.selectbox("🚗 Parking Slots", [0, 1, 2])

Builtup_Area_sqft = st.sidebar.slider(
    "📐 Built-up Area (sqft)", 300, 10000, 1200
)

Land_Area_cent = st.sidebar.slider(
    "🌱 Land Area (cent)", 0.5, 50.0, 5.0
)

Age_of_Property = st.sidebar.slider(
    "🏚️ Age of Property (years)", 0, 50, 5
)

Distance_to_City_km = st.sidebar.slider(
    "📏 Distance to City (km)", 0.5, 50.0, 5.0
)

predict_btn = st.sidebar.button("🔮 Predict Price")

# ----------------------------------
# Main Layout
# ----------------------------------
col1, col2 = st.columns([2, 1])

with col1:
    st.subheader("📊 Property Summary")
    st.markdown(
        f"""
        - **District:** {District}  
        - **Property Type:** {Property_Type}  
        - **Configuration:** {BHK} BHK, {Bathrooms} Bathrooms  
        - **Built-up Area:** {Builtup_Area_sqft} sqft  
        - **Land Area:** {Land_Area_cent} cent  
        - **Age of Property:** {Age_of_Property} years  
        - **Parking Slots:** {Parking_Slots}  
        - **Distance to City:** {Distance_to_City_km} km  
        """
    )

with col2:
    st.subheader("💰 Estimated Price")

    if predict_btn:
        input_df = pd.DataFrame({
            'District': [District],
            'Property_Type': [Property_Type],
            'BHK': [BHK],
            'Builtup_Area_sqft': [Builtup_Area_sqft],
            'Land_Area_cent': [Land_Area_cent],
            'Age_of_Property': [Age_of_Property],
            'Bathrooms': [Bathrooms],
            'Parking_Slots': [Parking_Slots],
            'Distance_to_City_km': [Distance_to_City_km]
        })

        prediction = model.predict(input_df)[0]

        st.markdown(
            f"""
            <div style="
                background-color:#f0f9ff;
                padding:25px;
                border-radius:15px;
                text-align:center;
                border:2px solid #1f77b4;
            ">
                <h2 style="color:#1f77b4;">₹ {prediction:,.2f} Lakhs</h2>
                <p style="color:grey;">Predicted Market Value</p>
            </div>
            """,
            unsafe_allow_html=True
        )
    else:
        st.info("👈 Enter property details and click **Predict Price**")

# ----------------------------------
# Footer
# ----------------------------------
st.markdown(
    """
    <hr>
    <p style="text-align:center; color:grey;">
    Built ❤️ by Ravi
    </p>
    """,
    unsafe_allow_html=True
)