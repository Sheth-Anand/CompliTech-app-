import streamlit as st
import pandas as pd
from sklearn.pipeline import Pipeline
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.preprocessing import LabelEncoder
from sklearn.ensemble import RandomForestClassifier
import io

# ------------------- Streamlit Page Setup -------------------
st.set_page_config(page_title="CompliTech - Certificate Predictor", layout="centered")

# ------------------- Train the Models -------------------
@st.cache_resource
def train_model():
    df = pd.read_csv("test.csv")
    df.columns = df.columns.str.strip()
    df = df.dropna(subset=['Certificates_Global', 'Certificates_India'])

    df['combined_text'] = (
        df['Product_Type'].astype(str) + ' ' +
        df['Material_Composition'].astype(str) + ' ' +
        df['Dimensions'].astype(str) + ' ' +
        df['Voltage_rating'].astype(str) + ' ' +
        df['Operating_Temp'].astype(str) + ' ' +
        df['IP_Rating'].astype(str) + ' ' +
        df['Hazardous_Components'].astype(str) + ' ' +
        df['Custom_Features'].astype(str)
    )

    le_global = LabelEncoder()
    le_india = LabelEncoder()
    df['target_global'] = le_global.fit_transform(df['Certificates_Global'])
    df['target_india'] = le_india.fit_transform(df['Certificates_India'])

    pipeline_global = Pipeline([
        ('tfidf', TfidfVectorizer(max_features=1000, ngram_range=(1, 2))),
        ('clf', RandomForestClassifier(n_estimators=100, random_state=42))
    ])

    pipeline_india = Pipeline([
        ('tfidf', TfidfVectorizer(max_features=1000, ngram_range=(1, 2))),
        ('clf', RandomForestClassifier(n_estimators=100, random_state=42))
    ])

    pipeline_global.fit(df['combined_text'], df['target_global'])
    pipeline_india.fit(df['combined_text'], df['target_india'])

    return pipeline_global, pipeline_india, le_global, le_india

# ------------------- Load Models -------------------
pipeline_global, pipeline_india, le_global, le_india = train_model()

# ------------------- Navigation Page State -------------------
if "page" not in st.session_state:
    st.session_state.page = "cover"

# ------------------- Cover Page -------------------
if st.session_state.page == "cover":
    st.image("cover.png",use_container_width=True)
    st.markdown("### 👋 Welcome to **CompliTech**")
    st.write("Predict India & Global compliance certificates by entering your product specs.")

    if st.button("🚀 Predict Now"):
        st.session_state.page = "predict"
        st.rerun()

# ------------------- Prediction Page -------------------
elif st.session_state.page == "predict":
    st.markdown("## 🧾 Enter Product Details Below")

    df = pd.read_csv("test.csv")
    df.columns = df.columns.str.strip()

    col1, col2 = st.columns(2)

    with col1:
        product_types = sorted(df['Product_Type'].dropna().unique())
        product_type = st.selectbox("🔧 Product Type", product_types)

        if product_type not in product_types:
            st.error("❌ Invalid Product Type selected.")
            st.stop()

        # Filter material options for selected product type
        material_options = df[df['Product_Type'] == product_type]['Material_Composition'].dropna().unique()
        if len(material_options) == 0:
            st.error("❌ No Material Composition found for selected Product Type.")
            st.stop()
        elif len(material_options) == 1:
            material = material_options[0]
            st.markdown(f"⚙️ **Material Composition**: `{material}` (Auto-selected)")
        else:
            material = st.selectbox("⚙️ Select Material Composition", sorted(material_options))

        if material not in material_options:
            st.error("❌ Invalid Material Composition selected.")
            st.stop()

        voltage = st.text_input("🔌 Voltage Rating", placeholder="e.g., 230V AC")

        # Filter IP Rating based on Product Type and Material
        ip_options = df[
            (df['Product_Type'] == product_type) & 
            (df['Material_Composition'] == material)
        ]['IP_Rating'].dropna().unique()

        if len(ip_options) == 0:
            ip_rating = st.text_input("💧 IP Rating", placeholder="e.g., IP65")
        elif len(ip_options) == 1:
            ip_rating = ip_options[0]
            st.markdown(f"💧 **IP Rating**: `{ip_rating}` (Auto-selected)")
        else:
            ip_rating = st.selectbox("💧 Select IP Rating", sorted(ip_options))

    with col2:
        dimension = st.text_input("📏 Dimensions", placeholder="e.g., 635x749x206mm")
        temp = st.text_input("🌡️ Operating Temperature", placeholder="e.g., -20 to 60 C")
        hazard = st.text_input("☣️ Hazardous Components", placeholder="e.g., None")
        custom_features = st.text_input("✨ Custom Features", placeholder="e.g., Touchscreen Interface")

        weights = df[
            (df['Product_Type'] == product_type) &
            (df['Material_Composition'] == material)
        ]['Weight_kg'].dropna().unique()

        if len(weights) == 1:
            base_weight = weights[0]
            min_wt = max(0, base_weight - 5)
            max_wt = base_weight + 5
            st.markdown(f"⚖️ **Enter Weight (kg)** – must be between `{min_wt}kg` and `{max_wt}kg`")
            weight = st.number_input("Weight (kg)", min_value=min_wt, max_value=max_wt, step=0.1, format="%.2f")
        else:
            st.warning("⚠️ Unable to determine weight range – multiple or no matching weights found.")
            weight = st.number_input("Weight (kg)", min_value=0.0, step=0.1, format="%.2f")

    input_text = f"{product_type} {material} {dimension} {voltage} {temp} {ip_rating} {hazard} {custom_features}"

    st.markdown("---")

    if st.button("🔍 Predict Certificates"):
        pred_global = pipeline_global.predict([input_text])
        pred_india = pipeline_india.predict([input_text])
        global_result = le_global.inverse_transform(pred_global)[0]
        india_result = le_india.inverse_transform(pred_india)[0]

        st.success("✅ Prediction Successful!")
        st.markdown("### 🇮🇳 India Certificate")
        st.info(f"📄 {india_result}")
        st.markdown("### 🌐 Global Certificate")
        st.info(f"📄 {global_result}")

        # Create result dataframe
        result_df = pd.DataFrame({
            "Product Type": [product_type],
            "Material Composition": [material],
            "Dimensions": [dimension],
            "Voltage": [voltage],
            "Operating Temperature": [temp],
            "IP Rating": [ip_rating],
            "Hazardous Components": [hazard],
            "Custom Features": [custom_features],
            "Weight (kg)": [weight],
            "India Certificate": [india_result],
            "Global Certificate": [global_result]
        })

        # CSV download button
        csv_buffer = io.StringIO()
        result_df.to_csv(csv_buffer, index=False)
        st.download_button(
            label="📥 Download Result as CSV",
            data=csv_buffer.getvalue(),
            file_name="certificate_prediction_result.csv",
            mime="text/csv"
        )

# ------------------- Footer -------------------
st.markdown("---")
st.markdown("<p style='text-align: center; color: gray;'>© 2025 CompliTech Group. All rights reserved.</p>", unsafe_allow_html=True)
