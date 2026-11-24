import streamlit as st
import joblib
import pandas as pd

# Page Configuration
st.set_page_config(page_title="Customer Classification app", page_icon="📊", layout="centered")


st.markdown("=" * 60)
st.title("📊 Customer Classification Dashboard")
st.write("Enter the transaction details to classify which group a customer belong to.")
st.markdown("=" * 60)

# Function to load the model using joblib
@st.cache_resource
def load_model(model_path="customer_segementation.pki"):
    """
    Loads a machine learning model from disk using joblib.
    Make sure your trained model file is in the same directory 
    or provide the correct path.
    """
    try:
        model = joblib.load(model_path)
        return model
    except FileNotFoundError:
        st.warning(f"Model file '{model_path}' not found. Please upload or place the model file in the directory.")
        return None
    except Exception as e:
        st.error(f"Error loading model: {e}")
        return None

# Load the model
model = load_model("customer_segementation.pki")

# Input Form for Features
with st.form("prediction_form"):
    st.subheader("Feature Inputs")
    
    # Sales input (Min: 0.99, Max: 13999.96)
    sales = st.number_input(
        label="Sales ($)",
        min_value=0.99,
        max_value=13999.96,
        value=500.00,
        step=0.01,
        help="Enter the sales value between 0.99 and 13999.96"
    )
    
    # Quantity input (Min: 1, Max: 14)
    quantity = st.number_input(
        label="Quantity",
        min_value=1,
        max_value=14,
        value=3,
        step=1,
        help="Enter the quantity between 1 and 14"
    )
    
    # Profit input (Min: -3399.98, Max: 6719.9808)
    profit = st.number_input(
        label="Profit ($)",
        min_value=-3399.98,
        max_value=6719.9808,
        value=50.00,
        step=0.01,
        help="Enter the profit value between -3399.98 and 6719.9808"
    )
    
    # Submit Button
    submitted = st.form_submit_button("Predict")

# Handle Prediction Logic
if submitted:
    if model is None:
        st.error("Cannot make predictions because the model is not loaded.")
    else:
        # Construct feature array/dataframe in the exact order your model expects
        input_data = pd.DataFrame([[sales, quantity, profit]], columns=['Sales', 'Quantity', 'Profit'])
        
        try:
            # Make prediction
            prediction = model.predict(input_data)
            
            st.success("Prediction generated successfully!")
            st.write("### Result:")
            if prediction[0] == 0:
                st.write("Customer falls under the Low Purchase group.")
            else:
                st.write("Customer falls under the High Purchase group.")
            
        except Exception as e:
            st.error(f"An error occurred during prediction: {e}")
