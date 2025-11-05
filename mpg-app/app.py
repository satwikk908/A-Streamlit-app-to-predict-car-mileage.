import streamlit as st
import pandas as pd
import pickle
import numpy as np

# Set the title of the web app
st.title(' Auto Mileage Predictor (Indian Units)')
st.write("This app predicts the fuel efficiency (Mileage) of a car using your inputs.")

# --- 1. Load the Model ---
try:
    with open('rf_model.pkl', 'rb') as file:
        model = pickle.load(file)
except FileNotFoundError:
    st.error("Error: Model file (rf_model.pkl) not found. Make sure it's in the same directory.")
    st.stop()
except Exception as e:
    st.error(f"Error loading model: {e}")
    st.stop()


# --- 2. Create the Sidebar for User Inputs ---
st.sidebar.header('User Input Features')

cylinders = st.sidebar.selectbox('Cylinders', (4, 6, 8), index=1)
displacement_cc = st.sidebar.slider('Displacement (cc)', 1100.0, 8200.0, 3150.0)
horsepower = st.sidebar.slider('Horsepower (hp)', 46.0, 230.0, 104.0)
weight = st.sidebar.slider('Weight (lbs)', 1600.0, 5200.0, 2970.0)
acceleration = st.sidebar.slider('Acceleration (0-60 mph in sec)', 8.0, 25.0, 15.5)
model_year = st.sidebar.slider('Model Year (e.g., 72 for 1972)', 70, 82, 76)
origin = st.sidebar.selectbox('Origin (1=USA, 2=Europe, 3=Japan)', (1, 2, 3), index=0)

# This logic is fine
origin_1 = 1 if origin == 1 else 0
origin_2 = 1 if origin == 2 else 0
origin_3 = 1 if origin == 3 else 0


# --- 3. Prediction Logic ---
if st.sidebar.button('Predict Mileage'):
    
    # --- CHANGE 1: Convert integer keys to STRING keys ---
    # Use '1', '2', '3' (with quotes)
    input_data = {
        'cylinders': cylinders,
        'displacement': (displacement_cc / 16.3871), # Convert cc to cu. in.
        'horsepower': horsepower,
        'weight': weight,
        'acceleration': acceleration,
        'model_year': model_year,
        '1': origin_1,  # Use string '1'
        '2': origin_2,  # Use string '2'
        '3': origin_3   # Use string '3'
    }

    # --- CHANGE 2: Convert integer names to STRING names in the list ---
    # Use '1', '2', '3' (with quotes)
    model_features = [
        'cylinders', 'displacement', 'horsepower', 'weight', 'acceleration',
        'model_year', '1', '2', '3'
    ]
    
    # Create the DataFrame that goes to the model
    input_df = pd.DataFrame([input_data])
    input_df = input_df[model_features]

    # 3.3: Make the prediction
    try:
        prediction = model.predict(input_df)
        
        # 3.4: Display the result
        st.success(f'Predicted Mileage: **{prediction[0]:.2f} MPG**')
        
        st.write("---")
        st.write("Inputs provided:")
        
        # Create a clean display for the user (showing CC)
        display_data = {
            'Feature': ['Cylinders', 'Displacement (cc)', 'Horsepower (hp)', 'Weight (lbs)', 'Acceleration', 'Model Year', 'Origin'],
            'Value': [cylinders, displacement_cc, horsepower, weight, acceleration, model_year, origin]
        }
        display_df = pd.DataFrame(display_data).set_index('Feature')
        st.dataframe(display_df.style.format("{:.2f}"))
        
    except Exception as e:
        st.error(f"Error during prediction: {e}")

else:
    # This message shows before the button is clicked
    st.info('Adjust the sliders and select options in the sidebar, then click "Predict Mileage".')