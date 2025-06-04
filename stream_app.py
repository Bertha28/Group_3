import joblib
joblib.dump(xgb_model, 'xgb_model.pkl')
joblib.dump(scaler, 'scaler.pkl')

import streamlit as st


# Load the trained model and scaler
try:
    xgb_model = joblib.load('xgb_model.pkl')
    scaler = joblib.load('scaler.pkl')
except FileNotFoundError:
    st.error("Model or scaler file not found. Please make sure 'xgb_model.pkl' and 'scaler.pkl' are in the same directory.")
    st.stop() # Stop execution if files are not found
#Create Streamlit UI elements for user
st.title("Credit Card Fraud Detection")
st.write("Enter the transaction details to predict if it's fraudulent.")

# Create input fields for the features
# You need to add input fields for all the features your model was trained on.
# Make sure the data types match the training data.
# Example (replace with your actual features):
amt = st.number_input("Transaction Amount", value=0.0)
city_pop = st.number_input("City Population", value=0)
lat = st.number_input("Latitude", value=0.0)
long = st.number_input("Longitude", value=0.0)
merch_lat = st.number_input("Merchant Latitude", value=0.0)
merch_long = st.number_input("Merchant Longitude", value=0.0)
time_diff_min = st.number_input("Time Difference from Previous Transaction (minutes)", value=0.0)
amount_zscore = st.number_input("Transaction Amount Z-score (per user)", value=0.0)
# Add all other relevant feature inputs here...

# You will also need to handle the categorical and temporal features you engineered.
# For example, for one-hot encoded 'category' and 'job' features, you might provide
# dropdown menus or selectboxes for the user to choose the category/job, and then
# create the corresponding dummy variables in the input DataFrame.
# Similarly, for 'day_of_week', you might ask for the day or date and derive the day of the week.

# Create a DataFrame from the user input
# Make sure the column names and order match the training data's columns after preprocessing.
# You'll need to create dummy variables for categorical inputs and derive temporal features
# just like you did in your notebook's feature engineering section.
# This part requires careful implementation based on your exact feature engineering steps.
input_data = pd.DataFrame({
    'amt': [amt],
    'city_pop': [city_pop],
    'lat': [lat],
    'long': [long],
    'merch_lat': [merch_lat],
    'merch_long': [merch_long],
    'time_diff_min': [time_diff_min],
    'amount_zscore': [amount_zscore],
    # Add all other feature columns based on user input and feature engineering
})

# Apply the same scaling used during training
# Ensure that the order of columns in input_data is the same as the order
# of columns that were used to fit the scaler initially.
try:
    # Select only numerical columns from the input data for scaling
    numerical_input = input_data[scaler.feature_names_in_] # Assuming scaler stores feature names
    input_scaled = scaler.transform(numerical_input)
    # Reconstruct the DataFrame with scaled numerical features and other features
    input_processed = pd.DataFrame(input_scaled, columns=scaler.feature_names_in_)
    # Add back any non-numerical features that were not scaled
    # (e.g., dummy variables, etc.) - this requires careful handling.
    # For simplicity in this example, we assume all features passed to the scaler were numerical.
    # You will need to adapt this based on your specific feature set.

    # Ensure the columns of the input_processed DataFrame match the columns
    # the model was trained on, and in the same order.
    # This is a critical step. You might need to reindex the columns.
    # Example: input_processed = input_processed[X_train.columns]
    # where X_train.columns is the list of columns your model was trained with.
    # If you dropped columns before training, make sure they are not in the input data
    # passed to the model prediction.

    # Predict using the loaded model
    prediction = xgb_model.predict(input_processed)
    prediction_proba = xgb_model.predict_proba(input_processed)[:, 1]

    st.subheader("Prediction Result:")
    if prediction[0] == 1:
        st.error(f"Fraudulent Transaction Detected (Probability: {prediction_proba[0]:.4f})")
    else:
        st.success(f"Legitimate Transaction (Probability: {prediction_proba[0]:.4f})")

except Exception as e:
    st.error(f"An error occurred during prediction: {e}")
    st.write("Please ensure the input data and preprocessing steps match the training data.")
