import streamlit as st
import pandas as pd
import numpy as np
import joblib
from datetime import datetime

# Load the trained model and encoders
clf = joblib.load('crime_classifier.pkl')  # Load the trained DecisionTree model
label_encoder = joblib.load('label_encoder.pkl')  # Load LabelEncoder
target_encoder = joblib.load('target_encoder.pkl')  # Load TargetEncoder

# Load the dataset
train_data = pd.read_csv('train.csv')

# Precompute the mode values for filling missing data
victim_sex_mode_by_crime = train_data.groupby('Crime_Category')['Victim_Sex'].apply(lambda x: x.mode()[0])
victim_descent_mode_by_sex = train_data.groupby('Victim_Sex')['Victim_Descent'].apply(lambda x: x.mode()[0])
weapon_description_default = 'unknown'  # Default for missing weapon description

# Extract unique values from the dataset for each feature
locations = train_data['Location'].unique().tolist()
victim_sex_options = train_data['Victim_Sex'].dropna().unique().tolist()  # Remove NaN values from Victim_Sex
victim_descent_options = train_data['Victim_Descent'].dropna().unique().tolist()  # Remove NaN values from Victim_Descent
weapon_description_options = train_data['Weapon_Description'].dropna().unique().tolist()  # Remove NaN values from Weapon_Description
status_options = train_data['Status'].dropna().unique().tolist()  # Remove NaN values from Status
time_occurred_labels = ['Night', 'Morning', 'Afternoon', 'Evening']  # Fixed labels for time of occurrence
reported_bins = ['Within 15 days', '15 days to 6 months', '6 months to 1 year', 'greater than 1 year']  # Fixed labels for reported bins
area_names = train_data['Area_Name'].dropna().unique().tolist()  # Extract unique Area Names, remove NaN
area_ids_dict = dict(zip(train_data['Area_Name'], train_data['Area_ID']))  # Create a dictionary for Area Name to Area ID mapping
reporting_districts = train_data['Reporting_District_no'].dropna().unique().tolist()  # Remove NaN values from Reporting Districts
part_1_2_options = train_data['Part 1-2'].dropna().unique().tolist()  # Remove NaN values from Part 1-2
premise_codes = train_data['Premise_Code'].dropna().unique().tolist()  # Remove NaN values from Premise Codes
premise_descriptions = train_data['Premise_Description'].dropna().unique().tolist()  # Remove NaN values from Premise Descriptions

# Create a mapping between Premise_Description and Premise_Code
description_to_code = dict(zip(train_data['Premise_Description'], train_data['Premise_Code']))

# Create a mapping for Status Description to Status code
status_description_to_status = dict(zip(train_data['Status_Description'], train_data['Status']))

# Function to get user input and make predictions
def get_user_input():
    st.subheader("Enter Crime Information:")

    # Manually get inputs for each feature from the dataset's unique values
    location = st.selectbox("Location", locations)
    area_name = st.selectbox("Area Name", area_names)  # Area Name dropdown for selection
    area_id = area_ids_dict[area_name]  # Get corresponding Area_ID from the dictionary
    reporting_district_no = st.selectbox("Reporting District No.", reporting_districts)
    part_1_2 = st.selectbox("Part 1-2", part_1_2_options)
    
    
    victim_age = st.number_input("Victim Age", min_value=0, max_value=100, value=0)  # Victim Age input
    
    
    victim_sex = st.selectbox("Victim Sex", victim_sex_options)  
    victim_descent = st.selectbox("Victim Descent", victim_descent_options)  
    
    # Handle Premise Description with a fallback
    premise_description = st.selectbox("Premise Description", premise_descriptions)
    premise_code = description_to_code.get(premise_description)  # Get corresponding Premise Code

    weapon_description = st.selectbox("Weapon Description", weapon_description_options + [weapon_description_default])
    
    # Status Description and map them to Status codes
    status_description = st.selectbox("Status Description", list(status_description_to_status.keys()))
    status = status_description_to_status[status_description]  # Get the corresponding Status code

    # Date input for Date Reported and Date Occurred
    date_reported = st.date_input("Date Reported", datetime.today())
    date_occurred = st.date_input("Date Occurred", datetime.today())

    # Validate Date Reported cannot be later than today
    if date_reported > datetime.today().date():
        st.error("Date Reported cannot be greater than today's date. Please select a valid Date Reported.")
        date_reported = datetime.today().date()  # Adjust Date Reported to today's date

    # Validate Date Occurred should not be later than Date Reported or current date
    if date_occurred > date_reported:
        st.error("Date Occurred cannot be later than Date Reported. Please select a valid Date Occurred.")
        date_occurred = date_reported  # Adjust Date Occurred to match Date Reported
    
    if date_occurred > datetime.today().date():
        st.error("Date Occurred cannot be later than today's date. Please select a valid Date Occurred.")
        date_occurred = datetime.today().date()  # Adjust Date Occurred to today's date

    # Check if all required fields are filled out
    missing_fields = []
    if not location:
        missing_fields.append('Location')
    if not area_name:
        missing_fields.append('Area Name')
    if not reporting_district_no:
        missing_fields.append('Reporting District No.')
    if not part_1_2:
        missing_fields.append('Part 1-2')
    if victim_age == 0:
        missing_fields.append('Victim Age')
    if not victim_sex:
        missing_fields.append('Victim Sex')
    if not victim_descent:
        missing_fields.append('Victim Descent')
    if not premise_description:
        missing_fields.append('Premise Description')
    if not weapon_description:
        missing_fields.append('Weapon Description')
    if not status_description:
        missing_fields.append('Status Description')
    if date_reported is None:
        missing_fields.append('Date Reported')
    if date_occurred is None:
        missing_fields.append('Date Occurred')

    # If any required fields are missing, show an error message
    if missing_fields:
        st.error(f"Please fill out the following fields: {', '.join(missing_fields)}")

    # Calculate the difference between the two dates in days
    reported_occurred = (date_reported - date_occurred).days

    # Handle reported_occurred value
    if reported_occurred < 0:
        reported_occurred = 0  # Avoid negative days if Date Occurred is later than Date Reported
    
    # Calculate the Reported Bin based on Reported-Occurred
    bins = [0, 15, 180, 365, float('inf')]
    labels = ['Within 15 days', '15 days to 6 months', '6 months to 1 year', 'greater than 1 year']
    reported_bins_value = pd.cut([reported_occurred], bins=bins, labels=labels, right=False, include_lowest=True)[0]

    # Time Occurred Label
    time_occurred_label = st.selectbox("Time Occurred Label", time_occurred_labels) 
    
    # Creating a dictionary with user input
    input_data = {
        'Location': location,
        'Area_ID': area_id,  # Use the corresponding Area_ID
        'Reporting_District_no': reporting_district_no,
        'Part 1-2': part_1_2,
        'Victim_Age': victim_age,
        'Victim_Sex': victim_sex,  # Use the mode value if it was 'Unknown'
        'Victim_Descent': victim_descent,  # Use the mode value if it was 'Unknown'
        'Premise_Code': premise_code,  # Use the Premise Code corresponding to the selected Premise Description
        'Weapon_Description': weapon_description,  # Use 'unknown' if missing
        'Status': status,  # Use Status internally, mapped from Status Description
        'Reported-Occured': reported_occurred,
        'Time_Occurred_Label': time_occurred_label,
        'Reported_bins': reported_bins_value
    }

    # Convert input_data to a pandas DataFrame
    input_df = pd.DataFrame([input_data])

    return input_df


# Function to preprocess user input
def preprocess_user_input(input_df):
    # In this case, we need to ensure that categorical variables are properly encoded
    input_df['Time_Occurred_Label'] = input_df['Time_Occurred_Label'].apply(lambda x: x.strip())  # Clean if necessary
    input_df['Reported_bins'] = input_df['Reported_bins'].apply(lambda x: x.strip())  # Clean if necessary

    # Apply target encoding or label encoding where needed
    # Example: Encoding 'Time_Occurred_Label' and 'Reported_bins' if they are categorical
    input_df['Time_Occurred_Label'] = input_df['Time_Occurred_Label'].map({label: idx for idx, label in enumerate(time_occurred_labels)})
    input_df['Reported_bins'] = input_df['Reported_bins'].map({label: idx for idx, label in enumerate(reported_bins)})

    # Encode other categorical features if necessary here
    input_encoded = target_encoder.transform(input_df)

    return input_encoded


# Prediction function
def predict():
    st.title("Crime Category Prediction")

    # Get user input
    user_input_df = get_user_input()

    
    if st.button("Predict Crime Category"):
        # Check if there were missing fields
        if not any(user_input_df.isnull().any()):
            # Preprocess the input data
            input_encoded = preprocess_user_input(user_input_df)

            # Make prediction using the model
            prediction = clf.predict(input_encoded)
            predicted_label = label_encoder.inverse_transform(prediction)

            # Display the prediction result
            st.subheader(f"Predicted Crime Category: {predicted_label[0]}")
        else:
            st.error("Please fill in all the required fields.")
            
if __name__ == '__main__':
    predict()
