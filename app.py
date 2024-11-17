import streamlit as st
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import xgboost as xgb

# Load the cleaned DataFrame
df_cleaned = pd.read_csv("cleaned_data.csv")

# --- Title and Overview Section ---

# Set page configuration to wide layout
st.set_page_config(
    page_title="Telecom Customer Churn Analysis",
    layout="wide"
)

st.title("Telecom Customer Churn Analysis")
st.markdown("Explore churn patterns and customer behavior to drive retention strategies")

# Add spacing
st.markdown("<br>", unsafe_allow_html=True)

# --- Top Section: Overview of Churn Metrics ---

st.subheader("Overview of Churn Metrics")

col1, col2 = st.columns([1, 1.5])

with col1:
    overall_churn_rate = df_cleaned['Churn'].mean() * 100
    churn_delta = -5
    st.metric("Churn Rate", f"{overall_churn_rate:.2f}%", delta=f"{churn_delta}%", delta_color="normal")
    st.caption("Percentage of customers who left.")

with col2:
    retention_rate = 100 - overall_churn_rate
    retention_delta = +5
    st.metric("Retention Rate", f"{retention_rate:.2f}%", delta=f"{retention_delta}%", delta_color="normal")
    st.caption("Percentage of customers retained.")

# Add a horizontal divider
st.markdown("---")

# --- Middle Section: Feature-wise Churn Distribution ---
st.subheader("Feature-wise Churn Distribution")

# Combine features into a single DataFrame for a stacked bar chart
features_to_merge = ['SeniorCitizen', 'Dependents', 'InternetService',
                     'OnlineSecurity', 'OnlineBackup',
                     'DeviceProtection', 'TechSupport', 'Contract']

# Melt the DataFrame to reshape it for visualization
melted_data = df_cleaned.melt(id_vars=['Churn'], value_vars=features_to_merge,
                              var_name='Feature', value_name='Category')

# Group by Feature, Category, and Churn to count occurrences
stacked_data = melted_data.groupby(['Feature', 'Category', 'Churn']).size().reset_index(name='Count')


# Define consistent color mapping for Churn
color_map = {0: 'blue', 1: 'red'}

# Create the stacked bar chart with labels inside the bars
fig = px.bar(
    stacked_data,
    x='Feature',
    y='Count',
    color='Churn',
    barmode='stack',
    title="Churn Across Multiple Features",
    color_discrete_map=color_map,
    text_auto=True  # Enables labels inside the bars
)

# Customize the layout to remove numbers above the bars
fig.update_traces(textposition='inside')  # Place labels inside the bars

# Display in Streamlit
st.plotly_chart(fig, use_container_width=True)

# Add spacing
st.markdown("<br>", unsafe_allow_html=True)

# --- Bottom Section: Customer Behavior by Financial and Tenure Metrics ---
st.subheader("Customer Behavior by Financial and Tenure Metrics")

# Prepare MonthlyCharges data
monthly_charges_data = df_cleaned.groupby(['MonthlyCharges', 'Churn']).size().reset_index(name='Count')
monthly_charges_data['Feature'] = 'Monthly Charges'

# Prepare Tenure data
tenure_data = df_cleaned.groupby(['tenure', 'Churn']).size().reset_index(name='Count')
tenure_data.rename(columns={'tenure': 'Value'}, inplace=True)  # Rename column for consistency
tenure_data['Feature'] = 'Tenure'

# Standardize MonthlyCharges column name for merging
monthly_charges_data.rename(columns={'MonthlyCharges': 'Value'}, inplace=True)

# Merge MonthlyCharges and Tenure data
combined_data = pd.concat([monthly_charges_data, tenure_data])

# Define custom colors for clarity
custom_colors = {
    'Monthly Charges': 'green',
    'Tenure': 'purple'
}

# Create a merged line plot with distinct colors for each line
fig = px.line(
    combined_data,
    x='Value',
    y='Count',
    color='Feature',
    line_dash='Churn',
    title="Churn Trends: Monthly Charges vs. Tenure by Churn",
    labels={'Value': 'Value (Monthly Charges or Tenure)', 'Count': 'Customer Count'},
    color_discrete_map=custom_colors
)

# Update trace names for a clearer legend
fig.for_each_trace(
    lambda trace: trace.update(
        name=f"{trace.name.split(',')[0]} - {'Churn' if '1' in trace.name else 'Not Churn'}"
    )
)

# Update layout for better readability
fig.update_layout(
    legend_title="Feature and Churn",
    xaxis_title="Value (Monthly Charges or Tenure)",
    yaxis_title="Customer Count",
    hovermode="x unified"  # Combine hover information
)

# Display the chart in Streamlit
st.plotly_chart(fig, use_container_width=True)

# --- Prediction Section ---
st.header("Predict Customer Churn")
st.markdown("Enter customer details to predict if they will churn or not.")

# Load the pre-trained XGBoost model
model = xgb.Booster()
model.load_model("/Users/afnanalamri/Desktop/MyProject/TelcoCustomerChurn/notebooks/best_xgb_model.json")

# Define model feature names
model_feature_names = model.feature_names

# Input fields for customer features
gender = st.selectbox("Gender", options=["Male", "Female"])
senior_citizen = st.selectbox("Senior Citizen", options=["Yes", "No"], help="Select if the customer is a senior citizen.")
partner = st.selectbox("Partner", options=["Yes", "No"])
dependents = st.selectbox("Dependents", options=["Yes", "No"])
tenure = st.slider("Tenure (in months)", min_value=0, max_value=72, value=12)
phone_service = st.selectbox("Phone Service", options=["Yes", "No"])
multiple_lines = st.selectbox("Multiple Lines", options=["No", "Yes", "No phone service"])
internet_service = st.selectbox("Internet Service", options=["DSL", "Fiber optic", "No"])
online_security = st.selectbox("Online Security", options=["No", "Yes", "No internet service"])
online_backup = st.selectbox("Online Backup", options=["No", "Yes", "No internet service"])
device_protection = st.selectbox("Device Protection", options=["No", "Yes", "No internet service"])
tech_support = st.selectbox("Tech Support", options=["No", "Yes", "No internet service"])
streaming_tv = st.selectbox("Streaming TV", options=["No", "Yes", "No internet service"])
streaming_movies = st.selectbox("Streaming Movies", options=["No", "Yes", "No internet service"])
contract = st.selectbox("Contract", options=["Month-to-month", "One year", "Two year"])
paperless_billing = st.selectbox("Paperless Billing", options=["Yes", "No"])
payment_method = st.selectbox(
    "Payment Method",
    options=["Electronic check", "Mailed check", "Bank transfer (automatic)", "Credit card (automatic)"]
)
monthly_charges = st.number_input("Monthly Charges ($)", min_value=0.0, max_value=120.0, value=50.0)
total_charges = st.number_input("Total Charges ($)", min_value=0.0, value=500.0)

# Transform user inputs to match the model's expected feature format
input_data = {
    'gender': 1 if gender == "Male" else 0,
    'SeniorCitizen': 1 if senior_citizen == "Yes" else 0,
    'Partner': 1 if partner == "Yes" else 0,
    'Dependents': 1 if dependents == "Yes" else 0,
    'tenure': tenure,
    'PhoneService': 1 if phone_service == "Yes" else 0,
    'MultipleLines_No': 1 if multiple_lines == "No" else 0,
    'MultipleLines_Yes': 1 if multiple_lines == "Yes" else 0,
    'MultipleLines_No_phone_service': 1 if multiple_lines == "No phone service" else 0,
    'InternetService_DSL': 1 if internet_service == "DSL" else 0,
    'InternetService_Fiber_optic': 1 if internet_service == "Fiber optic" else 0,
    'InternetService_No': 1 if internet_service == "No" else 0,
    'OnlineSecurity_No': 1 if online_security == "No" else 0,
    'OnlineSecurity_Yes': 1 if online_security == "Yes" else 0,
    'OnlineSecurity_No_internet_service': 1 if online_security == "No internet service" else 0,
    'OnlineBackup_No': 1 if online_backup == "No" else 0,
    'OnlineBackup_Yes': 1 if online_backup == "Yes" else 0,
    'OnlineBackup_No_internet_service': 1 if online_backup == "No internet service" else 0,
    'DeviceProtection_No': 1 if device_protection == "No" else 0,
    'DeviceProtection_Yes': 1 if device_protection == "Yes" else 0,
    'DeviceProtection_No_internet_service': 1 if device_protection == "No internet service" else 0,
    'TechSupport_No': 1 if tech_support == "No" else 0,
    'TechSupport_Yes': 1 if tech_support == "Yes" else 0,
    'TechSupport_No_internet_service': 1 if tech_support == "No internet service" else 0,
    'StreamingTV_No': 1 if streaming_tv == "No" else 0,
    'StreamingTV_Yes': 1 if streaming_tv == "Yes" else 0,
    'StreamingTV_No_internet_service': 1 if streaming_tv == "No internet service" else 0,
    'StreamingMovies_No': 1 if streaming_movies == "No" else 0,
    'StreamingMovies_Yes': 1 if streaming_movies == "Yes" else 0,
    'StreamingMovies_No_internet_service': 1 if streaming_movies == "No internet service" else 0,
    'Contract_Month_to_month': 1 if contract == "Month-to-month" else 0,
    'Contract_One_year': 1 if contract == "One year" else 0,
    'Contract_Two_year': 1 if contract == "Two year" else 0,
    'PaperlessBilling_Yes': 1 if paperless_billing == "Yes" else 0,
    'PaperlessBilling_No': 1 if paperless_billing == "No" else 0,
    'PaymentMethod_Electronic_check': 1 if payment_method == "Electronic check" else 0,
    'PaymentMethod_Mailed_check': 1 if payment_method == "Mailed check" else 0,
    'PaymentMethod_Bank_transfer_(automatic)': 1 if payment_method == "Bank transfer (automatic)" else 0,
    'PaymentMethod_Credit_card_(automatic)': 1 if payment_method == "Credit card (automatic)" else 0,
    'MonthlyCharges': monthly_charges,
    'TotalCharges': total_charges
}

# Convert the input data to a DataFrame
input_df = pd.DataFrame([input_data])

# Rearrange the columns to match the model's feature names
input_df = input_df[model_feature_names]

# Convert the DataFrame to DMatrix (XGBoost format)
input_dmatrix = xgb.DMatrix(input_df)


if st.button("Predict"):
    # Make prediction with probabilities
    probabilities = model.predict(input_dmatrix)

    # For binary classification, probabilities are returned as the predicted score for the positive class
    churn_probability = probabilities[0]
    not_churn_probability = 1 - churn_probability  # Probability of Not Churn (class 0)

    # Create a bar chart with Plotly
    fig = go.Figure(
        data=[
            go.Bar(
                x=["Not Churn", "Churn"],
                y=[not_churn_probability, churn_probability],
                marker_color=["blue", "red"],
                text=[f"{not_churn_probability:.2%}", f"{churn_probability:.2%}"],
                textposition="auto",
            )
        ]
    )

    # Customize layout
    fig.update_layout(
        title="Churn Prediction Probabilities",
        xaxis_title="Prediction Outcome",
        yaxis_title="Probability",
        template="plotly_white",
        showlegend=False
    )

    # Display the chart in Streamlit
    st.plotly_chart(fig, use_container_width=True)

    # Display probabilities
    st.write("Churn Probability:", f"{churn_probability:.2%}")
    st.write("Not Churn Probability:", f"{not_churn_probability:.2%}")
