import streamlit as st
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import xgboost as xgb

# Load the cleaned DataFrame
df_cleaned = pd.read_csv("cleaned_data.csv")

# Load the pre-trained XGBoost model
model = xgb.Booster()
model.load_model("/Users/afnanalamri/Desktop/MyProject/TelcoCustomerChurn/notebooks/best_xgb_model.json")

# Define model feature names
model_feature_names = model.feature_names

# Set page configuration
st.set_page_config(
    page_title="Telecom Customer Churn Dashboard",
    layout="wide"
)

# Sidebar Navigation
st.sidebar.title("Dashboard Sections")
page = st.sidebar.radio("Explore", ["Customer Churn Analysis", "Customer Prediction"])

# Page 1: Churn Analysis
if page == "Customer Churn Analysis":
    st.title("Telecom Customer Churn Analysis")
    st.markdown("Explore churn patterns and customer behavior to drive retention strategies")

    # Add spacing
    st.markdown("<br>", unsafe_allow_html=True)

    # --- Top Section: Overview of Churn Metrics ---
    st.subheader("Overview of Churn Metrics 📊")
    col1, col2, col3, col4 = st.columns([1.5, 1.5, 1.5, 1.5])

    # Calculate Total Revenue and Total Customers
    total_customers = df_cleaned.shape[0]
    total_revenue = (df_cleaned["MonthlyCharges"] * df_cleaned["tenure"]).sum()

    with col1:
        overall_churn_rate = df_cleaned['Churn'].mean() * 100
        st.metric("Churn Rate 📉", f"{overall_churn_rate:.2f}%")
        st.caption("Percentage of customers who left")

    with col2:
        retention_rate = 100 - overall_churn_rate
        st.metric("Retention Rate 💚", f"{retention_rate:.2f}%")
        st.caption("Percentage of customers retained ")

    with col3:
        st.metric("Total Customers 👥", f"{total_customers:,}")
        st.caption("Number of customers ")

    with col4:
        st.metric("Total Revenue ($) 💵", f"${total_revenue:,.2f}")
        st.caption("Estimated revenue from all customers ")



    # Add a horizontal divider
    st.markdown("---")

    # --- Middle Section: Feature-wise Churn Distribution ---
    st.subheader("Feature-wise Churn Distribution 📈")

    features_to_merge = ['SeniorCitizen', 'Dependents', 'InternetService',
                         'OnlineSecurity', 'OnlineBackup',
                         'DeviceProtection', 'TechSupport', 'Contract']

    melted_data = df_cleaned.melt(id_vars=['Churn'], value_vars=features_to_merge,
                                  var_name='Feature', value_name='Category')

    stacked_data = melted_data.groupby(['Feature', 'Category', 'Churn']).size().reset_index(name='Count')
    color_map = {0: 'blue', 1: 'red'}

    fig = px.bar(
        stacked_data,
        x='Feature',
        y='Count',
        color='Churn',
        barmode='stack',
        title="Churn Across Multiple Features",
        color_discrete_map=color_map,
        text_auto=True
    )
    fig.update_traces(textposition='inside')
    st.plotly_chart(fig, use_container_width=True)

    # Add spacing
    st.markdown("<br>", unsafe_allow_html=True)

    # --- Bottom Section: Customer Behavior by Financial and Tenure Metrics ---
    st.subheader("Customer Behavior by Financial and Tenure Metrics ⏳📈 ")

    monthly_charges_data = df_cleaned.groupby(['MonthlyCharges', 'Churn']).size().reset_index(name='Count')
    monthly_charges_data['Feature'] = 'Monthly Charges'

    tenure_data = df_cleaned.groupby(['tenure', 'Churn']).size().reset_index(name='Count')
    tenure_data.rename(columns={'tenure': 'Value'}, inplace=True)
    tenure_data['Feature'] = 'Tenure'

    monthly_charges_data.rename(columns={'MonthlyCharges': 'Value'}, inplace=True)
    combined_data = pd.concat([monthly_charges_data, tenure_data])

    custom_colors = {'Monthly Charges': 'green', 'Tenure': 'purple'}

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

    fig.for_each_trace(
        lambda trace: trace.update(
            name=f"{trace.name.split(',')[0]} - {'Churn' if '1' in trace.name else 'Not Churn'}"
        )
    )

    fig.update_layout(
        legend_title="Feature and Churn",
        xaxis_title="Value (Monthly Charges or Tenure)",
        yaxis_title="Customer Count",
        hovermode="x unified"
    )
    st.plotly_chart(fig, use_container_width=True)

# Page 2: Customer Prediction
elif page == "Customer Prediction":
    st.title("Predict Customer Churn 🔍")
    st.markdown("Enter customer details to predict if they will churn or not.")

    # Input fields for customer features
    gender = st.selectbox("Gender", options=["Male", "Female"])
    senior_citizen = st.selectbox("Senior Citizen", options=["Yes", "No"])
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

    input_df = pd.DataFrame([input_data])
    input_df = input_df[model_feature_names]
    input_dmatrix = xgb.DMatrix(input_df)

    if st.button("Predict 🔮"):
        probabilities = model.predict(input_dmatrix)
        churn_probability = probabilities[0]
        not_churn_probability = 1 - churn_probability

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
        fig.update_layout(
            title="Churn Prediction Probabilities",
            xaxis_title="Prediction Outcome",
            yaxis_title="Probability",
            template="plotly_white",
            showlegend=False
        )
        st.plotly_chart(fig, use_container_width=True)
        st.write("Churn Probability:", f"{churn_probability:.2%}")
        st.write("Not Churn Probability:", f"{not_churn_probability:.2%}")
