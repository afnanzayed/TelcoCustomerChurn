data = pd.read_csv("/Users/afnanalamri/Desktop/MyProject/TelcoCustomerChurn/raw_data/Customers_with_Churn_Prediction.csv")
data['InternetService'] = data[['InternetService_DSL', 'InternetService_Fiber_optic', 'InternetService_No']].idxmax(axis=1)
data['InternetService'] = data['InternetService'].str.replace('InternetService_', '')

data['Contract'] = data[['Contract_Month_to_month', 'Contract_One_year', 'Contract_Two_year']].idxmax(axis=1)
data['Contract'] = data['Contract'].str.replace('Contract_', '')


data.drop(columns= ['InternetService_DSL', 'InternetService_Fiber_optic', 'InternetService_No', 'Contract_Month_to_month', 'Contract_One_year', 'Contract_Two_year'])



# Visualization Data
churn_by_contract = data.groupby('Contract')['Churn'].sum().reset_index()
service_distribution = data['InternetService'].value_counts().reset_index()
churn_by_service = data.groupby('InternetService')['Churn'].mean().reset_index()
churn_over_time = data.groupby('tenure')['Churn'].sum().reset_index()


# Streamlit Dashboard Layout
st.title("Telecom Customer Churn Analysis")

# Key Metrics (Top Section)
st.subheader("Key Metrics")
col1, col2, col3 = st.columns(3)

churn_rate = (data['Churn'].value_counts(normalize=True)[1] * 100).round(2)
retention_rate = 100 - churn_rate
revenue_loss = 100000  # Example value, replace with actual calculation

col1.metric("Overall Churn Rate", f"{churn_rate}%")
col2.metric("Retention Rate", f"{retention_rate}%")
col3.metric("Revenue Loss", f"${revenue_loss}")

# Customer Segmentation Section
st.subheader("Customer Segmentation")
col4, col5, col6 = st.columns(3)

# Senior Citizen vs Churn (Pie Chart)
senior_churn = data.groupby(['SeniorCitizen', 'Churn']).size().unstack()
senior_churn.plot(kind='pie', y=1, autopct='%1.1f%%', legend=False, figsize=(3, 3), colors=['skyblue', 'orange'])
col4.pyplot(plt.gcf())
plt.clf()

# Dependents vs Churn (Stacked Bar Chart)
dependents_churn = data.groupby(['Dependents', 'Churn']).size().unstack()
dependents_churn.plot(kind='bar', stacked=True, color=['skyblue', 'orange'])
col5.pyplot(plt.gcf())
plt.clf()

# Partner vs Churn (Stacked Bar Chart)
partner_churn = data.groupby(['Partner', 'Churn']).size().unstack()
partner_churn.plot(kind='bar', stacked=True, color=['skyblue', 'orange'])
col6.pyplot(plt.gcf())
plt.clf()

# Service Utilization Section
st.subheader("Service Utilization and Churn")
col7, col8 = st.columns(2)

# Internet Service vs Churn
internet_churn = data.groupby(['InternetService', 'Churn']).size().unstack()
internet_churn.plot(kind='bar', stacked=True, color=['skyblue', 'orange'])
col7.pyplot(plt.gcf())
plt.clf()

# Streaming Services vs Churn
streaming_churn = data[['StreamingTV_No', 'StreamingTV_Yes']].sum()
streaming_churn.plot(kind='bar', color=['skyblue', 'orange'])
col8.pyplot(plt.gcf())
plt.clf()

# Contract and Billing Preferences Section
st.subheader("Contract and Billing Preferences")
col9, col10, col11 = st.columns(3)

# Contract Type vs Churn
contract_churn = data[['Contract_Month_to_month', 'Contract_One_year', 'Contract_Two_year']].sum()
contract_churn.plot(kind='bar', color=['skyblue', 'orange'])
col9.pyplot(plt.gcf())
plt.clf()

# Payment Method vs Churn (Pie Chart)
payment_churn = data[['PaymentMethod_Bank_transfer_(automatic)',
                      'PaymentMethod_Credit_card_(automatic)',
                      'PaymentMethod_Electronic_check',
                      'PaymentMethod_Mailed_check']].sum()
payment_churn.plot(kind='pie', autopct='%1.1f%%', colors=['skyblue', 'orange', 'green', 'purple'])
col10.pyplot(plt.gcf())
plt.clf()

# Paperless Billing vs Churn (Horizontal Bar Chart)
paperless_churn = data[['PaperlessBilling_No', 'PaperlessBilling_Yes']].sum()
paperless_churn.plot(kind='barh', color=['skyblue', 'orange'])
col11.pyplot(plt.gcf())
plt.clf()

# Financial Insights Section
st.subheader("Financial Insights")
col12, col13 = st.columns(2)

# Monthly Charges vs Churn (Scatter Plot)
sns.scatterplot(data=data, x="MonthlyCharges", y="Churn", hue="Churn", palette="coolwarm")
col12.pyplot(plt.gcf())
plt.clf()

# Total Charges vs Tenure (Line Chart)
sns.lineplot(data=data, x="tenure", y="TotalCharges", hue="Churn", palette="coolwarm")
col13.pyplot(plt.gcf())
plt.clf()

# Tenure and Churn Trends Section
st.subheader("Tenure and Churn Trends")
col14, col15 = st.columns(2)

# Tenure vs Churn (Line Chart)
tenure_churn = data.groupby('tenure')['Churn'].mean()
tenure_churn.plot(kind='line', color='orange')
col14.pyplot(plt.gcf())
plt.clf()

# Average Tenure for Churners (Single Value Card)
avg_tenure_churn = data[data['Churn'] == 1]['tenure'].mean()
col15.metric("Avg. Tenure (Churned Customers)", f"{avg_tenure_churn:.1f} months")


