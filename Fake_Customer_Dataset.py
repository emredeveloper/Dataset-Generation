import pandas as pd
import numpy as np
from faker import Faker
from sklearn.preprocessing import MinMaxScaler

# Initialize Faker
fake = Faker()

# Number of synthetic records
num_records = 1000

# Generate synthetic customer data
data = {
    "Customer_ID": [fake.uuid4() for _ in range(num_records)],
    "Name": [fake.name() for _ in range(num_records)],
    "Age": np.random.randint(18, 80, num_records),
    "Gender": np.random.choice(["Male", "Female", "Other"], num_records),
    "Email": [fake.email() for _ in range(num_records)],
    "Phone": [fake.phone_number() for _ in range(num_records)],
    "Country": [fake.country() for _ in range(num_records)],
    "Signup_Date": [fake.date_between(start_date='-5y', end_date='today') for _ in range(num_records)],
    "Annual_Income": np.random.randint(20000, 150000, num_records),
    "Purchase_Amount": np.round(np.random.uniform(10, 5000, num_records), 2),
    "Loyalty_Score": np.round(np.random.uniform(0, 1, num_records), 2)  # Normalized score
}

# Convert to DataFrame
df = pd.DataFrame(data)

# Normalize numerical columns (optional)
scaler = MinMaxScaler()
df[['Annual_Income', 'Purchase_Amount']] = scaler.fit_transform(df[['Annual_Income', 'Purchase_Amount']])

# Display sample data
print(df.head())

# Save to CSV (optional)
df.to_csv("synthetic_customer_data.csv", index=False)
