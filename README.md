import pandas as pd
import numpy as np


# Load Excel File
File_Path = r"C:\Users\Admin\Desktop\EXCEL PROJECT\Mobile_Sales_Data.xlsx"
df = pd.read_excel(File_Path)
print(df.head()) 


# Ensure 'Units Sold' and 'Price Per Unit' are numeric
df['Units Sold'] = pd.to_numeric(df['Units Sold'], errors='coerce')
df['Price Per Unit'] = pd.to_numeric(df['Price Per Unit'], errors='coerce')

# Drop rows with NaN values in 'Units Sold' and 'Price Per Unit'
df = df.dropna(subset=['Units Sold', 'Price Per Unit'])

# Convert DataFrame to NumPy Array
data = df.to_numpy()

# Columns indices for relevant columns
units_sold_idx = df.columns.get_loc('Units Sold')
price_per_unit_idx = df.columns.get_loc('Price Per Unit')
brand_idx = df.columns.get_loc('Brand')
city_idx = df.columns.get_loc('City')
payment_method_idx = df.columns.get_loc('Payment Method')
customer_age_idx = df.columns.get_loc('Customer Age')
customer_ratings_idx = df.columns.get_loc('Customer Ratings')
mobile_model_idx = df.columns.get_loc('Mobile Model')
day_name_idx = df.columns.get_loc('Day Name')


# Basic statistical analysis
Total_Unite_Sold = np.sum(data[:, units_sold_idx])
Total_Price_Per_Unite = np.sum(data[:, price_per_unit_idx])


print("Total Units Sold:", Total_Unite_Sold)
print("Total Price Per Unit:", Total_Price_Per_Unite)
print()


# Average Units Sold and Revenue per Model
Average_Unite_Sold = np.mean(data[:, units_sold_idx])
Average_Price_Per_Unite = np.mean(data[:, price_per_unit_idx])

print("Average Units Sold per Model:", Average_Unite_Sold)
print("Average Price Per Unit Model:", Average_Price_Per_Unite)
print()


# Model with Maximum and Minimum Sales
max_unite_index = np.argmax(data[:, units_sold_idx])
min_unite_index = np.argmin(data[:, units_sold_idx])

print("Model with Maximum Units Sold:", data[max_unite_index, mobile_model_idx])
print("Model with Minimum Units Sold:", data[min_unite_index, mobile_model_idx])

# Variance and Standard Deviation
variance = np.var(data[:, units_sold_idx])
std_deviation = np.std(data[:, units_sold_idx])

print("Variance:", variance)
print("Standard Deviation:", std_deviation)

# 90th Percentile
percentile_90 = np.percentile(data[:, units_sold_idx], 90)
print("90th Percentile:", percentile_90)
print()


# Cumulative Sum and Product
cumsum = np.cumsum(data[:, units_sold_idx])
cumprod = np.cumprod(data[:, units_sold_idx])


print("Cumulative Sum:", cumsum)
print("Cumulative Product (first 10 values):", cumprod[:10])
print()


# High Sales Models
high_sales = data[data[:, units_sold_idx] > 6]
print("High Sales Mobile Models:", high_sales)
print()


#Analysis Functions

# Sales by Brand
brands, units_sold_by_brand = np.unique(data[:, brand_idx], return_counts=True)
print("Units Sold by Brand:")
for brand, units in zip(brands, units_sold_by_brand):
    print(f"{brand}: {units}")
print()

# Top Selling Brands (Top 5)
top_brands_idx = np.argsort(units_sold_by_brand)[::-1][:5]
print("Top 5 Selling Brands:")
for idx in top_brands_idx:
    print(f"{brands[idx]}: {units_sold_by_brand[idx]}")
print()

# Customer Demographics - Average Age by City

cities, customer_ages = np.unique(data[:, city_idx], return_inverse=True)
average_age_by_city = [np.mean(data[customer_ages == i, customer_age_idx]) for i in range(len(cities))]
print("Average Customer Age by City:")
for city, avg_age in zip(cities, average_age_by_city):
    print(f"{city}: {avg_age:.2f}")
print()

# Payment Methods Analysis
payment_methods, payment_counts = np.unique(data[:, payment_method_idx], return_counts=True)
print("Sales by Payment Method:")
for method, count in zip(payment_methods, payment_counts):
    print(f"{method}: {count}")
print()

# Customer Ratings
models, customer_ratings = np.unique(data[:, mobile_model_idx], return_inverse=True)
average_ratings_by_model = [np.mean(data[customer_ratings == i, customer_ratings_idx]) for i in range(len(models))]
print("Average Customer Ratings by Mobile Model:")
for model, avg_rating in zip(models, average_ratings_by_model):
    print(f"{model}: {avg_rating:.2f}")
print()

# Sales Trends - Total Sales by Day of the Week+
days, sales_by_day = np.unique(data[:, day_name_idx], return_counts=True)
print("Total Sales by Day of the Week:")
for day, sales in zip(days, sales_by_day):
    print(f"{day}: {sales}")
print()
