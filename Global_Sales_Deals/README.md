
# Sales Transaction Insights Analysis

## Project Overview
This **Sales Transaction Insights** project provides a detailed analysis of sales transaction data to uncover insights that drive business decisions. Using **pandas**, **numpy**, **matplotlib**, and **seaborn**, the notebook covers steps such as data cleaning, exploratory data analysis (EDA), and visualizing key business metrics, with the goal of optimizing sales strategies.

---

## Analysis Process

### **1. Data Loading and Basic Information**
The analysis begins by loading the sales transaction dataset using pandas. The dataset consists of key columns such as:
- **Customer ID**: Unique identifier for each customer.
- **Customer Name**: Name of the customer.
- **Region**: Sales region.
- **Sales Representative**: Name of the sales representative handling the deal.
- **Product**: Product sold.
- **Lead Source**: The lead source for the transaction.
- **Revenue**: The deal size in USD.
- **Date**: The transaction date.

After loading the dataset, basic information such as data types, missing values, and summary statistics are explored.

```python
import pandas as pd
# Loading Dataset
df = pd.read_csv('sales_001_dataset.csv')
# Basic information
df.info()
df.describe()
```

### **2. Data Cleaning and Preprocessing**
This phase focuses on transforming the data to ensure it is in a proper format for analysis:
- **Renaming Columns**: Columns are renamed to more meaningful names (e.g., ‘Deal Size (USD)’ to ‘Revenue’).
- **Converting Data Types**: The `Deal Date` column is converted to a datetime object for further analysis.
- **Handling Missing Values**: The dataset is checked for any missing values, and since there are none, no further action is required.
- **Duplicate Values**: The dataset is checked for duplicates, and no duplicate entries are found.

```python
# Renaming columns
df.rename(columns={'Sales Rep':'Sales Representative','Deal Size (USD)' : 'Revenue', 'Deal Date' : 'Date'}, inplace=True)
# Converting Date to datetime
df['Date'] = pd.to_datetime(df['Date'])
```

### **3. Exploratory Data Analysis (EDA)**
EDA helps in understanding the data and discovering patterns:
- **Univariate Analysis**: A histogram is used to visualize the distribution of **Revenue**.

```python
sns.histplot(df['Revenue'], kde=True)
plt.title('Revenue Distribution')
plt.show()
```

- **Bivariate Analysis**: A boxplot is used to explore the relationship between **Sales Representative** and **Revenue**.

```python
sns.boxplot(x='Sales Representative', y='Revenue', data=df)
plt.xticks(rotation=45)
plt.title('Sales Rep vs Deal Size')
plt.show()
```

- **Categorical Data Distribution**: Counts of different **Product**, **Lead Source**, **Region**, and **Sales Representative** are displayed to explore the data distribution across categories.

### **4. One-Hot Encoding**
To prepare categorical variables for machine learning or further statistical analysis, **one-hot encoding** is applied to the categorical columns: **Region**, **Sales Representative**, **Product**, and **Lead Source**.

```python
df = pd.get_dummies(df, columns=['Region', 'Sales Representative', 'Product', 'Lead Source'])
```

### **5. Correlation Matrix**
A correlation matrix is generated to examine the relationships between the numeric columns in the dataset. A heatmap visualization helps in identifying any significant correlations.

```python
corr_matrix = df.select_dtypes(include=['number', 'bool']).corr()
sns.heatmap(corr_matrix, annot=True, fmt=".2f", cmap='coolwarm')
plt.title('Correlation Matrix')
plt.show()
```

### **6. Trend Analysis**
- **Monthly Deal Size Trend**: A time-series analysis is performed to visualize the monthly deal size trend.

```python
monthly_sales['Year-Month'] = monthly_sales['Year-Month'].dt.to_timestamp()
sns.lineplot(x='Year-Month', y='Revenue', data=monthly_sales, marker='o')
plt.xticks(rotation=45)
plt.title('Monthly Deal Size Trend')
plt.show()
```

The line plot shows fluctuations in sales over time, highlighting any seasonal or cyclical trends.

---

## Installation Instructions

1. **Clone the repository**:
   ```bash
   git clone <https://github.com/Mrdevloper6667/Data-Analysis-Projects/tree/main/Global_Sales_Deals>
   ```

2. **Install the required dependencies**:
   Make sure you have Python 3.x installed, then install the necessary libraries:
   ```bash
   pip install pandas numpy matplotlib seaborn
   ```

3. **Open the notebook**:
   Open the Jupyter notebook to start the analysis:
   ```bash
   jupyter notebook Sales_Transaction_Insights.ipynb
   ```

---

## License Information
This project is licensed under the MIT License. See the [LICENSE](LICENSE) file for details.
