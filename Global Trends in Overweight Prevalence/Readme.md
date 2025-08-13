**Title**

**Global Trends in Overweight Prevalence by Region and Year**

**Objective:**
The goal of this project is to analyze worldwide trends in overweight prevalence, grouped by region and year, to identify the regions with the highest rates and track how these rates have changed over time. This helps in understanding regional disparities and highlighting areas for public health intervention.

**Dataset**
Source: Our World in Data  / WHO Overweight Prevalence Data (or the dataset you used)


**Key Features:**

Country — Country name

Code — Country ISO code

Year — Year of observation

Overweight_Prevalence — Percentage of population considered overweight

Region — Region name (derived from country)

**Methods**
1. Data Cleaning
Removed missing and duplicate values

Standardized column names for consistency

Added a Region column based on country names using a mapping dictionary

2. Data Processing
Grouped data by Region and Year to calculate average overweight prevalence

Extracted top 10 regions for each year based on prevalence

Sorted values to prepare for visualization

3. Data Visualization
Used Matplotlib, Seaborn, and Plotly for interactive and static charts

**Created:**
Bar plots to show top 10 regions per year

Line plots to track trends in overweight prevalence across years

Heatmaps for quick regional comparisons

**Key Findings**
Certain regions consistently rank high in overweight prevalence across multiple years

Some regions show a steady increase over the decades, suggesting lifestyle changes and dietary shifts

Other regions remain relatively low, often due to different socio-economic and cultural factors

**Conclusion**
The analysis reveals significant disparities in overweight prevalence between regions. Public health policies should be tailored regionally, focusing on prevention strategies where rates are rising rapidly. Future work could combine this dataset with GDP, diet, and urbanization data to better understand underlying causes.

**Tools Used**
Python — Core programming language

Pandas — Data cleaning and analysis

Matplotlib / Seaborn / Plotly — Data visualization

Jupyter Notebook — Interactive analysis environment
