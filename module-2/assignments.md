# Module 2: Programming and Mathematical Foundations — Activities & Assignments

---

## Activity 1: Python Data Preparation Lab (Retail Dataset)

**Objective:** Load, inspect, clean, and prepare a structured retail dataset using Python and Pandas. This is the bread and butter of any data role — you will do this hundreds of times in your career, so build the muscle memory now.

**Dataset:** [Superstore Sales Dataset](https://www.kaggle.com/datasets/vivek468/superstore-dataset-final) from Kaggle. Download the CSV and place it in your working directory.

### Tasks

1. **Import the data** using Pandas. Load it into a DataFrame and display the first 10 rows.
2. **Inspect the structure:**
   - Run `.info()` — check data types and non-null counts.
   - Run `.describe()` — look at summary statistics for numerical columns.
   - Check `.shape` — how many rows and columns?
   - Identify which columns have missing values and how many.
3. **Handle missing values:**
   - For numerical columns: fill with mean or median (choose one and justify why).
   - For categorical columns: fill with mode.
   - Document how many values you filled and in which columns.
4. **Remove duplicates:**
   - Check for duplicate rows using `.duplicated()`.
   - Remove them and report how many were dropped.
5. **Convert data types:**
   - Convert `Order Date` and `Ship Date` to datetime using `pd.to_datetime()`.
   - Verify the conversion worked with `.dtypes`.
6. **Create derived features:**
   - `total_spending_per_customer`: Total sales grouped by Customer ID.
   - `order_frequency`: Number of orders per customer.
   - `avg_order_value`: Average sales amount per order per customer.
   - Merge these back into the main DataFrame or create a separate customer-level summary table.

### Tips for Students

- Always look at your data before doing anything to it. `df.head()`, `df.info()`, `df.describe()` — run these first, every time. No exceptions.
- When filling missing values, median is more robust than mean if your data has outliers. Check the distribution before deciding.
- `pd.to_datetime()` can fail silently if dates are in unexpected formats. After conversion, spot-check a few values to make sure they parsed correctly.
- If you get a `SettingWithCopyWarning`, you are probably modifying a slice instead of the original DataFrame. Use `.loc[]` or `.copy()` to avoid this.
- `groupby()` is your best friend for derived features. Learn it well — `df.groupby('Customer ID')['Sales'].sum()` is a pattern you will use constantly.
- Save your cleaned DataFrame to a new CSV at the end. Never overwrite the original file.

### Reflection Questions

1. Why is it important to inspect data before cleaning it? What would happen if you skipped `.info()` and `.describe()`?
2. You chose mean or median for filling numerical missing values. In what situation would the other choice be better?
3. What is the difference between a row-level DataFrame (one row per order) and a customer-level summary? When would you use each?
4. If two rows have identical values in every column, are they always duplicates? Can you think of a case where they might not be?

### Submission Requirements

- Cleaned dataset saved as a new CSV file (not overwriting the original).
- Jupyter notebook with all code cells executed and visible output.
- Short text explanation (2-3 sentences) after each major step explaining what you did and why.

### What a Good Submission Looks Like

- Code runs top-to-bottom without errors. Someone else can open your notebook and hit "Run All" successfully.
- Every cleaning decision is justified, not just executed. "I used median because the Sales column has right-skewed outliers" beats "I filled nulls with median."
- Derived features are correctly computed and verified with a sanity check (e.g., spot-checking one customer manually).

**Grading Hints:** Full marks require working code, clean output, and written justifications. Code-only submissions without explanations cap at 75%. If derived features are missing or incorrect, that is a significant deduction.

---

## Activity 2: Data Visualization and Trend Analysis

**Objective:** Create meaningful visualizations from the cleaned Superstore dataset and interpret the patterns you find. A chart without interpretation is just a picture.

**Prerequisite:** Use the cleaned dataset from Activity 1.

### Tasks

1. **Distribution plots:**
   - Histogram of `Sales` — what does the distribution look like? Is it normal, skewed, bimodal?
   - Boxplot of `Profit` — where are the quartiles? Any visible outliers?

2. **Category-wise trends:**
   - Bar chart of total sales by `Category` (Technology, Furniture, Office Supplies).
   - Bar chart of total sales by `Region`.
   - Which category and region generate the most revenue?

3. **Outlier identification:**
   - Use the boxplot from step 1 to visually identify outliers in Profit.
   - Calculate the IQR (Interquartile Range) and determine the outlier boundaries numerically.
   - How many data points fall outside these boundaries?

4. **Correlation heatmap:**
   - Select all numerical columns (Sales, Quantity, Discount, Profit).
   - Create a correlation heatmap using Seaborn's `heatmap()`.
   - Which pairs have the strongest positive/negative correlation?

5. **Time-based trend:**
   - Plot monthly total sales over time (line chart).
   - Are there seasonal patterns? Any notable spikes or dips?
   - Add a title and axis labels that make the chart self-explanatory.

### Tips for Students

- Always label your axes. A chart with "0, 1, 2, 3" on the x-axis tells nobody anything.
- Use `figsize=(10, 6)` or similar — default Matplotlib sizes are often too small to read.
- For bar charts, sort the bars by value (descending) so the ranking is immediately obvious.
- Seaborn's `heatmap()` with `annot=True` prints correlation values on the cells. Use it.
- When interpreting a correlation heatmap, remember: correlation is not causation. A strong correlation between Discount and Profit does not mean discounts cause profit changes — it means they move together (or opposite).
- For time series: resample your data to monthly using `df.resample('M', on='Order Date')['Sales'].sum()`.

### Reflection Questions

1. The Sales histogram is likely right-skewed. Why does this happen in retail data? What would it mean if it were perfectly normal?
2. You found outliers in Profit. Should you remove them? What information might you lose if you do?
3. If Discount and Profit have a negative correlation, does that mean the company should stop giving discounts? What other factors might explain this?
4. Looking at your monthly sales trend — if you were a store manager, what action would you take based on this chart?

### Submission Requirements

- Jupyter notebook with all visualizations rendered (not just code — the plots must be visible).
- Written interpretation for each plot: 2-3 sentences explaining what the chart shows and what it means.
- All charts must have titles, axis labels, and appropriate formatting.

### What a Good Submission Looks Like

- Every chart is paired with a written interpretation. The interpretation goes beyond description ("Sales is highest in Technology") to insight ("Technology dominates sales likely because of high-ticket items like phones and laptops, despite lower order volume").
- Outlier analysis includes both visual and numerical methods.
- Time series chart clearly shows the trend with proper date formatting on the x-axis.

**Grading Hints:** Charts without interpretation are worth half credit. Messy or unreadable charts (no labels, tiny size, overlapping text) lose points. The reflection questions matter — they test whether you understood what you plotted.

---

## Activity 3: Mathematical Foundations in Practice

**Objective:** Connect the mathematical concepts from lectures to actual data operations. The goal here is interpretation, not derivation. You do not need to prove theorems — you need to understand what the math means and why it matters for ML.

### Tasks

1. **Vectors and distances:**
   - Pick any 3 products from the Superstore dataset.
   - Represent each product as a 3D feature vector: `[Price, Quantity, Discount]`.
   - Compute the Euclidean distance between each pair of products (3 pairs total).
   - Which two products are most "similar" based on distance? Does this match your intuition?
   - Do this manually first (pen and paper or calculator), then verify with NumPy: `np.linalg.norm(a - b)`.

2. **Linear combinations:**
   - Define a weighted sum: `score = 0.5 * Sales + 0.3 * Quantity - 0.2 * Discount` for 5 products.
   - Compute this manually for each product.
   - Verify your results using NumPy's `np.dot()` or element-wise operations.
   - What does this "score" represent conceptually? Could you use it to rank products?

3. **Descriptive statistics — with meaning:**
   - Choose one numerical feature from the dataset (e.g., Sales, Profit, or Discount).
   - Calculate the mean, variance, and standard deviation manually for a sample of 10 values.
   - Verify with `np.mean()`, `np.var()`, `np.std()`.
   - Interpret each:
     - What does the mean tell you?
     - What does variance tell you that mean does not?
     - If standard deviation is large relative to the mean, what does that imply about the data?

4. **Correlation analysis:**
   - Compute the correlation between Sales and Profit. Then between Sales and Discount.
   - Use `df[['Sales', 'Profit', 'Discount']].corr()` to generate a correlation matrix.
   - Interpret: which pair is most strongly related? Is any relationship negative? Does the result surprise you?
   - Think about it: if two features have a correlation of 0.95, should you include both in a model? Why or why not?

### Tips for Students

- For vectors: think of each product as a point in 3D space. Distance tells you how far apart two points are. In ML, "similar" items have small distances between their feature vectors.
- For the linear combination: this is literally what a linear model does — it assigns weights to features and adds them up. You just did a forward pass.
- For statistics: variance and standard deviation are about spread. Two datasets can have the same mean but wildly different variance. That difference matters for ML — high-variance features dominate distance calculations.
- For correlation: a high positive value means the features move together. Negative means they move in opposite directions. Near zero means no linear relationship.

### Reflection Questions

1. When computing distances, you used raw values (Price, Quantity, Discount). These features have different scales. How would this affect the distance calculation, and what could you do about it?
2. In the weighted sum, the weights were arbitrary (0.5, 0.3, -0.2). In a real ML model, how are these weights determined? (We will answer this in Module 3.)
3. The mean and median of Sales are likely different. What does this tell you about the distribution? Which one would you report to a business stakeholder, and why?

### Submission Requirements

- Jupyter notebook with all calculations (manual work can be in markdown cells, code verification in code cells).
- Written interpretation for each task — not just numbers, but what they mean.

### What a Good Submission Looks Like

- Manual calculations match code verification. If they do not match, the student identifies and explains the discrepancy.
- Interpretations show understanding, not just computation. "The standard deviation of Sales is 230, which means individual orders vary widely from the average" is good. "std = 230" is not.
- Correlation analysis includes genuine interpretation, not just the numbers.

**Grading Hints:** This activity is about understanding, not code complexity. A student who computes everything correctly but cannot explain what the numbers mean should score lower than a student who makes a minor calculation error but demonstrates clear conceptual understanding.

---

## Activity 4: Real-World Integration — Customer Segmentation Simulation

**Objective:** Apply programming, math, and visualization to simulate a real ITES task. This is the closest thing to what a data analyst or ML engineer actually does on the job.

### Scenario

An e-commerce company operating in Uzbekistan (think Uzum, Sello, or a regional marketplace) wants to segment its customers by purchasing behavior. The marketing team is tired of sending the same promotion to everyone — they want targeted campaigns. Your job: use the Superstore dataset to build a customer segmentation model and translate the results into business recommendations.

### Tasks

1. **Feature selection:**
   - Create a customer-level summary with features like: total spending, order count, average order value, average discount used, total profit generated.
   - Select 3-4 features that you think best capture "purchasing behavior."
   - Justify your choices in writing — why these features and not others?

2. **Normalization:**
   - Scale your selected features using `StandardScaler` from sklearn.
   - Explain in 2-3 sentences why scaling is necessary before clustering. What would happen if you skipped it?

3. **K-Means clustering:**
   - Apply K-Means with k=3 (start here).
   - Use the Elbow Method to check if 3 is a reasonable number of clusters. Plot the inertia curve for k=2 through k=6.
   - Assign cluster labels back to your customer DataFrame.

4. **Visualization:**
   - Create a scatter plot of customers colored by cluster (pick 2 features for x and y axes).
   - Create a bar chart showing the average feature values per cluster.
   - Make the charts readable — legend, title, axis labels.

5. **Interpretation:**
   - Describe each segment in plain language. For example: "Cluster 0: High-spending, frequent buyers with low discount usage — likely loyal premium customers."
   - Give each segment a name that a marketing manager would understand.

6. **Business recommendations:**
   - For each segment, suggest one specific marketing action. Be concrete: "Send a 10% loyalty discount via SMS" is better than "Target with promotions."
   - Consider the Uzbekistan context — what channels are realistic? (SMS, Telegram, in-app notifications, etc.)

7. **Limitations:**
   - List at least 3 limitations of your segmentation. Think about: data quality, feature selection, number of clusters, what the model cannot capture.

### Starter Template

```python
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans

# --- Step 1: Load cleaned data ---
df = pd.read_csv('superstore_cleaned.csv')

# --- Step 2: Create customer-level features ---
# TODO: Group by Customer ID and compute summary features
# customer_df = df.groupby('Customer ID').agg(...)

# --- Step 3: Select features for clustering ---
# TODO: Pick 3-4 features and justify your choices
# features = customer_df[['total_spending', 'order_count', ...]]

# --- Step 4: Scale features ---
# TODO: Apply StandardScaler
# scaler = StandardScaler()
# features_scaled = scaler.fit_transform(features)

# --- Step 5: Elbow Method ---
# TODO: Run KMeans for k=2 to k=6, plot inertia
# inertias = []
# for k in range(2, 7):
#     km = KMeans(n_clusters=k, random_state=42, n_init=10)
#     km.fit(features_scaled)
#     inertias.append(km.inertia_)

# --- Step 6: Fit final model ---
# TODO: Choose k, fit KMeans, assign labels
# km_final = KMeans(n_clusters=3, random_state=42, n_init=10)
# customer_df['segment'] = km_final.fit_predict(features_scaled)

# --- Step 7: Visualize ---
# TODO: Scatter plot colored by segment
# TODO: Bar chart of average feature values per segment

# --- Step 8: Interpret and recommend ---
# TODO: Describe each segment in markdown cells below
# TODO: Suggest marketing actions per segment
```

### Tips for Students

- K-Means is sensitive to initialization. Always set `random_state=42` and `n_init=10` so your results are reproducible.
- The Elbow Method does not always give you a clear "elbow." If it is ambiguous, pick the simplest model (fewer clusters) and justify it.
- When interpreting clusters, look at the cluster centers (`km.cluster_centers_`) — they tell you the average feature values for each group.
- Your business recommendations should be actionable. "Do more marketing" is not actionable. "Send a Telegram message to Cluster 2 customers offering free delivery on orders above 500,000 sum" is.
- In Uzbekistan, Telegram is the dominant messaging platform. If you are suggesting customer outreach, Telegram and SMS are more realistic than email.

### Reflection Questions

1. If you ran K-Means 10 times without setting `random_state`, would you get the same clusters every time? Why or why not?
2. You used K-Means, which assumes spherical clusters. What if your customer segments are not spherical? What alternative algorithm could you try?
3. A marketing manager looks at your 3 segments and says "I want 7 segments instead." How would you respond? What are the tradeoffs?
4. What customer behaviors does this segmentation miss? (Hint: think about what is not in the data — recency, product preferences, returns, complaints.)

### Submission Requirements

- Jupyter notebook with all code executed and outputs visible.
- Segment interpretation: a named description of each cluster (3-5 sentences each).
- Business recommendation summary: one concrete action per segment.
- Limitations section: at least 3 points.

### What a Good Submission Looks Like

- The code runs end-to-end without errors.
- Feature selection is justified, not arbitrary.
- Cluster interpretations use the actual numbers from cluster centers, not guesses.
- Business recommendations are specific to the Uzbekistan market, not generic textbook answers.
- Limitations section shows critical thinking — the student knows what the model cannot do.

**Grading Hints:** This is the heaviest activity. Weight it accordingly. A student who completes the code but skips interpretation and recommendations has done half the work. The business thinking is as important as the technical execution. Deduct for generic recommendations that could apply to any country or any dataset.

---

## Grading Summary

| Activity | Weight | Key Criteria |
|----------|--------|-------------|
| Activity 1: Data Preparation | 20% | Clean code, correct cleaning steps, written justifications |
| Activity 2: Visualization | 20% | Chart quality, interpretation depth, all 5 chart types present |
| Activity 3: Math Foundations | 25% | Manual + code verification, conceptual understanding, correlation interpretation |
| Activity 4: Segmentation | 35% | End-to-end execution, feature justification, cluster interpretation, business recommendations |

General deductions:
- Code does not run: -20% on that activity.
- No written explanations: cap at 75% on that activity.
- Missing reflection questions: -5% per missing question.

---

## Homework

**Due:** End of Module 2 (before Module 3 starts).

### Kaggle Micro-Courses (Certificates Required)

Complete both and submit your certificates:

1. **[Kaggle Python Course](https://www.kaggle.com/learn/python)** — covers fundamentals you need for everything ahead.
2. **[Kaggle Pandas Course](https://www.kaggle.com/learn/pandas)** — the single most useful library for tabular data work.

These are free and take 4-6 hours each. Do them early in the module, not the night before the deadline.

### Math Problem Set

#### Part A: Vectors and Distances (5 problems)

1. Given vectors **a** = [3, 4] and **b** = [1, 2], compute **a** + **b** and **a** - **b**. Plot all four vectors on a 2D graph.
2. Compute the dot product of **a** = [2, 5, 1] and **b** = [3, -1, 4]. What does the sign of the dot product tell you about the angle between the vectors?
3. Calculate the Euclidean norm (magnitude) of the vector [6, 8]. Then normalize it to a unit vector.
4. Two customers are represented as feature vectors: Customer A = [50000, 12, 0.05] (spending, orders, discount rate) and Customer B = [48000, 15, 0.08]. Compute the Euclidean distance. Are these customers "similar"?
5. What happens to distances when features have very different scales (e.g., spending in thousands vs discount rate between 0 and 1)? How do you fix this?

#### Part B: Descriptive Statistics (5 problems)

1. A dataset of 1000 customer transactions has a mean purchase value of 150,000 soum and a standard deviation of 40,000 soum. Approximately what percentage of transactions fall between 110,000 and 190,000 soum? (Use the 68-95-99.7 rule.)
2. You have test scores: [72, 85, 90, 68, 95, 88, 76, 82, 91, 79]. Calculate the mean, median, variance, and standard deviation. Which measure of central tendency is more appropriate here — mean or median? Why?
3. A dataset shows that the mean salary in Tashkent is 8,000,000 soum/month but the median is 4,500,000 soum/month. What does this gap tell you about the distribution? Draw a rough sketch of what the histogram might look like.
4. Two features have a correlation of -0.7. What does this mean in plain language? Give an example of two real-world variables that might have this relationship.
5. Feature A has a standard deviation of 50,000 and Feature B has a standard deviation of 0.03. You are about to use a distance-based algorithm (like KNN). What problem do you anticipate, and how would you solve it?

### EDA Notebook

Using the Superstore dataset, create an exploratory data analysis notebook that includes:
- At least 5 different visualizations (mix of chart types).
- A summary of 3 key findings from your analysis.
- One question that your EDA raises but cannot answer — something that would require additional data or a model.

This is separate from Activity 2. Go deeper here — explore relationships that Activities 1 and 2 did not cover.

**Format:** Jupyter notebook, all cells executed, submitted as .ipynb file.
