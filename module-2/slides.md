# Module 2: Programming and Mathematical Foundations — Slide Blueprint

> 6 classes (5 lectures + 1 lab) | ~15-20 slides per class
> Each slide: Title, bullets, speaker notes, visual suggestion
> Activity integration points marked with [ACTIVITY]
> Demo points marked with [DEMO]

---

## CLASS 1: Python Crash Course + Environment Setup (1.5h)

### Slide 1 — Title Slide
**Title:** Python for Data Science — Getting Started
- Module 2, Class 1
- Setting up your environment
- Python fundamentals: variables, types, control flow, functions
- Today's goal: write your first Python code in Google Colab

**Speaker notes:** Welcome to Module 2. This is where we get our hands dirty. Everything we talked about in Module 1 — ML, AI, data science — it all runs on code. Python is the language of data science, and today you'll write your first lines. No prior experience needed.

**Visual:** Title card with Python logo, Google Colab icon, module/class number.

---

### Slide 2 — Why Python?
**Title:** Why Python Dominates Data Science
- Most popular language for ML/AI (used by Google, Meta, OpenAI)
- Readable syntax — closest to writing in English
- Massive ecosystem: NumPy, Pandas, Scikit-learn, TensorFlow, PyTorch
- Free, open-source, huge community

**Speaker notes:** You might wonder why Python and not Java or C++. Simple — Python reads almost like English, and every major ML library is built for it. If you Google any data science question, the answer will be in Python. That's the ecosystem effect.

**Visual:** Bar chart — programming language popularity in data science (Python, R, SQL, Julia). Source: Stack Overflow Developer Survey or TIOBE index.

---

### Slide 3 — Google Colab: Your Lab Environment
**Title:** Google Colab — Free Cloud Notebooks
- No installation needed — runs in your browser
- Free GPU access for ML experiments later
- Built-in Python libraries (NumPy, Pandas, Matplotlib pre-installed)
- Saves to Google Drive automatically

**Speaker notes:** We're using Google Colab so nobody gets stuck on installation issues. Open your browser, go to colab.research.google.com, sign in with your Google account, and you have a full Python environment. It's the same tool many professionals use for prototyping.

**Visual:** Screenshot of Google Colab interface — empty notebook with a code cell. Annotate: "Code cell," "Text cell," "Run button," "File menu."

---

### Slide 4 — [DEMO] Setting Up Colab
**Title:** Let's Open Our First Notebook
- Go to colab.research.google.com
- Click "New Notebook"
- Rename it: "Module2_Class1_YourName"
- Type `print("Salom, dunyo!")` and hit Shift+Enter

[DEMO]: Open Colab live. Create notebook. Run first print statement. Show how cells work — code cells vs text cells. Show keyboard shortcuts (Shift+Enter to run, Ctrl+M B for new cell).

**Speaker notes:** Follow along with me. Open Colab now. If anyone doesn't have a Google account, pair up with someone who does — we'll sort it out after class. Let's type our first line. When you see "Salom, dunyo!" printed below the cell, you've just run Python.

**Visual:** Live screen share — Colab interface.

---

### Slide 5 — Python Data Types
**Title:** The Building Blocks: Data Types
- **int** — whole numbers: `age = 22`
- **float** — decimals: `gpa = 3.85`
- **str** — text: `name = "Nodira"`
- **bool** — True/False: `is_student = True`
- **list** — ordered collection: `cities = ["Tashkent", "Samarkand", "Bukhara"]`

**Speaker notes:** Every piece of data in Python has a type. Numbers can be integers or floats. Text is called a string. Booleans are just yes/no values. Lists hold multiple items in order. Think of these as containers — the type tells Python what kind of container it is and what operations are allowed.

**Visual:** Data types pyramid — simple types (int, float, str, bool) at the base, collections (list, dict, tuple) above, with examples beside each level.

---

### Slide 6 — Variables and Assignment
**Title:** Variables — Naming Your Data
- A variable is a label that points to a value
- `population = 35_000_000` — Uzbekistan's population
- `capital = "Tashkent"`
- Naming rules: start with letter/underscore, no spaces, case-sensitive
- Use descriptive names: `student_count` not `x`

**Speaker notes:** A variable is just a name you give to a piece of data so you can reuse it. Python doesn't need you to declare the type — it figures it out. Good variable names make your code readable. If I see `x = 35000000`, I have no idea what that is. If I see `population = 35_000_000`, it's obvious.

**Visual:** Diagram — variable name on the left with an arrow pointing to a box containing the value in memory. Show 2-3 examples stacked.

---

### Slide 7 — [DEMO] Playing with Variables
**Title:** Hands-On: Variables and Types
- Create variables for: your name, age, city, GPA
- Use `type()` to check what Python thinks
- Try arithmetic: `birth_year = 2026 - age`
- String concatenation: `greeting = "Salom, " + name`

[DEMO]: Create variables in Colab. Show type() function. Demonstrate what happens when you add a string and int (TypeError). Show f-strings: `f"My name is {name}, I am {age}"`.

**Speaker notes:** Let's try this together. Create a few variables about yourself. Now use `type()` to check — Python will tell you it's an int, str, or whatever. Try adding your name and age together — you'll get an error. That's Python telling you it can't add text and numbers. This is your first lesson in debugging.

**Visual:** Live screen share.

---

### Slide 8 — Control Flow: if/elif/else
**Title:** Making Decisions in Code
- `if` checks a condition and runs code if True
- `elif` (else if) — additional conditions
- `else` — everything that didn't match
- Indentation matters — Python uses it to define code blocks

```
temperature = 42
if temperature > 40:
    print("Juda issiq!")  # Very hot
elif temperature > 25:
    print("Yoqimli ob-havo")  # Nice weather
else:
    print("Sovuq")  # Cold
```

**Speaker notes:** Control flow is how your program makes decisions. If this, do that. Otherwise, do something else. The key thing in Python is indentation — the spaces before each line tell Python which code belongs to which block. This trips up beginners, so pay attention to it.

**Visual:** Flowchart — diamond shape for condition, two branches (True/False), each leading to an action box. Map it to the code example.

---

### Slide 9 — [DEMO] Control Flow in Action
**Title:** Let's Write Decision Logic
- Grade checker: input a score, output a letter grade
- 90+ = A, 80+ = B, 70+ = C, below 70 = F
- Extend: add a message for each grade

[DEMO]: Build the grade checker live. Start simple (just if/else), then add elif branches. Show what happens with wrong indentation. Let students suggest edge cases.

**Speaker notes:** We'll build a grade checker step by step. I'll start with just pass/fail, then we'll add more grades. Watch what happens if I mess up the indentation — Python will throw an error immediately. That's actually helpful — it forces clean code.

**Visual:** Live screen share.

---

### Slide 10 — Loops: for and while
**Title:** Repeating Actions — Loops
- `for` loop — iterate over a sequence (list, range, string)
- `while` loop — repeat as long as a condition is True
- `range(n)` generates numbers 0 to n-1

```
cities = ["Tashkent", "Samarkand", "Bukhara", "Nukus"]
for city in cities:
    print(f"{city} is in Uzbekistan")
```

**Speaker notes:** Loops let you repeat an action without writing the same line 100 times. The for loop is the one you'll use 90% of the time in data science. It goes through each item in a list or range and does something with it. While loops are less common but useful when you don't know in advance how many times to repeat.

**Visual:** Loop animation concept — a circular arrow showing iteration, with each pass highlighting the next item in the list.

---

### Slide 11 — [DEMO] Loops Practice
**Title:** Loop Exercises
- Loop through a list of Uzbek cities and print each
- Use `range(1, 11)` to print numbers 1 to 10
- Calculate the sum of numbers 1 to 100 using a loop
- Bonus: loop through a string character by character

[DEMO]: Build each example live. Show `enumerate()` for getting both index and value. Demonstrate a while loop that counts down from 10.

**Speaker notes:** Let's practice. Type along with me. First, loop through our cities list. Then let's use range to generate numbers. Here's a classic — can you sum all numbers from 1 to 100? Gauss did this in his head, but we'll let Python do it.

**Visual:** Live screen share.

---

### Slide 12 — Functions
**Title:** Functions — Reusable Code Blocks
- A function takes inputs, does something, and returns output
- `def` keyword to define, `return` to send back a result
- Parameters = inputs, return value = output

```
def celsius_to_fahrenheit(celsius):
    return celsius * 9/5 + 32

tashkent_summer = celsius_to_fahrenheit(42)
print(tashkent_summer)  # 107.6
```

**Speaker notes:** Functions are how you organize code into reusable pieces. Instead of writing the same formula every time, you wrap it in a function and call it by name. Every library you'll use — NumPy, Pandas, Sklearn — is just a collection of functions someone else wrote.

**Visual:** Function anatomy diagram — labeled parts: `def` keyword, function name, parameters in parentheses, colon, indented body, `return` statement. Arrows pointing to each part with labels.

---

### Slide 13 — [DEMO] Writing Functions
**Title:** Build Your Own Functions
- BMI calculator: `def bmi(weight_kg, height_m)`
- Grade converter from Slide 9, now as a function
- Function that takes a list of numbers and returns the average

[DEMO]: Write BMI calculator function. Call it with different inputs. Show default parameters. Show what happens when you forget `return` (you get None).

**Speaker notes:** Let's turn our earlier grade checker into a function. Instead of hardcoding the score, we pass it as a parameter. Now we can check any score by calling the function. This is the core idea — write once, use many times.

**Visual:** Live screen share.

---

### Slide 14 — Lists and Dictionaries
**Title:** Collections: Lists and Dictionaries
- **List**: Ordered, indexed, mutable — `scores = [85, 92, 78, 95]`
- **Dictionary**: Key-value pairs — `student = {"name": "Aziz", "age": 21, "city": "Fergana"}`
- List: access by position `scores[0]` → 85
- Dict: access by key `student["name"]` → "Aziz"
- Both are everywhere in data science

**Speaker notes:** Lists and dictionaries are the two collections you'll use constantly. Lists are ordered sequences — think rows in a spreadsheet. Dictionaries are key-value pairs — think a real dictionary where you look up a word to find its definition. When we get to Pandas DataFrames tomorrow, you'll see they're basically fancy dictionaries of lists.

**Visual:** Side-by-side comparison — List as a numbered column, Dictionary as a two-column table (key | value).

---

### Slide 15 — Common Beginner Mistakes
**Title:** Mistakes You'll Make (And That's Fine)
- Forgetting the colon after `if`, `for`, `def`
- Wrong indentation — Python is strict about this
- Using `=` (assignment) instead of `==` (comparison)
- Index out of range — lists start at 0, not 1
- Calling a function without parentheses: `print` vs `print()`

**Speaker notes:** Everyone makes these mistakes. I still make some of them. The point isn't to memorize every rule — it's to learn to read error messages. Python's error messages are actually pretty helpful once you get used to them. "IndentationError" means your spaces are wrong. "NameError" means you used a variable that doesn't exist. Read the error, fix the line.

**Visual:** Table — two columns: "What you wrote" vs "What Python expected." Show 4-5 common error examples with the fix.

---

### Slide 16 — Class Recap + Homework
**Title:** What We Covered Today
- Set up Google Colab — your data science workspace
- Python basics: variables, data types, control flow, loops, functions
- Key takeaway: Python is a tool, not a test. Practice > perfection.

**Homework:**
- Complete Kaggle Python Micro-Course (first 3 lessons)
- Write 3 functions: temperature converter, list averager, one of your choice
- [ACTIVITY] Activity 1 Step 1: Open the Superstore dataset in Colab, load it with `pd.read_csv()` (we'll cover Pandas properly next class — just try it)

**Speaker notes:** Good first class. You now have a working Python environment and you've written real code. The homework is on Kaggle — it's interactive, gives you instant feedback, and doesn't require any setup. The third task is a preview of tomorrow — try loading a CSV file. If it breaks, that's fine. We'll cover it.

**Visual:** Checklist graphic with the three homework items. QR code or shortened link to Kaggle Python course.

---

## CLASS 2: NumPy and Pandas (1.5h)

### Slide 1 — Title Slide
**Title:** NumPy and Pandas — Data Tools That Do the Heavy Lifting
- Module 2, Class 2
- NumPy: fast math on arrays
- Pandas: structured data (tables, CSVs, filtering, aggregation)
- Today's goal: load, inspect, filter, and summarize a real dataset

**Speaker notes:** Yesterday you learned base Python. Today we add two libraries that turn Python into a data science powerhouse. NumPy handles numerical computation. Pandas handles tabular data — think Excel but programmable and a thousand times more powerful. By the end of today, you'll load a real dataset and answer questions about it using code.

**Visual:** Title card with NumPy and Pandas logos side by side.

---

### Slide 2 — What is NumPy?
**Title:** NumPy — Numerical Python
- The foundation of almost every data science library
- Core object: **ndarray** (n-dimensional array)
- Why not just use Python lists? Speed. NumPy arrays are 10-100x faster.
- Operations apply to entire arrays at once (vectorization)

**Speaker notes:** NumPy stands for Numerical Python. Its main contribution is the array — a data structure optimized for math. Python lists are flexible but slow for number crunching. NumPy arrays store data in contiguous memory and let you operate on millions of numbers in one line. Every ML library — Pandas, Sklearn, TensorFlow — is built on top of NumPy.

**Visual:** Speed comparison — bar chart showing time to sum 1 million numbers: Python list loop vs NumPy sum. Show the 50-100x difference.

---

### Slide 3 — NumPy Arrays: 1D and 2D
**Title:** Arrays — From Lists to Matrices
- **1D array**: `np.array([10, 20, 30, 40])` — like a row of data
- **2D array**: `np.array([[1, 2, 3], [4, 5, 6]])` — like a table
- `.shape` tells you dimensions: (4,) vs (2, 3)
- Indexing: `arr[0]` for 1D, `arr[0, 1]` for 2D (row, column)

**Speaker notes:** A 1D array is just a list of numbers. A 2D array is a table — rows and columns. Think of a spreadsheet: each row is a customer, each column is a feature like age, income, data usage. The shape attribute tells you how many rows and columns you have. You'll use shape constantly to sanity-check your data.

**Visual:** 1D array shown as a single horizontal row of boxes with values. 2D array as a grid (2 rows x 3 columns) with row/column indices labeled. Arrows showing how indexing works.

---

### Slide 4 — NumPy Operations
**Title:** Vectorized Operations — Math Without Loops
- Element-wise: `arr * 2` doubles every element
- Aggregation: `arr.mean()`, `arr.sum()`, `arr.std()`
- Boolean indexing: `arr[arr > 25]` — filter in one line
- Broadcasting: NumPy figures out shape mismatches automatically

**Speaker notes:** This is where NumPy shines. Instead of writing a loop to multiply each number by 2, you just write `arr * 2` and NumPy does it all at once. Same for finding the mean, sum, or standard deviation. Boolean indexing is especially powerful — give it a condition and it returns only the elements that match. No loop, no if statement.

**Visual:** Before/after diagram — left side shows a Python loop (4 lines of code), right side shows the NumPy one-liner doing the same thing. Highlight the simplicity.

---

### Slide 5 — What is Pandas?
**Title:** Pandas — Data Analysis Made Simple
- Built on top of NumPy
- Core objects: **Series** (one column) and **DataFrame** (full table)
- Think of a DataFrame as a programmable Excel spreadsheet
- Reads CSV, Excel, JSON, SQL — almost any data format
- Named columns and row indices — not just numbers

**Speaker notes:** If NumPy is the engine, Pandas is the car. You'll rarely use NumPy directly in data analysis — Pandas wraps it in a much friendlier interface. The DataFrame is the object you'll work with 90% of the time. It's a table with named columns and rows, and it comes with hundreds of built-in methods for filtering, grouping, merging, and summarizing data.

**Visual:** Pandas DataFrame anatomy — a table with column headers (Name, Age, City, Salary), row index on the left (0, 1, 2...), one column highlighted as a Series. Labels pointing to: "Column names," "Row index," "Values."

---

### Slide 6 — [DEMO] Loading Data with Pandas
**Title:** Load, Inspect, Understand
- `pd.read_csv("file.csv")` — one line to load data
- `df.head()` — first 5 rows
- `df.shape` — how many rows and columns
- `df.info()` — column types, missing values
- `df.describe()` — statistical summary

[DEMO]: Load the Superstore Sales dataset in Colab. Run each inspection command. Walk through what each output tells you. Point out data types, null counts, and summary stats.

**Speaker notes:** Let's load real data. I'll use the Superstore dataset — it has sales records with categories, regions, and amounts. First thing after loading: always inspect. Head shows you the first rows so you know what it looks like. Shape tells you size. Info shows types and missing values. Describe gives you stats. This is your standard first-five-minutes workflow with any new dataset.

**Visual:** Live screen share — Colab with dataset loaded.

---

### Slide 7 — Selecting and Filtering Data
**Title:** Getting the Data You Need
- Select a column: `df["Sales"]` or `df.Sales`
- Select multiple columns: `df[["City", "Sales", "Profit"]]`
- Filter rows: `df[df["Region"] == "Central"]`
- Combine conditions: `df[(df["Sales"] > 1000) & (df["Category"] == "Technology")]`

**Speaker notes:** Once your data is loaded, you need to slice it. Selecting a single column gives you a Series. Selecting multiple columns gives you a smaller DataFrame. Filtering is the most common operation — you write a condition inside brackets and Pandas returns only the rows that match. Think of it as SQL's WHERE clause but in Python.

**Visual:** Filtering pipeline diagram — full DataFrame on the left, condition in the middle (funnel icon), filtered subset on the right. Show row count shrinking.

---

### Slide 8 — [DEMO] Filtering in Practice
**Title:** Hands-On: Answering Questions with Filters
- "Show me all orders from Tashkent" (if using Uzbekistan data)
- "Which products had profit > $500?"
- "How many Technology orders were shipped Standard Class?"
- Chain filters to narrow down

[DEMO]: Apply filters to the Superstore dataset. Show the thought process: question → condition → code → result. Demonstrate .value_counts() on categorical columns.

**Speaker notes:** Here's how data analysis actually works. Someone asks a question, you translate it into a filter, and the data gives you the answer. Let's try a few. How many orders had negative profit? Which category is most profitable? You can answer these in one line each.

**Visual:** Live screen share.

---

### Slide 9 — Aggregation and Groupby
**Title:** Summarizing Data — groupby()
- `df.groupby("Category")["Sales"].mean()` — average sales per category
- Common aggregations: `.sum()`, `.mean()`, `.count()`, `.max()`, `.min()`
- Multiple aggregations: `.agg(["mean", "sum", "count"])`
- This is the Pandas equivalent of SQL's GROUP BY

**Speaker notes:** Groupby is where Pandas becomes really powerful. Instead of filtering one category at a time, you group by a column and apply a function to each group. Average sales by region? One line. Total profit by product category? One line. If you've used pivot tables in Excel, this is the same concept but more flexible.

**Visual:** Groupby diagram — original table on left, "split" into colored groups by category in the middle, "apply" (mean) to each group, "combine" into summary table on right. The split-apply-combine pattern.

---

### Slide 10 — [DEMO] Aggregation Exercises
**Title:** Group, Aggregate, Analyze
- Total sales by region
- Average profit by product category
- Count of orders per shipping mode
- Top 5 cities by total sales

[DEMO]: Build each aggregation live. Show sort_values() to rank results. Demonstrate .nlargest() and .nsmallest() shortcuts.

**Speaker notes:** Let's work through these. Total sales by region — one line. Now sort it to see which region is biggest. Average profit by category — and suddenly you see that Furniture barely makes money while Technology is a gold mine. This is the power of aggregation — patterns emerge from raw data.

**Visual:** Live screen share.

---

### Slide 11 — Handling Missing Data
**Title:** Missing Values — The Reality of Real Data
- Real datasets always have gaps
- `df.isnull().sum()` — count missing values per column
- Options: drop rows (`dropna()`), fill with a value (`fillna(0)`), fill with mean
- Decision depends on context — dropping isn't always the right move
- We'll go deeper on this in Module 3

**Speaker notes:** Perfect data doesn't exist. Every real dataset has missing values — someone didn't fill out a form, a sensor failed, a column wasn't required. You need to detect and handle these gaps. For now, know the basics — isnull to find them, dropna to remove rows with gaps, fillna to substitute a value. The right strategy depends on the situation, which we'll cover in the data preparation module.

**Visual:** DataFrame with highlighted NaN cells in red. Three arrows pointing to three strategies: Drop, Fill with 0, Fill with mean.

---

### Slide 12 — Creating New Columns
**Title:** Feature Engineering Preview: New Columns from Old
- `df["Profit_Margin"] = df["Profit"] / df["Sales"] * 100`
- String operations: `df["City_Upper"] = df["City"].str.upper()`
- Conditional columns with `np.where()`:
  `df["Profitable"] = np.where(df["Profit"] > 0, "Yes", "No")`

**Speaker notes:** You're not limited to the columns that came in the CSV. You can create new ones from existing data. Profit margin from profit and sales. A flag for whether an order was profitable. This is the beginning of feature engineering — creating useful inputs for ML models. We'll do a lot more of this in Module 3.

**Visual:** Table showing original columns (Sales, Profit) and a new derived column (Profit_Margin) with the formula as a header annotation.

---

### Slide 13 — Pandas Cheat Sheet
**Title:** Pandas Quick Reference
- Load: `pd.read_csv()`, `pd.read_excel()`
- Inspect: `.head()`, `.shape`, `.info()`, `.describe()`, `.dtypes`
- Select: `df["col"]`, `df[["col1", "col2"]]`
- Filter: `df[df["col"] > value]`
- Aggregate: `.groupby()`, `.value_counts()`, `.agg()`
- Clean: `.isnull()`, `.dropna()`, `.fillna()`, `.drop_duplicates()`

**Speaker notes:** Here's a cheat sheet you'll keep coming back to. You don't need to memorize all of this — the goal today is to know these operations exist so you can look them up when you need them. Bookmark this slide or take a screenshot.

**Visual:** Clean reference card layout — two columns, function on left, description on right. Color-coded by category (load = blue, inspect = green, filter = orange, etc.).

---

### Slide 14 — Class Recap + Homework
**Title:** What We Covered Today
- NumPy: arrays, vectorized operations, speed
- Pandas: DataFrames, loading data, filtering, groupby
- Key takeaway: Pandas is your Swiss Army knife for data. Learn it well.

**Homework:**
- Complete Kaggle Pandas Micro-Course (first 3 lessons)
- Practice: Load Superstore dataset, answer 5 questions using filters and groupby
- [ACTIVITY] Activity 1 Step 2: Basic data inspection — shape, dtypes, missing values, first statistical summary

**Speaker notes:** NumPy and Pandas are the two libraries you'll use in every single project going forward. The Kaggle course gives you interactive exercises with instant feedback — do at least the first three lessons. For the activity, run the inspection commands on the Superstore dataset and write down what you notice. Any surprises in the data? Missing values? Weird distributions? That's the start of EDA.

**Visual:** Checklist with homework items. Links to Kaggle Pandas course.

---

## CLASS 3: Data Visualization — Matplotlib and Seaborn (1.5h)

### Slide 1 — Title Slide
**Title:** Data Visualization — Making Data Speak
- Module 2, Class 3
- Matplotlib: the foundation of Python plotting
- Seaborn: statistical visualizations made easy
- Today's goal: build 5 plot types and know when to use each

**Speaker notes:** Numbers in a table are hard to interpret. A chart makes the pattern obvious in seconds. Today we learn two plotting libraries — Matplotlib is the low-level workhorse, Seaborn builds beautiful statistical charts on top of it. By the end of class, you'll be able to create histograms, scatter plots, bar charts, box plots, and heatmaps.

**Visual:** Title card with a collage of chart types: histogram, scatter plot, bar chart, box plot, heatmap.

---

### Slide 2 — Why Visualize Data?
**Title:** The Same Data, Different Story
- Anscombe's Quartet: 4 datasets with identical stats but totally different patterns
- Numbers alone can mislead — always plot your data
- Visualization is not decoration — it's analysis
- "If you can't see it, you can't model it"

**Speaker notes:** This is Anscombe's Quartet — four datasets that have the exact same mean, variance, and correlation. But look at the plots. They're completely different. One is a line, one is a curve, one has an outlier pulling the stats. This is why we plot. The numbers tell you what the data says on average. The plot tells you what the data actually looks like.

**Visual:** Anscombe's Quartet — 4 scatter plots in a 2x2 grid, each with the same stats displayed but visually different patterns.

---

### Slide 3 — Chart Type Selection Guide
**Title:** Which Chart Should I Use?
- **Distribution of one variable** → Histogram, KDE plot
- **Compare categories** → Bar chart, count plot
- **Relationship between two variables** → Scatter plot
- **Distribution across groups** → Box plot, violin plot
- **Correlation between many variables** → Heatmap
- When in doubt: start with a histogram and a scatter plot

**Speaker notes:** This is the chart you should bookmark. The question "which chart do I use?" comes up every time. The answer depends on what you're trying to show. One variable's distribution? Histogram. Two variables' relationship? Scatter. Comparing groups? Bar chart. Over time, this becomes intuition, but for now, use this as a decision tree.

**Visual:** Flowchart / decision tree — starts with "What are you trying to show?" and branches into: Distribution → Histogram, Comparison → Bar, Relationship → Scatter, Spread → Boxplot, Correlation → Heatmap.

---

### Slide 4 — Matplotlib Basics
**Title:** Matplotlib — The Foundation
- `import matplotlib.pyplot as plt`
- `plt.plot(x, y)` — line plot
- `plt.xlabel()`, `plt.ylabel()`, `plt.title()` — labels
- `plt.show()` — render the plot
- Figure and Axes: `fig, ax = plt.subplots()` for more control

**Speaker notes:** Matplotlib is the original Python plotting library. It's not the prettiest by default, but it's incredibly flexible. You can customize every pixel if you want. The basic pattern is simple — call a plot function, add labels, call show. For fancier layouts, use the subplots approach — it gives you a figure and axes objects you can control independently.

**Visual:** Simple line plot with labeled parts — title, x-label, y-label, legend, gridlines. Annotations pointing to each element.

---

### Slide 5 — [DEMO] Histogram
**Title:** Histogram — Distribution of a Single Variable
- Shows how values are distributed (frequency of ranges)
- `plt.hist(df["Sales"], bins=30)`
- Useful for: spotting skewness, outliers, data range
- Tip: adjust `bins` parameter — too few hides patterns, too many adds noise

[DEMO]: Plot histogram of Sales from Superstore dataset. Show effect of different bin counts (10, 30, 50). Add title and labels. Show the skewed distribution — most sales are small, a few are very large.

**Speaker notes:** Let's build our first real plot. A histogram of sales values. Notice how most orders are small — under $500 — but there's a long tail of big orders. That's called a right skew. If we change the number of bins, watch how the shape changes. Too few bins and it looks flat. Too many and it's noisy. Around 30 usually works.

**Visual:** Live screen share — histogram with labeled axes.

---

### Slide 6 — [DEMO] Bar Chart
**Title:** Bar Chart — Comparing Categories
- `df.groupby("Category")["Sales"].sum().plot(kind="bar")`
- Seaborn: `sns.countplot(x="Category", data=df)`
- Horizontal bars for long labels: `kind="barh"`
- Always sort bars by value for readability

[DEMO]: Bar chart of total sales by category. Then by region. Show horizontal bars. Add value labels on top of bars.

**Speaker notes:** Bar charts compare quantities across categories. Which product category brings the most revenue? Which region sells the most? Always sort your bars — it's easier to read. Unsorted bar charts are one of the most common visualization mistakes.

**Visual:** Live screen share — sorted bar chart.

---

### Slide 7 — [DEMO] Scatter Plot
**Title:** Scatter Plot — Relationships Between Variables
- `plt.scatter(df["Sales"], df["Profit"])`
- Each dot is one data point
- Reveals: correlations, clusters, outliers
- Add color for a third variable: `c=df["Discount"]`
- Seaborn: `sns.scatterplot(x="Sales", y="Profit", hue="Category", data=df)`

[DEMO]: Scatter plot of Sales vs Profit. Color by Category. Point out the negative profit cluster (high discount items losing money). Show how hue adds a dimension.

**Speaker notes:** Scatter plots show the relationship between two numeric variables. Here, Sales vs Profit. If every sale were equally profitable, you'd see a straight diagonal line. But look — there are dots below zero. Some sales actually lose money. Color them by category and you might see which categories have this problem. This is exploratory analysis in action.

**Visual:** Live screen share — scatter plot with colored categories and a horizontal line at Profit = 0.

---

### Slide 8 — [DEMO] Box Plot
**Title:** Box Plot — Distribution and Outliers at a Glance
- Shows: median, quartiles (25th/75th), whiskers, outliers
- `sns.boxplot(x="Category", y="Sales", data=df)`
- Great for comparing distributions across groups
- Outliers shown as individual dots beyond the whiskers

[DEMO]: Box plot of Sales by Category. Explain each part of the box plot: median line, box edges (IQR), whiskers, outlier dots. Compare the three categories — Technology has the widest spread.

**Speaker notes:** Box plots pack a lot of information into a small space. The line in the middle is the median — half the data is above, half below. The box covers the middle 50%. The whiskers extend to the rest, and dots beyond them are outliers. When you put multiple box plots side by side, you can instantly compare distributions across groups.

**Visual:** Live screen share + annotated box plot diagram showing median, Q1, Q3, IQR, whiskers, and outlier dots with labels.

---

### Slide 9 — [DEMO] Heatmap
**Title:** Heatmap — Correlation Matrix
- `sns.heatmap(df.corr(), annot=True, cmap="coolwarm")`
- Shows correlation between all numeric variables
- +1 = strong positive, -1 = strong negative, 0 = no relationship
- Useful for quick feature selection in ML

[DEMO]: Compute correlation matrix on Superstore numeric columns. Plot as heatmap. Point out which variables correlate (Sales-Profit positive, Discount-Profit negative). Explain what this means for modeling.

**Speaker notes:** A heatmap of the correlation matrix shows you at a glance which variables move together. Red means positive correlation — when one goes up, the other does too. Blue means negative. Look at Discount and Profit — negative correlation. Higher discounts, lower profit. That's a real business insight from one line of code.

**Visual:** Live screen share — annotated heatmap with color bar.

---

### Slide 10 — Seaborn: Built for Statistics
**Title:** Why Seaborn on Top of Matplotlib?
- Better defaults — looks professional out of the box
- Statistical plots built in: violin plots, pair plots, distribution plots
- Integrates directly with Pandas DataFrames
- `sns.set_style("whitegrid")` — one line for clean aesthetics

**Speaker notes:** Matplotlib can do everything, but it often needs 10 lines of code to look good. Seaborn makes common statistical plots in one line and they look great by default. It also understands DataFrames directly — you pass column names as strings instead of extracting arrays. Use Matplotlib for custom plots, Seaborn for standard analysis charts.

**Visual:** Side-by-side comparison — same scatter plot in raw Matplotlib (default gray background, no legend) vs Seaborn (clean grid, colored by category, legend).

---

### Slide 11 — Customizing Plots
**Title:** Making Charts Presentation-Ready
- Always add: title, axis labels, units
- Use `figsize=(12, 6)` for proper proportions
- Color palettes: `sns.color_palette("husl", 5)`
- Save plots: `plt.savefig("chart.png", dpi=300, bbox_inches="tight")`
- Multiple plots: `fig, axes = plt.subplots(1, 3, figsize=(18, 5))`

**Speaker notes:** A chart without labels is useless. Always add a title that tells the viewer what they're looking at, and label your axes with units. When you're making charts for a report or presentation, save them at 300 DPI for crisp images. Subplots let you put multiple charts side by side — great for comparing related views.

**Visual:** Before/after — a bare plot with no labels vs the same plot with title, labels, gridlines, legend, and proper sizing.

---

### Slide 12 — [ACTIVITY] Activity 2: Data Visualization and Trend Analysis
**Title:** Activity 2 — Visualize the Superstore Data
[ACTIVITY]
- Create at least 5 charts from the Superstore dataset:
  1. Histogram of Sales distribution
  2. Bar chart: Total sales by Category
  3. Scatter plot: Sales vs Profit, colored by Category
  4. Box plot: Profit distribution by Region
  5. Heatmap: Correlation matrix
- Each chart must have: title, labeled axes, appropriate colors
- Write 1-2 sentences interpreting each chart

**Speaker notes:** This is your first visualization activity from the professor's curriculum. You'll create five charts, each targeting a different question about the data. The interpretation part is key — a chart without a sentence explaining what it shows is just a picture. What pattern do you see? What does it mean for the business? Start in class, finish as homework.

**Visual:** Activity worksheet layout — 5 numbered slots, each with the chart type and a "What does this tell us?" prompt.

---

### Slide 13 — Uzbekistan Context: What Would You Visualize?
**Title:** Local Data, Real Questions
- Tashkent metro ridership by time of day — what chart?
- Regional internet penetration across Uzbekistan's 14 regions — what chart?
- Uztelecom customer complaints by category — what chart?
- Cotton export volume over 20 years — what chart?
- Correlation between education spending and literacy rates by region?

**Speaker notes:** Let's make this real. I'll give you a question, you tell me what chart type fits. Tashkent metro ridership by hour — line chart, because it's a trend over time. Regional internet access — bar chart or choropleth map. Customer complaints by type — bar chart. Export trends — line chart. Spending vs literacy — scatter plot. See how the question dictates the chart?

**Visual:** Map of Uzbekistan with 14 regions outlined, each region colored by a hypothetical metric (e.g., internet penetration percentage).

---

### Slide 14 — Class Recap + Homework
**Title:** What We Covered Today
- Matplotlib for foundational plotting
- Seaborn for statistical charts
- 5 chart types: histogram, bar, scatter, box plot, heatmap
- Key takeaway: always visualize before modeling. Always label your charts.

**Homework:**
- Complete Kaggle Data Visualization Micro-Course (first 3 lessons)
- [ACTIVITY] Activity 2: finish all 5 charts with interpretations
- Bonus: create one chart not covered today (line plot, violin plot, or pair plot)

**Speaker notes:** You now have 5 chart types in your toolkit. That covers about 80% of what you'll need in data analysis. The Kaggle course will reinforce this with hands-on exercises. For the activity, make your charts clean and interpreted — imagine you're showing them to a manager who doesn't know Python.

**Visual:** Gallery of today's 5 chart types in a row, each labeled.

---

## CLASS 4: Linear Algebra for ML (1.5h)

### Slide 1 — Title Slide
**Title:** Linear Algebra — The Math Behind Machine Learning
- Module 2, Class 4
- Vectors, matrices, dot products, matrix multiplication
- Why this math is unavoidable in ML
- Today's goal: build geometric intuition, not memorize formulas

**Speaker notes:** Math class. Before you panic — we're not doing proofs. We're building intuition. Linear algebra is the language that ML algorithms speak. When you train a model, it's doing matrix multiplication under the hood. When you compute a prediction, it's a dot product. Today you'll understand what those words mean and why they matter. Intuition first, formulas second.

**Visual:** Title card with a stylized vector arrow and a matrix grid.

---

### Slide 2 — What is a Vector?
**Title:** Vectors — Numbers with Direction
- A vector is just an ordered list of numbers: `[age, income, data_usage]`
- In ML: each data point is a vector of features
- Geometrically: an arrow in space with magnitude and direction
- Example: A customer = `[25, 3_500_000, 15]` (age, income in UZS, GB/month)

**Speaker notes:** Forget the physics definition for now. In data science, a vector is a list of numbers that describes something. A customer has an age, an income, and a monthly data usage — that's a 3D vector. Each number is a feature. Every row in your Pandas DataFrame is a vector. When we say "feature vector," this is what we mean.

**Visual:** Vector as arrow diagram — 2D coordinate plane with an arrow from origin to point (3, 4). Next to it, the same concept as a customer feature list: [age=25, income=3.5M UZS, data=15GB].

---

### Slide 3 — Vectors as Customer Profiles
**Title:** Every Customer is a Point in Space
- Customer A: `[22, 2_000_000, 30]` — young, lower income, heavy data user
- Customer B: `[45, 8_000_000, 5]` — older, higher income, light data user
- Plot them: each axis is a feature, each customer is a point
- Similar customers = points close together
- This is the foundation of clustering, recommendation, and similarity

**Speaker notes:** Let me make this concrete. Two Uztelecom customers. One is 22, earns 2 million sum, uses 30GB. The other is 45, earns 8 million, uses 5GB. Plot them in 3D space and they're far apart. Now imagine 10,000 customers plotted like this. You'd see clusters — groups of similar people. That's literally what unsupervised learning does. The math we're learning today makes that possible.

**Visual:** 3D scatter plot (simplified) — two customer points plotted in (age, income, data_usage) space. Arrows from origin to each point. Distance line between them.

---

### Slide 4 — Vector Operations
**Title:** Adding, Scaling, and Measuring Vectors
- **Addition**: `[1, 2] + [3, 4] = [4, 6]` — combine features
- **Scalar multiplication**: `2 * [1, 2] = [2, 4]` — scale up/down
- **Magnitude** (length): `||v|| = sqrt(v1^2 + v2^2)` — "how big" a vector is
- **Unit vector**: divide by magnitude to get direction only

**Speaker notes:** Three basic operations. Adding vectors means combining them — element by element. Scaling means multiplying every element by the same number. Magnitude tells you the length of the vector — how far the point is from the origin. These are simple operations, but they power everything from normalizing data to computing distances in k-nearest neighbors.

**Visual:** 2D coordinate plane showing vector addition (head-to-tail method), scalar multiplication (same direction, longer arrow), and magnitude (dotted line from tip to each axis).

---

### Slide 5 — The Dot Product
**Title:** Dot Product — Measuring Similarity
- `[1, 2, 3] . [4, 5, 6] = 1*4 + 2*5 + 3*6 = 32`
- Multiply corresponding elements, then sum
- Geometric meaning: how aligned two vectors are
- Large positive → similar direction
- Zero → perpendicular (unrelated)
- Negative → opposite directions

**Speaker notes:** The dot product is one of the most important operations in ML. Multiply each pair of elements and sum the results. Geometrically, it tells you how similar two vectors are in direction. If two customer profiles point in the same direction in feature space, their dot product is large. If they're unrelated, it's near zero. Cosine similarity, which you'll use in NLP and recommendation systems, is just a normalized dot product.

**Visual:** Dot product geometric interpretation — two vectors with angle theta between them. Show cases: angle = 0 (parallel, max dot product), angle = 90 (perpendicular, dot product = 0), angle = 180 (opposite, negative dot product).

---

### Slide 6 — What is a Matrix?
**Title:** Matrices — Tables of Numbers
- A matrix is a 2D grid of numbers: rows and columns
- In ML: your entire dataset is a matrix
- Rows = data points (customers), Columns = features
- Shape notation: m x n (m rows, n columns)

| Customer | Age | Income (M UZS) | Data (GB) | Calls/month |
|----------|-----|-----------------|-----------|-------------|
| A        | 22  | 2.0             | 30        | 120         |
| B        | 35  | 5.5             | 12        | 45          |
| C        | 45  | 8.0             | 5         | 200         |

**Speaker notes:** A matrix is just a table of numbers. Your Pandas DataFrame? That's a matrix with labels. When you loaded the Superstore dataset, you were looking at a matrix. The shape — 3 rows, 4 columns — is a 3x4 matrix. ML algorithms don't see column names or row labels. They see a matrix of numbers and operate on it. That's why we convert everything to numbers before modeling.

**Visual:** The customer table above, with row and column indices labeled. Highlight that this is a 3x4 matrix. Show the same data as a plain number grid without headers.

---

### Slide 7 — Matrix Multiplication
**Title:** Matrix Multiplication — The Core ML Operation
- Not element-wise — it's row-by-column dot products
- Matrix A (m x n) times Matrix B (n x p) = Result (m x p)
- Inner dimensions must match: (3x**4**) x (**4**x2) = (3x2)
- Each element in the result = dot product of a row from A and a column from B

**Speaker notes:** Matrix multiplication is where most students get confused, so let's go slow. You take each row from the first matrix and compute its dot product with each column from the second matrix. The result goes in the corresponding position of the output matrix. The key rule: the inner dimensions must match. 3x4 times 4x2 works. 3x4 times 3x2 doesn't.

**Visual:** Matrix multiplication step-by-step visual — two matrices side by side, with one row from A highlighted and one column from B highlighted. Arrows showing element-wise multiply and sum to produce one element of the result. Animate through 2-3 positions.

---

### Slide 8 — Why Matrix Multiplication Matters for ML
**Title:** What's Actually Happening When You Train a Model
- Linear regression: `y = X @ w` (dataset times weights)
- Neural network layer: `output = activation(X @ W + b)`
- The entire training process is repeated matrix multiplications
- GPUs exist because they're extremely good at matrix multiplication
- NumPy: `np.dot(A, B)` or `A @ B`

**Speaker notes:** Every prediction a linear model makes is a matrix multiplication. Your feature matrix X times a weight vector w gives you predictions. Neural networks? Same thing, stacked in layers. The reason GPUs are used for deep learning isn't graphics — it's because GPUs can do thousands of matrix multiplications in parallel. When you hear "this model has 7 billion parameters," those parameters are weights in matrices.

**Visual:** Diagram showing the flow: Input data (matrix) → multiply by weights (matrix) → predictions (vector). Label each component.

---

### Slide 9 — Matrix Operations in NumPy
**Title:** Doing Linear Algebra in Python
- `A = np.array([[1, 2], [3, 4]])`
- Multiply: `A @ B` or `np.dot(A, B)`
- Transpose: `A.T` — swap rows and columns
- Identity matrix: `np.eye(3)` — the "1" of matrices
- Inverse: `np.linalg.inv(A)` — undo a transformation

**Speaker notes:** You don't need to do matrix multiplication by hand. NumPy does it in one line. The @ operator is the cleanest way. Transpose flips rows and columns — you'll use this constantly for reshaping data. The identity matrix is like multiplying by 1 — it doesn't change anything. And the inverse undoes a matrix transformation. These come up in linear regression's closed-form solution.

**Visual:** Code snippets with output shown beside each operation. Transpose visualized as a table being rotated.

---

### Slide 10 — [ACTIVITY] Activity 3: Mathematical Foundations in Practice
**Title:** Activity 3 — Vectors and Matrices with Real Data
[ACTIVITY]
- Represent 5 customers as feature vectors (age, income, data usage, calls)
- Compute the dot product between customer pairs
- Interpret: which customers are most similar?
- Build a feature matrix, compute its transpose
- Bonus: compute a correlation matrix manually using dot products

**Speaker notes:** This is the professor's Activity 3. You'll take 5 hypothetical Uzbek telecom customers, represent them as vectors, and use dot products to measure similarity. This directly connects to how recommendation systems and clustering algorithms work. Start in class, finish as homework.

**Visual:** Activity worksheet — table of 5 customers with feature columns, space for dot product calculations.

---

### Slide 11 — Key Takeaways: Linear Algebra for ML
**Title:** What to Remember
- A vector = one data point = one row in your dataset
- A matrix = your entire dataset
- Dot product = similarity measure between vectors
- Matrix multiplication = the engine of ML predictions
- You don't need to be a mathematician — you need to understand what the operations mean

**Speaker notes:** Here's what matters from today. You don't need to hand-compute matrix multiplications on an exam. You need to understand that when your model makes a prediction, it's multiplying your features by learned weights. When it measures similarity, it's computing dot products. When it processes a dataset, it's doing matrix operations. The intuition is what makes you a better ML practitioner.

**Visual:** Summary diagram — connecting each concept to its ML application: Vector → Features, Matrix → Dataset, Dot Product → Similarity, MatMul → Prediction.

---

### Slide 12 — Homework
**Title:** Before Next Class
- Watch: 3Blue1Brown — Essence of Linear Algebra, Chapters 1-4 (required, ~1.5h)
- Practice: Create 2D and 3D NumPy arrays, perform dot products and matrix multiplications
- [ACTIVITY] Activity 3 continued: complete the customer similarity calculations
- Think about: Why does distance between feature vectors tell us about customer similarity?

**Speaker notes:** The 3Blue1Brown videos are the best visual explanation of linear algebra ever made. Chapters 1 through 4 are required — they'll reinforce everything we covered today with beautiful animations. Watch them before the next class, because we're moving to calculus and probability next, and the linear algebra intuition carries over.

**Visual:** Homework checklist with 3Blue1Brown video thumbnail and link.

---

## CLASS 5: Calculus, Probability, and Statistics (1.5h)

### Slide 1 — Title Slide
**Title:** Calculus, Probability, and Statistics for ML
- Module 2, Class 5
- Derivatives and gradient descent — how models learn
- Probability distributions and Bayes' theorem
- Today's goal: understand the math that drives model training

**Speaker notes:** Last math class. Today we cover three topics that underpin all of machine learning: calculus tells us how to minimize errors, probability tells us how to reason under uncertainty, and statistics tells us how to summarize and test claims about data. We're going for intuition, not rigor.

**Visual:** Title card with three icons — a slope/tangent line (calculus), a bell curve (statistics), a Bayes diagram (probability).

---

### Slide 2 — What is a Derivative?
**Title:** Derivatives — The Slope of a Curve
- A derivative tells you the rate of change at any point
- Slope of a line: rise over run, constant everywhere
- Slope of a curve: changes at every point — that's the derivative
- Positive derivative = function going up; negative = going down; zero = flat (peak or valley)

**Speaker notes:** A derivative is just the slope of a curve at a specific point. If you're driving and the speedometer says 80 km/h, that's the derivative of your position with respect to time — how fast your position is changing right now. For ML, we care about how fast the error is changing as we adjust the model's parameters. That slope tells us which direction to adjust.

**Visual:** Derivative as slope visual — a curve (parabola) with a tangent line drawn at three points: one going up (positive slope), one at the bottom (zero slope), one going down (negative slope). Labels at each point.

---

### Slide 3 — The Loss Function
**Title:** How Wrong is Our Model? — The Loss Function
- Loss function measures the gap between prediction and reality
- Simple example: Mean Squared Error = average of (predicted - actual)^2
- Low loss = good predictions; high loss = bad predictions
- Our goal: find the model parameters that minimize the loss
- The loss is a function of the weights — it forms a surface

**Speaker notes:** Before gradient descent makes sense, you need the loss function. It's a formula that scores how bad your model is. Take your predictions, subtract the real values, square the differences, and average them. That's MSE. The key insight: the loss depends on your model's parameters. Change the parameters, the loss changes. We want to find the parameters that make the loss as small as possible.

**Visual:** Loss function concept — a plot where x-axis is a model parameter (e.g., slope of a line), y-axis is the loss value. The curve is U-shaped. The minimum point is labeled "Best parameter value."

---

### Slide 4 — Gradient Descent
**Title:** Gradient Descent — Walking Downhill to Find the Best Model
- Start with random parameters
- Compute the gradient (slope) of the loss
- Take a step in the direction that reduces the loss
- Repeat until you reach the bottom (minimum loss)
- Learning rate controls step size — too big = overshoot, too small = slow

**Speaker notes:** Gradient descent is how ML models learn. Imagine you're blindfolded on a hilly landscape and you need to find the lowest valley. You feel the slope under your feet. If it's tilting left, you step left. If it's tilting right, you step right. Each step takes you a bit lower. That's gradient descent. The gradient is the slope, the learning rate is your step size. This is literally what happens millions of times when you train a neural network.

**Visual:** Gradient descent as walking downhill — a 3D loss surface (bowl shape) with a ball rolling downhill in steps. Each step labeled with an arrow. Show three scenarios: good learning rate (reaches bottom), too large (bounces around), too small (barely moves).

---

### Slide 5 — Gradient Descent: Concrete Example
**Title:** Fitting a Line to Data
- We have data points: Tashkent apartment sizes and rents
- Model: `rent = w * size + b` (linear regression)
- Loss: MSE between predicted and actual rents
- Gradient descent adjusts w and b to minimize MSE
- After many iterations: the line fits the data

| Size (m2) | Rent (M UZS) |
|-----------|--------------|
| 40        | 3.2          |
| 60        | 4.5          |
| 80        | 6.1          |
| 100       | 7.8          |

**Speaker notes:** Let's make this concrete. You have apartment sizes and monthly rents in Tashkent. You want a formula that predicts rent from size. Start with a random line — it's probably way off. Compute the loss — how far the line is from the actual data. Compute the gradient — which direction should you tilt the line? Adjust. Repeat. After enough iterations, the line fits the data. That's linear regression trained with gradient descent.

**Visual:** Scatter plot of apartment data with an evolving best-fit line — show 3 iterations: bad fit (random), better fit (after 10 steps), good fit (converged). Arrows showing the line rotating into position.

---

### Slide 6 — Partial Derivatives and Gradients
**Title:** Multiple Parameters = Multiple Slopes
- When the loss depends on multiple parameters, each gets its own derivative
- Partial derivative: the slope with respect to one parameter, holding others constant
- Gradient = vector of all partial derivatives
- Points in the direction of steepest increase (we go the opposite way)

**Speaker notes:** In our rent example, the loss depends on both w (slope) and b (intercept). The partial derivative with respect to w tells you how the loss changes if you nudge w. The partial derivative with respect to b tells you the same for b. The gradient is just both of these packed into a vector. It points uphill, so we go in the opposite direction. Real neural networks have millions of parameters, but the principle is the same.

**Visual:** Loss surface 3D visual — a bowl-shaped surface over a 2D parameter plane (w, b). Gradient arrows on the surface pointing uphill. Descent path marked going downhill to the minimum.

---

### Slide 7 — Probability: The Language of Uncertainty
**Title:** Probability — Quantifying "How Likely?"
- Probability: a number between 0 (impossible) and 1 (certain)
- P(customer churns) = 0.15 → 15% chance of leaving
- Joint probability: P(A and B) — both events happening
- Conditional probability: P(A | B) — probability of A given B already happened
- ML predictions are probabilities: "This email is 94% likely to be spam"

**Speaker notes:** ML models don't give you yes or no — they give you probabilities. A churn model says "this customer has a 73% probability of leaving." A spam filter says "this email is 94% spam." To understand what those numbers mean and when to trust them, you need basic probability. It's not complicated — probability is just a fraction of favorable outcomes over total outcomes. The interesting part is combining probabilities.

**Visual:** Probability scale bar — 0 (impossible) to 1 (certain), with examples placed along it: 0.01 (meteorite hits your house), 0.15 (customer churns), 0.5 (coin flip), 0.94 (spam classification), 1.0 (sun rises tomorrow).

---

### Slide 8 — Probability Distributions
**Title:** Distributions — The Shape of Randomness
- **Uniform**: every outcome equally likely (rolling a fair die)
- **Normal (Gaussian)**: bell curve — most values near the mean (heights, test scores)
- **Binomial**: number of successes in n trials (how many customers out of 100 churn?)
- Key stats: **mean** (center), **variance** (spread), **standard deviation** (sqrt of variance)

**Speaker notes:** A distribution tells you the pattern of possible outcomes. Roll a die — each number is equally likely, that's uniform. Measure heights of 1000 Uzbek university students — most are near the average, fewer are very tall or very short, that's a normal distribution. In ML, we assume our data comes from some distribution, and our job is to learn what that distribution looks like.

**Visual:** Three distribution plots side by side — Uniform (flat rectangle), Normal (bell curve with mean and std dev marked), Binomial (bar chart for n=20, p=0.3). Label mean and spread on each.

---

### Slide 9 — The Normal Distribution
**Title:** The Bell Curve — Why It's Everywhere
- Most real-world measurements are approximately normal
- Defined by just two parameters: mean (mu) and standard deviation (sigma)
- 68-95-99.7 rule: 68% within 1 std, 95% within 2, 99.7% within 3
- Central Limit Theorem: averages of random samples tend toward normal, regardless of the original distribution

**Speaker notes:** The normal distribution is the most important distribution in statistics. It shows up everywhere — test scores, measurement errors, biological traits. It's defined by just two numbers: the mean and the standard deviation. The 68-95-99.7 rule is a quick way to remember how data spreads. And the Central Limit Theorem is almost magical — even if your raw data isn't normal, the average of samples from it will be.

**Visual:** Bell curve with the 68-95-99.7 rule illustrated — shaded regions at 1, 2, 3 standard deviations from the mean, each labeled with the percentage.

---

### Slide 10 — Bayes' Theorem
**Title:** Bayes' Theorem — Updating Beliefs with Evidence
- P(A|B) = P(B|A) * P(A) / P(B)
- In words: "How should I update my belief in A after seeing evidence B?"
- Example: A customer complains (B). What's the probability they'll churn (A)?
- Prior: P(churn) = 15% (base rate)
- Likelihood: P(complaint | churn) = 60% (churners complain more)
- Evidence: P(complaint) = 25% (overall complaint rate)
- Posterior: P(churn | complaint) = 0.60 * 0.15 / 0.25 = 36%

**Speaker notes:** Bayes' theorem is how you update your beliefs with new evidence. You start with a prior — your initial belief. Then you get evidence, and Bayes tells you the posterior — your updated belief. In our example, the base churn rate is 15%. But if a customer has complained, the probability jumps to 36%. This is the foundation of Bayesian classification and spam filters.

**Visual:** Bayes' theorem diagram — tree diagram showing the branching probabilities. Start with Churn/No Churn, then branch into Complaint/No Complaint. Highlight the path: Churn → Complaint. Show the calculation step by step.

---

### Slide 11 — Descriptive Statistics Recap
**Title:** Statistics You Already Know (And Need for ML)
- **Mean**: average — sensitive to outliers
- **Median**: middle value — robust to outliers
- **Mode**: most frequent value
- **Standard deviation**: how spread out the data is
- **Percentiles**: Q1 (25th), Q2 (50th = median), Q3 (75th)
- These are the numbers behind `df.describe()`

**Speaker notes:** These are the stats you already computed with Pandas describe(). Mean tells you the center but gets pulled by outliers. Median is more robust. Standard deviation tells you the spread — small means tightly packed, large means scattered. Percentiles divide your data into chunks. The box plots from last class? Those are built from percentiles. Now you know the math behind the visualization.

**Visual:** A histogram with overlay lines marking mean, median, and standard deviation. Show a skewed distribution where mean and median differ.

---

### Slide 12 — Hypothesis Testing (Conceptual)
**Title:** Hypothesis Testing — Is This Result Real or Random?
- Null hypothesis (H0): "There's no difference" / "This happened by chance"
- Alternative hypothesis (H1): "There is a real effect"
- p-value: probability of seeing this result if H0 is true
- p < 0.05 → reject H0, the result is statistically significant
- ML application: Is this feature actually useful? Is model A actually better than model B?

**Speaker notes:** Hypothesis testing asks: "Is this result real or did I get lucky?" The null hypothesis says nothing is happening — any pattern is just noise. The p-value tells you how likely you'd see your result by pure chance. Below 0.05, we call it significant. In ML, you'll use this to decide if a feature matters, if a model improvement is real, or if A/B test results are trustworthy. You don't need to compute p-values by hand — Python does it — but you need to understand what they mean.

**Visual:** Two overlapping bell curves — one for "null hypothesis" (no effect), one for "alternative hypothesis." Shade the tail area (p-value region) on the null curve.

---

### Slide 13 — Connecting Math to ML
**Title:** Everything Connects
- **Linear algebra** → Data representation, model computation
- **Calculus** → Training models via gradient descent
- **Probability** → Predictions as likelihoods, Bayesian reasoning
- **Statistics** → Understanding data, evaluating models
- You now have the mathematical vocabulary for ML

**Speaker notes:** Take a step back. Linear algebra is how we represent and transform data. Calculus is how models learn from mistakes. Probability is how models express uncertainty. Statistics is how we validate results. You don't need to be a mathematician, but understanding these concepts means you'll know what's happening inside the black box. That's the difference between using ML and understanding ML.

**Visual:** Four quadrants — Linear Algebra, Calculus, Probability, Statistics — each with an icon and its ML connection. Arrows from each to a central "Machine Learning" circle.

---

### Slide 14 — Homework
**Title:** Before the Lab
- Watch: 3Blue1Brown — "What is a gradient?" and "Gradient descent, how neural networks learn"
- Watch: StatQuest — "Histograms" and "P-values" (optional but helpful)
- Review: your Activity 3 calculations — ensure dot products are correct
- Prepare: download Superstore dataset to Colab — we're doing a full EDA lab next class

**Speaker notes:** Next class is the lab — we're putting everything together. You'll do a full exploratory data analysis combining Pandas, visualizations, and basic statistics. We'll also cover Google Cloud at a conceptual level and kick off Activity 4, which is customer segmentation. Come prepared with your Colab environment ready and the dataset loaded.

**Visual:** Homework checklist with video thumbnails and links.

---

## CLASS 6: Lab — EDA + Google Cloud Overview + Activity 4 (1.5h)

### Slide 1 — Title Slide
**Title:** Lab Day — EDA, Google Cloud, and Customer Segmentation
- Module 2, Class 6
- Part 1: EDA workflow recap (20 min)
- Part 2: Google Cloud for Data Science — conceptual overview (25 min)
- Part 3: Activity 4 — Customer Segmentation Simulation (40 min)
- Wrap-up and module review (15 min)

**Speaker notes:** Lab day. Today is about applying everything from this module. We'll start with a quick recap of the EDA workflow, then I'll give you a conceptual tour of Google Cloud tools for data science, and then we'll spend the bulk of the time on Activity 4 — customer segmentation. This is where linear algebra meets Pandas meets visualization.

**Visual:** Title card with three-part agenda, time allocations shown as a horizontal bar.

---

### Slide 2 — EDA Workflow Recap
**Title:** The Standard EDA Process
1. **Load and inspect**: shape, dtypes, head, describe
2. **Clean**: handle missing values, fix data types, remove duplicates
3. **Explore distributions**: histograms for numeric, value_counts for categorical
4. **Explore relationships**: scatter plots, correlation heatmap
5. **Summarize findings**: what patterns did you find? what's unexpected?

**Speaker notes:** Before we start the lab, let's formalize the EDA process. These five steps are what you do every time you get a new dataset. Load it, inspect it, clean it, visualize it, summarize what you found. This isn't a rigid checklist — it's iterative. You'll often go back and forth between steps as you discover things.

**Visual:** Flowchart — five steps connected with arrows, with a feedback loop from step 5 back to step 3 (showing iteration).

---

### Slide 3 — EDA Checklist
**Title:** Before You Model: The EDA Checklist
- How many rows and columns?
- Any missing values? How many, in which columns?
- What are the data types? Any mismatches (numbers stored as strings)?
- What do the distributions look like? Any extreme outliers?
- Which variables correlate with each other?
- Are there any duplicates?
- What's the target variable's distribution (if supervised learning)?

**Speaker notes:** Print this out or save it. Every time you start a new project, run through this list. It takes 15 minutes and saves you hours of debugging later. The number of ML models that fail because someone didn't check for missing values or outliers is staggering.

**Visual:** Printable checklist layout — checkbox next to each item. Clean, minimal design.

---

### Slide 4 — Google Cloud for Data Science
**Title:** Google Cloud Platform — The Big Picture
- Cloud = someone else's computers that you rent
- Why cloud? Scale, GPUs/TPUs, managed services, collaboration
- Key services for data science:
  - **BigQuery**: SQL on massive datasets (serverless, fast)
  - **Vertex AI**: End-to-end ML platform (training, deployment, monitoring)
  - **Cloud Storage**: Store datasets, models, artifacts
  - **Colab Enterprise**: Managed notebooks with cloud compute

**Speaker notes:** We've been using Google Colab, which runs on Google's cloud. But Google Cloud Platform is much bigger. BigQuery lets you run SQL on datasets with billions of rows without managing any servers. Vertex AI lets you train and deploy ML models at scale. Cloud Storage is like Google Drive but for big data. You won't use these hands-on today, but understanding what's available is important.

**Visual:** Google Cloud architecture diagram — simplified. Show: Data sources → Cloud Storage → BigQuery (analysis) → Vertex AI (modeling) → Deployment. Use Google Cloud icons.

---

### Slide 5 — BigQuery: SQL at Scale
**Title:** BigQuery — Query Massive Data Without Infrastructure
- Write SQL, get results in seconds — even on terabytes
- No servers to manage, no indexes to configure
- Pay per query (first 1 TB/month free)
- Directly connects to Vertex AI for ML
- Example: Uztelecom analyzing call records across all customers — millions of rows in seconds

**Speaker notes:** BigQuery is Google's flagship data tool. Imagine running a SQL query on a table with 100 million rows and getting the result in 5 seconds. No database to install, no cluster to manage. You just write SQL. The free tier is generous enough for learning. In a real telecom company, this is how you'd analyze call records, customer behavior, network logs at scale.

**Visual:** BigQuery interface screenshot (or mockup) — SQL editor on top, results table on bottom. Show a simple query and its result.

---

### Slide 6 — Vertex AI: ML Platform
**Title:** Vertex AI — Train, Deploy, Monitor ML Models
- AutoML: train models without writing code
- Custom training: bring your own Python code + framework
- Model Registry: version and manage trained models
- Endpoints: deploy models as APIs (send data, get predictions)
- Pipelines: automate the entire ML workflow

**Speaker notes:** Vertex AI is where Google Cloud and ML meet. It supports two modes: AutoML, where you upload data and it builds a model for you, and custom training, where you write your own code. We'll use it in later modules when we deploy models. For now, just know it exists and what it can do. The professor wants you to explore it through Google Skills Boost.

**Visual:** Vertex AI workflow diagram — Data → Feature Store → Training (AutoML or Custom) → Model Registry → Endpoint (API) → Monitoring. Linear flow with icons.

---

### Slide 7 — Google Skills Boost
**Title:** Your Cloud Learning Path
- Google Cloud Big Data and ML Fundamentals (required for this module)
- Free labs and courses on skills.google.com
- Earn badges that go on your LinkedIn profile
- Complete the required course as homework this week

**Speaker notes:** The professor requires one Google Skills Boost course per module. For Module 2, it's "Google Cloud Big Data and Machine Learning Fundamentals." It's free, takes about 8 hours total, and gives you hands-on experience with BigQuery, Cloud Storage, and Vertex AI. You earn a badge when you finish. Do it this week — it's required, not optional.

**Visual:** Google Skills Boost course card — title, estimated time, difficulty level. QR code linking to the course.

---

### Slide 8 — [ACTIVITY] Activity 4: Customer Segmentation Simulation
**Title:** Activity 4 — Segment Customers Using What You've Learned
[ACTIVITY]

**Scenario:** You work at an Uzbek telecom company. Management wants to understand customer segments to design targeted plans.

**Dataset:** Use the Superstore dataset as a proxy, or a provided telecom-style CSV with:
- Customer ID, Age, Monthly spend, Data usage (GB), Calls/month, Region

**Tasks:**
1. Load and inspect the data (EDA checklist)
2. Compute summary statistics per region
3. Create feature vectors for each customer
4. Visualize: scatter plot of spend vs data usage, colored by region
5. Compute correlation matrix and plot heatmap
6. Identify 2-3 distinct customer groups from your visualizations
7. Write a short report: Who are the segments? What would you recommend?

**Speaker notes:** This is the big activity for Module 2. It brings together Pandas, visualization, linear algebra concepts, and basic statistics. You're playing the role of a data analyst at a telecom company. The manager doesn't care about code — they want to know who the customer segments are and what to do about them. Your deliverable is a notebook with charts and a written summary. Start now, finish as homework.

**Visual:** Activity overview — numbered steps as a workflow, with icons for each tool used (Pandas, Matplotlib, NumPy).

---

### Slide 9 — Live Walkthrough: Getting Started
**Title:** Let's Start Together
- Step 1: Load the dataset in Colab
- Step 2: Run the EDA checklist (shape, info, describe, null check)
- Step 3: What's the first chart you'd make? (Discuss)
- Step 4: Start coding — 20 minutes of independent work

**Speaker notes:** Let's do the first two steps together so everyone's on the same page. Load the data, run the inspection commands. Now — what chart would you make first? Think about what question you're trying to answer. I'll give you 20 minutes to work independently. I'll walk around and help. Don't worry about making it perfect — the goal is exploration.

**Visual:** Colab notebook — live screen share for setup, then students work independently.

---

### Slide 10 — Segmentation Thinking
**Title:** How to Think About Customer Segments
- Look for clusters in scatter plots — groups of dots that bunch together
- High spend + high data = premium users
- Low spend + high calls = traditional users (voice-heavy)
- Low everything = at-risk / inactive customers
- Segments should be actionable — "So what?" for each group

**Speaker notes:** When you make your scatter plots, look for natural groupings. Are there dots that cluster together? Maybe you see a group of heavy data users who spend a lot — those are your premium customers. Another group uses mostly voice calls and spends less. A third group barely uses the service at all — those might be about to churn. The key is: each segment should lead to an action. Premium users get loyalty rewards. At-risk users get a retention call.

**Visual:** Annotated scatter plot — spend vs data usage with three circled clusters labeled "Premium," "Traditional," "At-risk." Each cluster has a text box with a recommended action.

---

### Slide 11 — Lab Work Time
**Title:** Independent Work (20 minutes)
- Complete Activity 4 Steps 1-5
- Ask questions — I'm here to help
- If stuck: check the Pandas cheat sheet from Class 2
- If done early: start on Step 6 (identifying segments)

**Speaker notes:** (No narration — walk around, help students, answer questions. Monitor progress and provide individual guidance.)

**Visual:** Timer display — 20 minutes. Activity steps listed for reference.

---

### Slide 12 — Group Discussion: What Did You Find?
**Title:** Share Your Findings
- What did the data look like? Any surprises?
- Which chart was most informative?
- Did you spot any customer segments?
- What would you tell the management?

> Open discussion — 2-3 students share their screens and walk through their analysis.

**Speaker notes:** Let's hear what you found. Who wants to share their screen? I want to see your charts and hear your interpretation. There's no single right answer — different people might see different patterns. That's normal in data analysis. The important thing is that your interpretation is supported by the data.

**Visual:** Student screen shares — no prepared visual.

---

### Slide 13 — Module 2 Recap
**Title:** What We Covered in Module 2
- **Python**: variables, types, control flow, functions — the language of data science
- **NumPy & Pandas**: arrays, DataFrames, filtering, aggregation — the tools
- **Visualization**: histograms, scatter plots, bar charts, box plots, heatmaps — the lens
- **Linear algebra**: vectors, matrices, dot products — the math behind ML
- **Calculus**: derivatives, gradient descent — how models learn
- **Probability & stats**: distributions, Bayes, hypothesis testing — reasoning under uncertainty
- **Google Cloud**: BigQuery, Vertex AI — the infrastructure

**Speaker notes:** Let's zoom out. In two weeks, you went from opening Colab for the first time to doing exploratory data analysis with real data and understanding the math behind ML. That's a lot. You don't need to have everything memorized — you need to know where to look and what the concepts mean. These foundations will carry you through every module that follows.

**Visual:** Module 2 content map — six boxes arranged in two rows (Python, NumPy/Pandas, Visualization in row 1; Linear Algebra, Calculus/Probability, Google Cloud in row 2), with connecting lines showing how they relate.

---

### Slide 14 — Homework and Looking Ahead
**Title:** Final Homework for Module 2
- Complete Activity 4: full notebook with charts, interpretations, and segment descriptions
- Complete Google Skills Boost: "Google Cloud Big Data and ML Fundamentals"
- Review: all Kaggle micro-courses (Python, Pandas, Data Visualization)
- Prepare for Module 3: Data Preparation and Feature Engineering

**What's next:** Module 3 dives into data cleaning, feature engineering, and handling messy real-world data. Everything you learned here is the foundation for that.

**Speaker notes:** Your final deliverable for Module 2 is the Activity 4 notebook — clean, well-documented, with charts that tell a story. Also finish the Google Skills Boost course if you haven't already. Module 3 starts with data cleaning — missing values, outliers, encoding — all the messy stuff that real data throws at you. The tools you learned this module are exactly what you'll use.

**Visual:** Checklist of final deliverables. Preview card for Module 3 topics.

---

### Slide 15 — Thank You + Q&A
**Title:** Questions?
- Office hours: [time/channel]
- Resources: Module 2 resources.md (all links in one place)
- Struggling? That's normal. Revisit the 3Blue1Brown videos and Kaggle exercises.
- "The best time to learn Python was 5 years ago. The second best time is now."

**Speaker notes:** Any questions before we wrap up? If something from the math classes didn't fully click, watch the 3Blue1Brown videos again — they explain it better than anyone. If the Python side is shaky, the Kaggle courses have infinite exercises. You're building the foundation for everything that comes next. See you in Module 3.

**Visual:** Q&A slide — clean, minimal. Contact info and resource links.
