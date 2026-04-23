# Variables and data types:
# --- EXAMPLES ---

# Integers (Whole numbers)
# Using underscores (10_000_000) makes large numbers easier to read for humans;
# Python ignores them during execution.
subscriber_count = 10_000_000
age = 25


# Floats (Decimal numbers)
# Used for precise values like currency or percentage-based metrics.
monthly_bill = 89_500.50
accuracy = 0.95


# Strings
# Used for names, locations, or any categorical text data.
customer_name = "Abdullayev Jasur"
region = "Tashkent"


# Booleans
# Often used as "flags" to check status (e.g., is the user logged in?).
is_active = True
payment_on_time = False


# Lists
# A collection of integers representing bill history over 5 months.
monthly_bills = [85000, 92000, 78000, 95000, 88000]
# A collection of strings representing different service areas.
regions = ["Tashkent", "Samarkand", "Bukhara", "Fergana"]


# Check types using f-strings (formatted strings)

# Prints the count and confirms it is an 'int'
print(f"subscriber_count: {subscriber_count} (type: {type(subscriber_count).__name__})")

# Prints the bill and confirms it is a 'float'
print(f"monthly_bill:     {monthly_bill} (type: {type(monthly_bill).__name__})")

# Prints the name and confirms it is a 'str'
print(f"customer_name:    {customer_name} (type: {type(customer_name).__name__})")

# Prints the status and confirms it is a 'bool'
print(f"is_active:        {is_active} (type: {type(is_active).__name__})")

# Prints the full list and confirms it is a 'list'
print(f"monthly_bills:    {monthly_bills} (type: {type(monthly_bills).__name__})")


# --- List operations ---

# Access by index (0-based)
# monthly_bills[0] gets the very first element (85000)
print(f"First bill: {monthly_bills[0]}")

# monthly_bills[-1] is a Pythonic shortcut to get the last element (88000)
print(f"Last bill:  {monthly_bills[-1]}")

# Slicing
# monthly_bills[:3] takes everything from the start up to (but not including) index 3.
# This returns indices 0, 1, and 2.
print(f"First 3 bills: {monthly_bills[:3]}")

# Length
# Very useful for loops or calculating averages (Total / len).
print(f"Number of bills: {len(monthly_bills)}")

# Append
# This modifies the original list 'in-place'.
monthly_bills.append(91000)
print(f"After append: {monthly_bills}")


# --- TODO: Variables exercise ---
# Create variables for a telecom customer profile:

# TODO: Create an integer variable for the customer's age
# Integer (int): Represents discrete, whole values. 
# Used for counts or identifiers where decimals make no sense.
customer_age = 20

# TODO: Create a float variable for their monthly data usage in GB
# Float (float): Used for continuous values requiring precision.
# In computing, floats are how we handle measurements and fractional amounts.
data_usage_gb = 25.5

# TODO: Create a string variable for their city
# String (str): A sequence of characters. 
# Strings are immutable in Python and used for any non-numeric labels or data.
city = "Tashkent"

# TODO: Create a boolean for whether they have a contract (vs prepaid)
# String (str): A sequence of characters. 
# Strings are immutable in Python and used for any non-numeric labels or data.
has_contract = True

# TODO: Create a list of their last 6 monthly bills (make up reasonable numbers in soum)
# List (list): A collection of objects in a specific order. 
# Lists are "mutable," meaning you can add, remove, or change values later.
bills = [85000, 92000, 78000, 110000, 95000, 88000]

# TODO: Print all variables with their types (follow the example pattern above)
print(f"Age: {customer_age} | Type: {type(customer_age)}")
print(f"Usage: {data_usage_gb} GB | Type: {type(data_usage_gb)}")
print(f"City: {city} | Type: {type(city)}")
print(f"Has Contract: {has_contract} | Type: {type(has_contract)}")
print(f"Bills: {bills} | Type: {type(bills)}")



# Dictionaries

# --- EXAMPLE ---

# A Dictionary is a collection of Key-Value pairs.
# Think of it like a real-world database record for a single user.
customer = {
    "id": "UZT-001234",                   # Key (id) : Value (String)
    "name": "Karimov Sherzod",
    "region": "Tashkent",
    "monthly_spend": 125000,              # Key (monthly_spend) : Value (Integer)
    "is_active": True,                    # Key (is_active) : Value (Boolean)
    "plan": "Premium"
}

# Access values by key
# Unlike lists where you use an index (0, 1, 2), in dictionaries you use the Key.
# This makes your code readable: anybody knows what 'name' refers to.
print(f"Customer: {customer['name']}")
print(f"Region:   {customer['region']}")
print(f"Spend:    {customer['monthly_spend']} soum")

# Add a new key
# You can add new data or change existing data on the fly.
customer["tenure_months"] = 24                 # Adds a new key-value pair
print(f"\nFull record: {customer}")            # Updates the existing value for 'plan'


# --- TODO: Dictionary exercise ---
# Create a dictionary representing a product from an e-commerce store
# Include at least 5 keys: name, category, price, quantity_sold, rating

# TODO: Create the product dictionary
product = {
    "name": "Samsung Galaxy S24",
    "category": "Electronics",
    "price": 9500000,         # Price in soum
    "quantity_sold": 150,     # Integer for sales tracking
    "rating": 4.8             # Float for precise user feedback
}

# TODO: Print the product name and price
print(f"Product Name: {product['name']}")
print(f"Current Price: {product['price']:,} soum")

# TODO: Add a new key "discount" with a value between 0 and 1
product["discount"] = 0.15
print(f"Applied Discount: {product['discount'] * 100}%")

# TODO: Print all keys using product.keys()
print(f"\nProduct Data Keys: {product.keys()}")


# Functions
 # --- EXAMPLE: calculate_average ---


# 'def' tells Python: "I am DEFINING a function now."
# 'calculate_average' is the name (use descriptive verbs!).
# 'numbers' is a PARAMETER—it’s a placeholder for the data we will pass in.
def calculate_average(numbers):
    """Calculate the average of a list of numbers."""  # This is a Docstring. It explains what the function does.
    total = sum(numbers)                               # sum() is a built-in Python function that adds everything in a list.
    count = len(numbers)                               # len() counts how many items are in the list.

    # 'return' is the most important part. It "spits out" the result 
    # so you can save it to a variable outside the function.
    return total / count


# --- Using the Function ---

bills = [85000, 92000, 78000, 95000, 88000, 91000]

# We CALL the function and pass our 'bills' list into it.
avg_bill = calculate_average(bills)

# Formatting: {avg_bill:,.0f} means:
# ',' -> Add thousands separators (e.g., 85,000)
# '.0f' -> Show 0 decimal places (round to the nearest whole number)
print(f"Average monthly bill: {avg_bill:,.0f} soum")

# --- EXAMPLE: function with multiple returns ---

def describe_bills(bills):
    """Return basic statistics for a list of bills."""
    """
    Analyzes billing data and returns stats.
    In a professional setting, this is called 'Data Aggregation'.
    """
    avg = sum(bills) / len(bills)
    minimum = min(bills)             # Built-in function to find the smallest value
    maximum = max(bills)             # Built-in function to find the largest value

    # Python packs these into a Tuple automatically
    return avg, minimum, maximum


# --- UNPACKING ---

# This is called 'Sequence Unpacking'. 
# The order must match the order in the return statement!
avg, low, high = describe_bills(bills)
avg, low, high = describe_bills(bills)
print(f"Average: {avg:,.0f} | Min: {low:,.0f} | Max: {high:,.0f}")

# --- TODO: Write your own functions ---

# TODO 1: Write a function called `classify_customer` that takes monthly_spend as input
#         and returns:
#         - "premium" if spend > 150000
#         - "standard" if spend is between 50000 and 150000 (inclusive)
#         - "basic" if spend < 50000

def classify_customer(monthly_spend):

    # TODO: implement this function
    pass
    if monthly_spend > 150000:
        return "premium"
    elif 50000 <= monthly_spend <= 150000:
        # Python allows this cool 'range' comparison: 50k <= spend <= 150k
        return "standard"
    else:
        # If it's not premium and not standard, it must be basic.
        return "basic"

# Test your function:
# print(classify_customer(200000))  # should print "premium"
# print(classify_customer(75000))   # should print "standard"
# print(classify_customer(30000))   # should print "basic"

print(f"Spend 200,000: {classify_customer(200000)}")  # Output: premium
print(f"Spend 75,000:  {classify_customer(75000)}")   # Output: standard
print(f"Spend 30,000:  {classify_customer(30000)}")   # Output: basic


# --- TODO: Write your own functions ---

# TODO 1: Write a function called `classify_customer` that takes monthly_spend as input
#         and returns:
#         - "premium" if spend > 150000
#         - "standard" if spend is between 50000 and 150000 (inclusive)
#         - "basic" if spend < 50000

def classify_customer(monthly_spend):

    # TODO: implement this function
    pass
    if monthly_spend > 150000:
        return "premium"
    elif 50000 <= monthly_spend <= 150000:
        # Python allows this cool 'range' comparison: 50k <= spend <= 150k
        return "standard"
    else:
        # If it's not premium and not standard, it must be basic.
        return "basic"

# Test your function:
# print(classify_customer(200000))  # should print "premium"
# print(classify_customer(75000))   # should print "standard"
# print(classify_customer(30000))   # should print "basic"

print(f"Spend 200,000: {classify_customer(200000)}")  # Output: premium
print(f"Spend 75,000:  {classify_customer(75000)}")   # Output: standard
print(f"Spend 30,000:  {classify_customer(30000)}")   # Output: basic


# Loops
# --- EXAMPLE: basic for loop ---

# This is our data source (a List of strings)
regions = ["Tashkent", "Samarkand", "Bukhara", "Fergana", "Namangan"]
# 'region' is a temporary variable name we create on the spot.
# In the first loop, region = "Tashkent"
# In the second loop, region = "Samarkand"... and so on.
for region in regions:
  # Everything indented (pushed to the right) belongs to the loop.
    # This line runs once for EVERY item in the list.
    print(f"Processing data for: {region}")


# --- EXAMPLE: enumerate (index + value) ---
# enumerate gives you both the position and the value

monthly_bills = [85000, 92000, 78000, 95000, 88000, 91000]


# 'i' is the index (starts at 0 by default)
# 'bill' is the actual value from the list
for i, bill in enumerate(monthly_bills):

  # We use i + 1 because humans start counting at 1, 
    # but computers start at 0.
    print(f"Month {i + 1}: {bill:,} soum")


# --- EXAMPLE: loop with conditional ---

customers = [
    {"name": "Alisher", "spend": 200000},
    {"name": "Dilnoza", "spend": 45000},
    {"name": "Bobur",   "spend": 120000},
    {"name": "Malika",  "spend": 35000},
    {"name": "Rustam",  "spend": 180000},
]

print("High-value customers (spend > 100,000):")

# 1. The Loop: Iterates through the list of dictionaries
for c in customers:

  # 2. The Conditional: Accesses the value of the 'spend' key
    # and compares it to the threshold (100,000)
    if c["spend"] > 100000:

      # 3. The Action: This only runs if the 'if' condition is True
        print(f"  {c['name']}: {c['spend']:,} soum")


# --- TODO: Loop exercises ---

# TODO 1: Using the `customers` list above, loop through and classify each customer
#         using your classify_customer function from Section 3.
#         Print: "Name: <name>, Tier: <tier>"

# Your code here:
customers = [
    {"name": "Alisher", "spend": 200000},
    {"name": "Dilnoza", "spend": 45000},
    {"name": "Bobur",   "spend": 120000},
    {"name": "Malika",  "spend": 35000},
    {"name": "Rustam",  "spend": 180000},
]

for c in customers:
    spend_value = c["spend"]
    tier = classify_customer(spend_value)
    print(f"Name: {c['name']:<8}, Tier: {tier}")



# TODO 2: Given the list of scores below, use enumerate to print each score
#         with its rank (starting from 1), and mark scores above 90 with " -- HIGH"
#
# Expected output:
#   Rank 1: 72
#   Rank 2: 95 -- HIGH
#   Rank 3: 88
#   ...

scores = [72, 95, 88, 61, 93, 77, 85, 91, 68, 99]

# Your code here:
scores = [72, 95, 88, 61, 93, 77, 85, 91, 68, 99]

for rank, score in enumerate(scores, start=1):
    label = ""
    if score > 90:
        label = " -- HIGH"
    print(f"Rank {rank}: {score}{label}")


# TODO 3: Write a loop that computes the total and average of the scores list.
#         Do NOT use sum() or len() -- compute manually with a loop.
#         Print: "Total: <total>, Average: <average>"

# Your code here:
scores = [72, 95, 88, 61, 93, 77, 85, 91, 68, 99]
total_sum = 0
count = 0
for score in scores:
    total_sum += score  
    count += 1         
if count > 0:
    average = total_sum / count
else:
    average = 0
print(f"Total: {total_sum}, Average: {average}")



# Putting it all together

# --- TODO: Final exercise ---
#
# You have a dataset of 8 customers (list of dictionaries).
# Write code that:
# 1. Loops through each customer
# 2. Classifies them into tiers (use classify_customer or write inline logic)
# 3. Counts how many customers are in each tier
# 4. Prints a summary: "Premium: X, Standard: Y, Basic: Z"

customers_data = [
    {"id": 1, "name": "Anvar",   "monthly_spend": 210000},
    {"id": 2, "name": "Nilufar", "monthly_spend": 55000},
    {"id": 3, "name": "Sardor",  "monthly_spend": 180000},
    {"id": 4, "name": "Gulnora", "monthly_spend": 32000},
    {"id": 5, "name": "Timur",   "monthly_spend": 95000},
    {"id": 6, "name": "Madina",  "monthly_spend": 160000},
    {"id": 7, "name": "Otabek",  "monthly_spend": 47000},
    {"id": 8, "name": "Zarina",  "monthly_spend": 125000},
]

# Your code here:
customers_data = [
    {"id": 1, "name": "Anvar",   "monthly_spend": 210000},
    {"id": 2, "name": "Nilufar", "monthly_spend": 55000},
    {"id": 3, "name": "Sardor",  "monthly_spend": 180000},
    {"id": 4, "name": "Gulnora", "monthly_spend": 32000},
    {"id": 5, "name": "Timur",   "monthly_spend": 95000},
    {"id": 6, "name": "Madina",  "monthly_spend": 160000},
    {"id": 7, "name": "Otabek",  "monthly_spend": 47000},
    {"id": 8, "name": "Zarina",  "monthly_spend": 125000},
]

stats = {
    "premium": 0,
    "standard": 0,
    "basic": 0
}

for customer in customers_data:
    spend = customer["monthly_spend"]
    if spend > 150000:
        tier = "premium"
    elif spend >= 50000:
        tier = "standard"
    else:
        tier = "basic"
    stats[tier] += 1
print("--- Customer Tier Summary ---")
print(f"Premium: {stats['premium']}, Standard: {stats['standard']}, Basic: {stats['basic']}")





















