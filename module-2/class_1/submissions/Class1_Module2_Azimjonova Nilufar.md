# This prints a simple string word
print("Hello, AI/ML")


Task 2: Variables and Data Types

# I create a variable called age and store an integer value (whole number)
age = 22

# I create another integer variable for GPA
gpa = 5

# This is a string (text), so I use quotes
name = "Elyor"

# This is a boolean value, meaning True or False
is_student = True

# This is a list, which can store multiple values in one variable
scores = [88, 83, 92, 95, 90]

# Print the value of age and its data type
print(age, type(age))

# Print the value of gpa and its data type
print(gpa, type(gpa))

# Print the value of name and its data type
print(name, type(name))

# Print the value of is_student and its data type
print(is_student, type(is_student))

# Print the list and its data type
print(scores, type(scores))


Task 3: Writing a Function

# I define a function that calculates the average of a list
def calculate_average(numbers):
    # Add all numbers in the list
    total = sum(numbers)

    # Count how many numbers are in the list
    count = len(numbers)

    # Return the average (total divided by count)
    return total / count

# Example lists of numbers
test1 = [200, 650, 653, 344, 368]
test2 = [13, 35, 67, 89, 90, 100]
test3 = [34, 678, 902, 783, 100, 345]

# Call the function and print the results
print("Average 1:", calculate_average(test1))
print("Average 2:", calculate_average(test2))
print("Average 3:", calculate_average(test3))


Task 4: Loops with enumerate()

# A tuple of city names (similar to a list but cannot be changed)
cities = ("San Fransisco", "Los Angeles", "Seattle", "Manhattan", "New York")

# Loop through the tuple, getting both index and value
for index, city in enumerate(cities):
    # Print the position (index) and the city name
    print(f"Index {index}: {city}")



    Task 5: Combining Everything

# A list of exam scores (I made these up)
exam_scores = [85.5, 80.4, 78.5, 90, 96, 100]

# Print all scores
print("Scores:", exam_scores)

# Calculate the average using my function
avg = calculate_average(exam_scores)

# Print the average rounded to 1 decimal place
print(f"Class average: {avg:.1f}")

# Check if average is 80 or higher
if avg >= 80:
    print("Result: PASS")  # If yes, print PASS
else:
    print("Result: FAIL")  # If no, print FAIL

    Combination of everything learned with Random Function

# Import the random module so I can generate random numbers
import random
# Create a list of 10 random numbers between 0 and 100
exam_scores=[random.randint(0,100) for _ in range(10)]
# Print all scores
print("Scores:", exam_scores)
# Calculate the average using my function
avg=calculate_average(exam_scores)
# Print the average rounded to 1 decimal place
print(f"Class average: {avg:.1f}")
# Check if average is 80 or higher
if avg>=80:
  print("Result: PASS")
else:
  print("Result: FAIL")
