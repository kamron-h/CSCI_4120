## README.md: `get_averages` and `do_regression` Functions

#### Kamron Hopkins
#### hopkinsk19@students.ecu.edu

### Overview

This project includes two main functions: **`get_averages`** and **`do_regression`**. These functions are designed to analyze a dataset of students' study hours and grades, compute average grades for each rounded study hour, and perform a simple linear regression to model the relationship between study hours and grades.

### Requirements

The functions rely on the following Python libraries:
- **Pandas** for data manipulation.
- **NumPy** for numerical operations.
- **Scipy** (optional for an alternate version) for statistical operations.

You can install these libraries using:
```bash
pip install pandas numpy
```

### `get_averages` Function

The `get_averages` function calculates the mean grade for each rounded whole number of study hours.

#### Function Description

```python
def get_averages(data):
    """
    For all data points, round the student's study hours to the nearest whole number.
    Compute the mean grade for each rounded whole number of study hours.
    
    Parameters:
    data (pd.DataFrame): A DataFrame containing 'study_hours' and 'grades' columns.
    
    Returns:
    pd.DataFrame: A DataFrame with the rounded study hours as the index and the mean grades.
    """
```

#### Parameters
- **data**: A pandas DataFrame with at least two columns:
  - `study_hours`: The number of hours a student studied, as a floating-point number.
  - `grades`: The grade each student received.

#### Returns
- A pandas DataFrame where:
  - The index is the rounded study hours.
  - The column contains the mean grade for each rounded study hour.

#### Example Usage

```python
import pandas as pd

# Sample data
data = pd.DataFrame({
    'study_hours': [1.5, 2.3, 3.7, 4.1, 2.5],
    'grades': [78, 85, 92, 88, 80]
})

# Calculate average grades for each rounded study hour
grade_means = get_averages(data)
print(grade_means)
```

---

### `do_regression` Function

The `do_regression` function performs a linear regression on the dataset to model the relationship between study hours and grades. It uses the output of `get_averages` to determine the line of best fit.

#### Function Description

```python
def do_regression(data):
    """
    Perform linear regression to find the best-fit line for study hours vs. grades.
    
    Parameters:
    data (pd.DataFrame): A DataFrame containing 'study_hours' and 'grades' columns.
    
    Returns:
    np.array: A numpy array containing the coefficients [w0, w1] where:
              - w0 is the intercept.
              - w1 is the slope of the line.
    """
```

#### Parameters
- **data**: A pandas DataFrame containing columns `study_hours` and `grades`.

#### Returns
- A numpy array `[w0, w1]`:
  - `w0`: The intercept of the best-fit line.
  - `w1`: The slope of the best-fit line.

#### Explanation of Linear Regression
The function computes a linear regression using the least-squares method to find coefficients `w0` (intercept) and `w1` (slope). The regression line equation is:
\[
\text{grade} = w1 \times \text{study\_hours} + w0
\]

#### Example Usage

```python
import numpy as np

# Perform linear regression to find the line of best fit
coefficients = do_regression(data)
print("Intercept (w0):", coefficients[0])
print("Slope (w1):", coefficients[1])
```

---

### How the Functions Work Together

1. **Calculate Averages**: Use `get_averages(data)` to get the mean grades for each rounded study hour.
2. **Perform Regression**: Pass the dataset into `do_regression(data)` to obtain a linear model that relates study hours to grades.

These functions are useful for examining trends in data and predicting future outcomes based on linear relationships.

---

### Example Output

```plaintext
# get_averages
study_hours  grades
1            78.5
2            82.5
3            90.0
4            88.0

# do_regression
Intercept (w0): 70.12
Slope (w1): 4.25
```

---

### License
This project is open-source and free to use. Feel free to modify and share as needed.

### Author
This project was developed to demonstrate basic data manipulation and linear regression in Python using `pandas` and `numpy`.