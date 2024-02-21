import numpy as n
from matplotlib import pyplot as plt

study_hours = [2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12]
marks = [30, 35, 40, 47, 59, 68, 78, 85, 90, 93, 94]

# Convert data to NumPy arrays
x = n.array(study_hours)
y = n.array(marks)


def gradient_descent(x, y, learning_rate=0.01, iterations=100):
    n = len(x)
    slope = 0 # initial guess
    intercept = 0

    for _ in range(iterations):
        if iterations % 100 == 0:
            plt.plot(x, slope * x + intercept, 'r')
            plt.scatter(x, y)
            plt.show()
        d_slope = (-2 / n) * sum(x * (y - (slope * x + intercept))) # partial derivative of the slope
        d_intercept = (-2 / n) * sum(y - (slope * x + intercept))
        slope = slope - learning_rate * d_slope # update the slope
        intercept = intercept - learning_rate * d_intercept # update the intercept

    return slope, intercept

slope, intercept = gradient_descent(x,y)
print(slope, intercept)