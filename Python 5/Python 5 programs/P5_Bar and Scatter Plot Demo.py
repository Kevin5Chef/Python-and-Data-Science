import matplotlib.pyplot as plt
import numpy as np
print("SY-5, Kevin Victor, Roll No.-30")
#Bar graph
cat=["A","B","C","D","E"]
marks=[95,87,64,82,99]

plt.bar(cat,marks)
plt.title("Students and their respective Marks")
plt.xlabel("Students")
plt.ylabel("Marks obtained")
plt.show()

#Scatter plot program to show the scattering of random values

# Generate random data
x = np.random.rand(50)  # 50 random x-values between 0 and 1
y = np.random.rand(50)  # 50 random y-values between 0 and 1

plt.scatter(x, y, color='green', marker='o')
plt.title("Scatter Plot of Random Values")
plt.xlabel("X values")
plt.ylabel("Y values")
plt.show()