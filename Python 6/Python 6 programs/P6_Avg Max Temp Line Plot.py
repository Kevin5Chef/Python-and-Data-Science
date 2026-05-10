import matplotlib.pyplot as plt
print("SY-5, Kevin Victor, Roll No.-30")

month=["Jan","Feb","March","April","May"]
avg_max_temp=[30,32,35,39,37]

plt.plot(month, avg_max_temp,
         color='red',
         marker='o',
         label='Avg Max Temperature')

plt.title("Average Maximum Temperature (Jan-May)")
plt.xlabel("Month")
plt.ylabel("Temperature (°C)")
plt.legend()
# Grid for better readability
plt.grid(True)
plt.show()