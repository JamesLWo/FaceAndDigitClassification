import matplotlib
import matplotlib.pyplot as plt
import numpy as np

percent = [10, 20, 30, 40, 50, 60, 70, 80, 90, 100]
k_d = [77, 84, 86, 86, 91, 92, 93, 93, 93, 93]
k_f
p_d
p_f
n_d
n_f 


matplotlib.rc('font', size=12)

plt.title("Accuracy: KNN Digits")
plt.xlabel("Training Data Size (%)")
plt.ylabel("Accuracy (%)")

plt.plot(percent, value, marker = 'o')
plt.show()