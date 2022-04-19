import numpy as np
import csv

# state is (position, velocity) in some units
# measurement is (position)

dt = 1e-3 # s
times = np.arange(0, 1, dt)
A = np.array([[1, dt], [0, 1]])
Q = np.array([[0, 0], [0, 1]])
H = np.array([[1, 0]])
R = np.array([[2.25]])

# to prove this works, let's generate truth states from this underlying distribution
# I might have this data in a CSV precomputed

truth_states = np.zeros((times.size, 2))
for i in range(1, times.size):
    truth_states[i] = np.random.multivariate_normal(A.dot(truth_states[i-1]), Q)

truth_states = truth_states.T
# plt.plot(times, truth_states[0], label="position (m)")
# plt.plot(times, truth_states[1], label="velocity (m/s)")
# plt.legend()
# plt.title("this is forbidden knowledge")

measurements = H.dot(truth_states).flatten() + np.random.multivariate_normal([0], R, size=times.size).flatten()
with open('kinematics.csv', 'w') as csvfile:
    writer = csv.writer(csvfile)
    writer.writerow(["t", "truth_p", "truth_v", "measurements"])
    for i in range(times.size):
        writer.writerow([np.round(dt*i, 3), truth_states[0][i], truth_states[1][i], measurements[i]])
