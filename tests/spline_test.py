import numpy as np
from scipy.interpolate import splprep, splev
import matplotlib.pyplot as plt


SEED = 8454 or np.random.randint(0, 10_000)
np.random.seed(SEED)
print(f'{SEED=}')

# Generate some example data points
N = 10
bounds = -1, 1
x, y = np.random.uniform(*bounds, size=[2, N])

# Use splprep to get the spline representation with k=4 for C1 continuity
tck, u = splprep([x, y], k=3, s=0, per=False)

# Evaluate the spline at a finer set of points
u_fine = np.linspace(0, 1, 1000)
xy_fine = splev(u_fine, tck)

# Plot the original points and the spline
plt.plot(x, y, 'o', label='Original Points')
plt.plot(xy_fine[0], xy_fine[1], label='Cubic Spline')

plt.legend()
plt.show()

# Generate some example data points
# N = 10
# bounds = -1, 1
# t = np.linspace(0, 1, N)
# # x = np.cos(t)
# # y = np.sin(t)
# x, y = np.random.uniform(*bounds, size=[2, N])


# # Use splprep to get the spline representation
# tck, u = splprep([x, y], k=2, per=True)

# # Evaluate the spline at a finer set of points
# u_fine = np.linspace(0, 1, 1000)
# xy_fine = splev(u_fine, tck)

# # Plot the original points and the spline
# plt.plot(x, y, 'o', label='Original Points')
# plt.plot(xy_fine[0], xy_fine[1], label='Spline')

# plt.legend()
# plt.show()