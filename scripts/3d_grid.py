import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

#belief update only points in fov

def generate_uniform_grid(w, l, h, offset_x=0.0, offset_y=0.0, offset_z=0.0, scale=10):
    # Generate uniform grid coordinates for left and right sides (XZ planes)
    num_points_x = max(round(w * scale), 10)
    num_points_z = max(round(h * scale), 10)

    x_left, z_left = np.meshgrid(np.linspace(0, w, num_points_x),
                                 np.linspace(0, h, num_points_z))
    x_right, z_right = np.meshgrid(np.linspace(0, w, num_points_x),
                                   np.linspace(0, h, num_points_z))

    # Generate uniform grid coordinates on the top face (XY plane)
    x_top, y_top = np.meshgrid(np.linspace(0, w, num_points_x),
                               np.linspace(0, l, round(l * scale)))

    left_points = np.column_stack((x_left.flatten(), np.zeros(num_points_x * num_points_z), z_left.flatten()))
    right_points = np.column_stack((x_right.flatten(), np.ones(num_points_x * num_points_z) * l, z_right.flatten()))
    top_points = np.column_stack((x_top.flatten(), y_top.flatten(), np.ones(num_points_x * round(l * scale)) * h))

    # Combine all points into a single array
    all_points = np.vstack((left_points, right_points, top_points))

    # Apply offsets
    all_points[:, 0] += offset_x
    all_points[:, 1] += offset_y
    all_points[:, 2] += offset_z

    # Remove duplicate points
    unique_points, unique_indices = np.unique(all_points, axis=0, return_index=True)

    return unique_points

# Dimensions of the rectangle
w = 1.4
l = 4
h = 0.8
scale = 10

# # Generate uniform grid point clouds for each face
# left_points, right_points, top_points = generate_uniform_grid(w, l, h, num_points_x, num_points_y, num_points_z)

# # Plotting all faces on the same graph with different colors
# fig = plt.figure(figsize=(8, 6))
# ax = fig.add_subplot(111, projection='3d')

# # Plot points for left side (blue)
# ax.scatter(left_points[:, 0], left_points[:, 1], left_points[:, 2], c='b', marker='o', label='Left Side')
# # Plot points for right side (red)
# ax.scatter(right_points[:, 0], right_points[:, 1], right_points[:, 2], c='r', marker='o', label='Right Side')
# # Plot points for top face (green)
# ax.scatter(top_points[:, 0], top_points[:, 1], top_points[:, 2], c='g', marker='o', label='Top Face')

# # Set labels and title
# ax.set_xlabel('X')
# ax.set_ylabel('Y')
# ax.set_zlabel('Z')
# ax.set_title('3D Uniform Grid Point Cloud for Rectangular Box')

# # Set equal scaling for axis
# max_range = np.array([left_points[:, 0].max()-left_points[:, 0].min(),
#                       left_points[:, 1].max()-left_points[:, 1].min(),
#                       left_points[:, 2].max()-left_points[:, 2].min()]).max() / 2.0
# mid_x = (left_points[:, 0].max()+left_points[:, 0].min()) * 0.5
# mid_y = (left_points[:, 1].max()+left_points[:, 1].min()) * 0.5
# mid_z = (left_points[:, 2].max()+left_points[:, 2].min()) * 0.5
# ax.set_xlim(mid_x - max_range, mid_x + max_range)
# ax.set_ylim(mid_y - max_range, mid_y + max_range)
# ax.set_zlim(mid_z - max_range, mid_z + max_range)

# # Add legend
# ax.legend()

# # Show plot
# plt.show()
unique_points = generate_uniform_grid(l, w, h, offset_x= -0.7, scale=scale)

# Plotting the unique points with equal scaling
fig = plt.figure(figsize=(8, 6))
ax = fig.add_subplot(111, projection='3d')

# Extract x, y, z coordinates from unique points
x = unique_points[:, 0]
y = unique_points[:, 1]
z = unique_points[:, 2]

# Plot points
ax.scatter(x, y, z, c='b', marker='o')

# Set labels and title
ax.set_xlabel('X')
ax.set_ylabel('Y')
ax.set_zlabel('Z')
ax.set_title('3D Uniform Grid Point Cloud with Equal Scaling')

# Set equal scaling for axis
ax.set_box_aspect([np.ptp(x), np.ptp(y), np.ptp(z)])

# Show plot
plt.show()