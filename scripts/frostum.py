import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from scipy.spatial import Delaunay
import numpy as np

def frostum_vertices(near, far, h_fov, v_fov, plot=False):
    """
    Calculate vertices of a frustum pyramid in 3D space based on given parameters.

    Parameters:
    near (float): Distance to the near plane of the frustum.
    far (float): Distance to the far plane of the frustum.
    h_fov (float): Horizontal field of view (in degrees).
    v_fov (float): Vertical field of view (in degrees).
    plot (bool, optional): Whether to plot the vertices in a 3D space. Default is False.

    Returns:
    ndarray: Array of shape (8, 3) containing the vertices of the frustum pyramid in local coordinates.
    """
    # convert from degree to rad
    h_fov_rad = np.radians(h_fov)
    v_fov_rad = np.radians(v_fov)
    
    # Calculate top and bottom near plane dimensions
    x_near = near 
    y_near = near *np.tan(h_fov_rad/2)
    z_near = near * np.tan(v_fov_rad / 2)

    x_far = far 
    y_far = far *np.tan(h_fov_rad/2)
    z_far = far * np.tan(v_fov_rad / 2)
    
    # Define the vertices of the frustum pyramid in local coordinates
    vertices = np.array([
        [x_near,  y_near, -z_near],     # Bottom left corner of near plane
        [x_near, -y_near, -z_near],     # Bottom right corner of near plane
        [x_near, -y_near,  z_near],     # Top right corner of near plane
        [x_near,  y_near,  z_near],     # Top left corner of near plane
        [x_far,  y_far, -z_far],        # Bottom left corner of far plane
        [x_far, -y_far, -z_far],        # Bottom right corner of far plane
        [x_far, -y_far,  z_far],        # Top right corner of far plane
        [x_far,  y_far,  z_far],        # Top left corner of far plane
    ])

    if plot:
        fig = plt.figure()
        ax = fig.add_subplot(111, projection='3d')
        
        # Plot vertices
        ax.scatter3D(vertices[:, 0], vertices[:, 1], vertices[:, 2], color='red', s=100)
        
        # Set labels and title
        ax.set_xlabel('X')
        ax.set_ylabel('Y')
        ax.set_zlabel('Z')
        ax.set_title('3D Plot of Vertices')
        
        plt.show()
    return vertices


def points_in_frustum(points, frustum_vertices):
    """
    Check if points are inside the convex hull defined by frustum vertices using Delaunay triangulation.

    Parameters:
    points (ndarray): Array of points to check, each row representing (x, y, z) coordinates.
    frustum_vertices (list): List of vertices defining the frustum in 3D space.

    Returns:
    ndarray: Boolean array indicating whether each point is inside or on the boundary of the frustum's convex hull.
    """
    # Convert frustum vertices to NumPy array
    frustum_vertices = np.array(frustum_vertices)
    
    # Create a Delaunay triangulation from the frustum vertices
    tri = Delaunay(frustum_vertices)
    
    # Check if points are inside the convex hull using the triangulation
    return tri.find_simplex(points) >= 0

def plot_frustum(vertices, point=None):
    """
    Plot the frustum pyramid and optionally a point in 3D space.

    Parameters:
    vertices (ndarray): Array of shape (8, 3) containing the vertices of the frustum pyramid.
    point (ndarray or None, optional): Coordinates of the point to plot. Default is None.
    """
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')
    
    # Plot frustum vertices
    ax.plot([vertices[0, 0], vertices[1, 0], vertices[2, 0], vertices[3, 0], vertices[0, 0]],
            [vertices[0, 1], vertices[1, 1], vertices[2, 1], vertices[3, 1], vertices[0, 1]],
            [vertices[0, 2], vertices[1, 2], vertices[2, 2], vertices[3, 2], vertices[0, 2]], 'b-')
    
    ax.plot([vertices[4, 0], vertices[5, 0], vertices[6, 0], vertices[7, 0], vertices[4, 0]],
            [vertices[4, 1], vertices[5, 1], vertices[6, 1], vertices[7, 1], vertices[4, 1]],
            [vertices[4, 2], vertices[5, 2], vertices[6, 2], vertices[7, 2], vertices[4, 2]], 'b-')
    
    for i in range(4):
        ax.plot([vertices[i, 0], vertices[i + 4, 0]],
                [vertices[i, 1], vertices[i + 4, 1]],
                [vertices[i, 2], vertices[i + 4, 2]], 'b-')
    
    # Plot the point if provided
    if point is not None:
        ax.scatter(point[0], point[1], point[2], color='red', s=10)
    
    # Set labels and title
    ax.set_xlabel('X')
    ax.set_ylabel('Y')
    ax.set_zlabel('Z')
    ax.set_title('3D Plot of Frustum and Point')
    
    # Equal scaling for all axes
    ax.set_box_aspect([1,1,1])
    
    plt.show()

# h_fov = 69
# v_fov = 55
# near = 0.5
# far = 6.0

# vertices = frostum_vertices(near, far,h_fov, v_fov, False)

# points = np.array([
#         [10.5, 0.5, 0.5],
#         [5, 0,0],
#         [0.7, 0.3, 0.9],
#     ])

# inside_mask = points_in_frustum(points, vertices)
# print(inside_mask)
# plot_frustum(vertices, points)