import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from scipy.spatial import Delaunay
import numpy as np
from utils import *
from action import *
from observation import *

def transform_points(points, tf):
    """Transform multiple points using translation and Euler angles (roll, pitch, yaw) without a loop."""
    x, y, z, roll, pitch, yaw = tf
    # Convert roll, pitch, yaw to a rotation matrix
    Rx = np.array([[1, 0, 0], [0, np.cos(roll), -np.sin(roll)], [0, np.sin(roll), np.cos(roll)]])
    Ry = np.array([[np.cos(pitch), 0, np.sin(pitch)], [0, 1, 0], [-np.sin(pitch), 0, np.cos(pitch)]])
    Rz = np.array([[np.cos(yaw), -np.sin(yaw), 0], [np.sin(yaw), np.cos(yaw), 0], [0, 0, 1]])
    R = Rz @ Ry @ Rx

    # Create transformation matrix
    T = np.eye(4)
    T[0:3, 0:3] = R
    T[0:3, 3] = [x, y, z]

    # Convert points to homogeneous coordinates (n, 4)
    points_homogeneous = np.hstack((points, np.ones((points.shape[0], 1))))

    # Apply transformation to all points at once
    transformed_points_homogeneous = points_homogeneous @ T.T

    # Convert back to Cartesian coordinates (n, 3)
    transformed_points = transformed_points_homogeneous[:, :3]
    
    return transformed_points

def euclid_distance(point1, point2):
    """
    point1: x y z theta
    point2: x y z
    """
    return np.linalg.norm(point1[:2] - point2[:2])

def sensor(fov, tf0, tf1, tf2, robot_tf):
    """
    tf0: (x,y,z,r,p,y)
    """
    fov0 = transform_points(fov, tf0)
    fov0 = transform_points(fov0, robot_tf)
    fov1 = transform_points(fov, tf1)
    fov1 = transform_points(fov1, robot_tf)
    fov2 = transform_points(fov, tf2)
    fov2 = transform_points(fov2, robot_tf)    
    return fov0, fov1, fov2

class MultiSensor:
    def __init__(self, robot_id, h_fov, v_fov, near, far, tf0, tf1, tf2):
        fov = frostum_vertices(near, far,h_fov, v_fov)
        # transform to base frame
        self.fov0 = transform_points(fov, tf0)
        self.fov1 = transform_points(fov, tf1)
        self.fov2 = transform_points(fov, tf2)
        self._robot_id = robot_id
    
    def observe(self, robot_pose, state):
        """ robot_pose: x,y,z,theta """
        defposes = {}
        pose = np.array([robot_pose[0], robot_pose[1], 0, 0, 0, robot_pose[2]])
        # transform to global 
        fov0 = transform_points(self.fov0, pose)
        fov1 = transform_points(self.fov1, pose)
        fov2 = transform_points(self.fov2, pose)    
        frustums = np.stack([fov0, fov1,fov2])
        # check if point is inside
        # points_inside, _ = partition_points_by_frustums(frustums, points)
        for defid in state.object_states:
            if state.object_states[defid].objclass == "robot":
                defposes[defid] = ObjectObservation.NULL
                # print(defposes)
                continue
            pose = np.array(state.object_states[defid]["pose"])
            # print(np.array(state.object_states[defid]["pose"]))
            # print(point_in_frustum(frustums, pose))
            # don't care about frustum just distance
            if euclid_distance(robot_pose,pose) < 0.5:
                defposes[defid] = state.object_states[defid]["pose"]
                # print(defposes)
            else:
                defposes[defid] = ObjectObservation.NULL
        # return points_inside
        return MosOOObservation(defposes)
    
    def within_range(self, robot_pose, point):
        """ robot_pose: x,y,z,theta """
        pose = np.array([robot_pose[0], robot_pose[1], 0, 0, 0, robot_pose[2]])
        # transform to global 
        # import time
        # start_time = time.time()
        fov0 = transform_points(self.fov0, pose)
        fov1 = transform_points(self.fov1, pose)
        fov2 = transform_points(self.fov2, pose)   
        # end_time = time.time()
        # elapsed_time = end_time - start_time
        # print(f"Time taken: {elapsed_time} seconds") 
        frustums = np.stack([fov0, fov1,fov2])
        # check if point is inside
        # t = point_in_frustum(frustums, point)
        unique_points_2d_list = []
        for points in frustums:
            points_2d = points[:, :2]  # Remove the last column (z) to get 2D points
            unique_points_2d = np.unique(points_2d, axis=0)  # Find unique points
            unique_points_2d_list.append(unique_points_2d)

        # Combine all unique points from the three arrays
        combined_unique_2d = np.vstack(unique_points_2d_list)
        combined_unique_2d = np.unique(combined_unique_2d, axis=0)  # Ensure all points are unique

        # Create a Delaunay triangulation to connect the dots and cover the area
        tri = Delaunay(combined_unique_2d)
        t =  tri.find_simplex(np.array([point[0], point[1]])) >= 0
        # end_time = time.time()
        # elapsed_time = end_time - start_time
        # print(f"Time taken: {elapsed_time} seconds") 
        return t
    
    @property
    def robot_id(self):
        # id of the robot equipped with this sensor
        return self._robot_id

# # FOV param
# h_fov = 61
# v_fov = 49
# near = 0.2
# far = 3.0

# # tunnel states params
# w = 1.2
# l = 4.0
# h = 0.7
# scale = 5.5
# offset_x = 0.0
# offset_y = -0.6 + 0.013
# offset_z = 0.0

# # Robot states param
# robot_l = 1.0
# robot_offset_x = offset_x-1.0
# robot_offset_y = offset_y+0.287
# robot_offset_z = offset_z
# robot_y_scale = 7.0

# # rosrun tf tf_echo base_link camera_<camera number>_depth_frame
# tf0 = np.array([0.155, 0.028, 0.313, 0.0, 0.0, 0.0])
# tf1 = np.array([0.084, 0.143, 0.313, 0.0, 0.0, 0.8])
# tf2 = np.array([0.125, -0.104, 0.313,0.0, 0.0, -0.8])

# # robot states and pose
# robot_position_states = generate_robot_state(w/2, l+robot_l, offset_x=robot_offset_x, offset_y=robot_offset_y, offset_z=robot_offset_z, scale_x=scale, scale_y=robot_y_scale)
# random_index = np.random.choice(robot_position_states.shape[0])
# robot_position = robot_position_states[0]
# robot_yaw = 0.7854 # 45 degrees
# robot_pose = np.append(robot_position, [0.0, 0.0, robot_yaw])

# # generate FOV at origin
# vertices = frostum_vertices(near, far,h_fov, v_fov)
# # transform to the right place
# fov0, fov1, fov2 = sensor(vertices,tf0,tf1,tf2,robot_pose)
# # generate tunnel
# tunnel_states = generate_uniform_grid(w, l, h, offset_x=offset_x, offset_y=offset_y, offset_z=offset_z, scale=scale)

# frustums = np.stack([fov0, fov1,fov2])

# sensor = MultiSensor(-1, h_fov, v_fov, near, far, tf0, tf1, tf2)
# # print(frustums)
# points_inside, points_outside = partition_points_by_frustums(frustums, tunnel_states[:, :3])
# points_insides = sensor.observe(robot_pose, tunnel_states[:, :3])
# """ Plot for visualization purposes """
# fig = plt.figure()
# ax = fig.add_subplot(111, projection='3d')
# # Plot FOV
# # ax.scatter3D(vertices[:, 0], vertices[:, 1], vertices[:, 2], color='red', s=100)
# ax.scatter3D(fov0[:, 0], fov0[:, 1], fov0[:, 2], color='blue', s=10)
# ax.scatter3D(fov1[:, 0], fov1[:, 1], fov1[:, 2], color='purple', s=10)
# ax.scatter3D(fov2[:, 0], fov2[:, 1], fov2[:, 2], color='purple', s=10)

# ax.scatter3D(robot_position_states[:, 0], robot_position_states[:, 1], robot_position_states[:, 2], color='green', s=5)
# ax.scatter3D(robot_pose[0], robot_pose[1], robot_pose[2], color='red', s=5)

# ax.scatter3D(points_outside[:, 0], points_outside[:, 1], points_outside[:, 2], color='yellow', s=10)
# ax.scatter3D(points_insides[:, 0], points_insides[:, 1], points_insides[:, 2], color='orange', s=10)

# # Set labels and title
# ax.set_xlabel('X')
# ax.set_ylabel('Y')
# ax.set_zlabel('Z')
# ax.set_title('3D Plot of Vertices')
# ax.set_box_aspect([1,1,1])
# ax.set_xlim([-1.5, 4.5])
# ax.set_ylim([-1.5, 4.5])
# ax.set_zlim([-1.5, 4.5])
# plt.show()