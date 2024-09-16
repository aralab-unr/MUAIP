#!/usr/bin/env python3

import rospy
from visualization_msgs.msg import Marker
from geometry_msgs.msg import *
from std_msgs.msg import ColorRGBA, Bool
from sensor_msgs.msg import Image, PointCloud2
from tf2_geometry_msgs.tf2_geometry_msgs import do_transform_point
from nav_msgs.msg import Odometry, Path
import tf.transformations as tft
from tf.transformations import quaternion_matrix
import tf2_ros
import message_filters
from cv_bridge import CvBridge, CvBridgeError
import ros_numpy
from ultralytics import YOLO
from utils import *
from belief import *
import cv2
from scipy.spatial import Delaunay, ConvexHull
class Sim:
    def __init__(self, unique_points, robot_state, center_line, vertices):
        """
        Initialize the MarkerPublisher.

        Parameters:
        topic_name (str): The topic name for the marker.
        frame_id (str): The frame ID for the marker.
        """
        # Initialize ROS node
        rospy.init_node('culvert_sim', anonymous=True)
        
        # Publishers
        self.culvert_points_pub = rospy.Publisher("culvert_points", Marker, queue_size=5)
        self.observations_pub = rospy.Publisher("observation", Marker, queue_size=5)
        self.culvert_pose_pub = rospy.Publisher("culvert_pose", PoseArray, queue_size=5)
        self.robot_points_pub = rospy.Publisher("robot_points", Marker, queue_size=5)
        self.target_pub = rospy.Publisher("target", Marker, queue_size=1)
        self.fov_pub = rospy.Publisher("fov", Marker, queue_size=5)
        self.cmd_pub = rospy.Publisher("cmd_vel", Twist, queue_size=5)
        self.path_pub = rospy.Publisher('robot_path', Path, queue_size=5)

        # Define subscribers for three image topics
        self.image_sub1 = message_filters.Subscriber('/camera_0/rgb/image_raw', Image)
        self.image_sub2 = message_filters.Subscriber('/camera_1/rgb/image_raw', Image)
        self.image_sub3 = message_filters.Subscriber('/camera_2/rgb/image_raw', Image)

        # Define subscribers for three point cloud topics
        self.pc_sub1 = message_filters.Subscriber('/camera_0/depth/points', PointCloud2)
        self.pc_sub2 = message_filters.Subscriber('/camera_1/depth/points', PointCloud2)
        self.pc_sub3 = message_filters.Subscriber('/camera_2/depth/points', PointCloud2)
        
        # Other subscribers
        self.odom_sub = rospy.Subscriber('/odom', Odometry, self.odomCb)

        # timer event
        self.timer = rospy.Timer(rospy.Duration(1.0/10.0), self.mainCb)

        # Synchronize all six topics
        self.ts = message_filters.ApproximateTimeSynchronizer(
            [self.image_sub1, self.image_sub2, self.image_sub3, 
             self.pc_sub1, self.pc_sub2, self.pc_sub3], 
            queue_size=10, slop=0.1)
        self.receive = False
        # Register callback function to be called when messages are synchronized
        self.ts.registerCallback(self.observationCb)
        self.bridge = CvBridge()

        # get only position (orientation is for future work) and add visited
        false_column = np.full((unique_points.shape[0], 1), False)
        self.culvert_points = np.hstack((unique_points[:, :3], false_column))        
        self.robot_state = robot_state
        false_column = np.full((center_line.shape[0], 1), False)
        self.center_line = np.hstack((center_line, false_column)) 
        self.frostum_fov = vertices
        """ input  """
        self.frames = np.array(["camera_0_depth_frame", "camera_1_depth_frame", "camera_2_depth_frame"])
        self.frames_id = np.array([0,1,2])
        self.img_debug = False
        self.global_frame = "odom"   
        self.delta = 0.25
        self.sigma = 0.5
        model_path = "/home/ara/catkin_ws/src/culvert_sim/model/all_cul.pt"
        self.TP = 0.81
        self.FOV_vol = 9.66
        self.d_threshold = 2.0
        self.w = -np.log(2) / 2
        self.gamma = 0.99
        """ arm inputs"""
        self.alpha = 0.95
        self.arm_l = 1.0
        """ go to goal input    """
        self.ang_tol = 0.1
        self.lin_tol = 0.05
        self.ang = 0.2
        self.lin = 0.05
        self.print_goal_action = False
        # 0 = coverage, 1 = goal, 2 = declare, 3 = exit
        self.ACTION = -1
        self.action_name = ["coverage", "move", "declare", "exit"]
        self.action_goal = None
        self.path_var = Path()
        self.path_var.header.frame_id = self.global_frame
        self.path_poses = []

        self.cmd_vel = Twist()
        self.tf_buffer = tf2_ros.Buffer()
        self.tf_listener = tf2_ros.TransformListener(self.tf_buffer)
        
        self.covariance = np.array([[self.sigma**2, 0, 0],
                       [0, self.sigma**2, 0],
                       [0, 0, self.sigma**2]])
        self.sample_b = None
        self.observations = None
        self.def_id = 0
        self.robot = None
        self.OOBelief = []
        self.reward = 0
        self.timestep = 1
        self.start = False
        # get new action when action is done
        self.new_action = True
        self.small_r = 2
        self.big_r = 1000
        print('Load model.......')
        self.model = YOLO(model_path)
        print('Model loaded')
        self.receive = True
        self.start = True

        print(f"getObservations initialized as: {self.receive}")
        rospy.wait_for_message("/camera_0/rgb/image_raw", Image)
        rospy.wait_for_message("/camera_0/depth/points", PointCloud2)

    def getObservation(self, img, pc, cam):
        try:
            rgb_np = self.bridge.imgmsg_to_cv2(img, "bgr8")
        except CvBridgeError as e:
            print(e)
            return None
        observations = []
        xyz_array = ros_numpy.point_cloud2.pointcloud2_to_xyz_array(pc,False)
        # Defect detection & localization 
        results = self.model(source=rgb_np, verbose=False)
        for box in results[0]: 
            loc = box.boxes.xywh.cpu()
            loc = loc.numpy().flatten().astype('int32')
            if box.boxes.conf.item() >= 0.5 and isDefect(loc, img.width, img.height):
                if self.img_debug:
                    cs = boundingbox(loc.flatten().astype('int32'))
                    # spalls
                    if box.boxes.cls.item() == 1.0:
                        color = (0,255,0)
                    # crack
                    elif box.boxes.cls.item() == 0.0:
                        color = (0,0,255)
                    cv2.rectangle(rgb_np, cs[0], cs[2], color=color, thickness=2)
                # detection
                r = loc[1]
                c = loc[0]
                # localization
                # print(xyz_array.shape, r,c)
                x_val, y_val, z_val = xyz_array[r,c]                
                # making sure it's not nan
                if all_real(x_val, y_val, z_val):
                    # transform to global frame
                    x, y, z = self.transform_point(x_val, y_val, z_val, pc.header.frame_id, self.global_frame)
                    # observations.append([x,y,z])
                    observations.append([x,y,z, box.boxes.cls.item(), box.boxes.conf.item()])
        if self.img_debug:
            cv2.imshow("Image window "+cam, rgb_np)
            cv2.waitKey(3)
        return np.array(observations)

    def observationCb(self, img0, img1, img2, pc0, pc1, pc2):
        # print(f'\rCurrent Value: {self.robot}', end='')
        # print(f"receive is: {self.receive}")
        if self.receive:
            observation0 = self.getObservation(img0, pc0, "1")
            observation1 = self.getObservation(img1, pc1, "2")
            observation2 = self.getObservation(img2, pc2, "3")
            # self.observations = combine_and_points(observation0, observation1, observation2)
            self.observations = merge_obs([observation0, observation1, observation2])
            # print("---")
            # print(observation0, observation1, observation2)
            # print(self.observations)
            # print("---")
            # marker = Marker()
            # marker.header.frame_id = self.global_frame
            # marker.header.stamp = rospy.Time.now()
            # marker.ns = "culvert_points"
            # marker.id = 10
            # marker.type = Marker.POINTS  # Use POINTS instead of SPHERE_LIST
            # marker.action = Marker.ADD
            # marker.pose.orientation.w = 1.0
            # marker.scale.x = 0.02  # Adjust size of points as needed

            # # Convert points to Marker format
            # for point in self.observations:
            #     p = Point()
            #     p.x, p.y, p.z = point[:3]
            #     marker.points.append(p)
            #     color = ColorRGBA()
            #     # spalls
            #     if point[-2] == 1.0:
            #         color.r, color.g, color.b, color.a = 211, 211, 211, 1.0
            #     # cracks
            #     else:
            #         color.r, color.g, color.b, color.a = 173, 216, 230, 1.0
            #     marker.colors.append(color)    
            # # self.receive = False    
            # self.observations_pub.publish(marker)

    def mainCb(self,event=None): 
        # self.action_goal = self.planner()
        # print("Action: ", self.ACTION, "|", self.action_goal)
        self.update_visited()
        # self.update_belief()
        self.publish_markers()
        return

        if not self.start:
            return
        else:
            self.update_path()

        # or for the loop 
        if self.new_action or self.ACTION == -1:
            self.action_goal = self.planner()
            self.print_goal_action = True
            self.new_action = True

        if self.new_action:    
            print("==========================")
            print("Timestep ", self.timestep)        
            print("Action: ", self.action_name[self.ACTION], "|", self.action_goal)
            self.timestep += 1

        if self.ACTION == 0 or self.ACTION == 1:
            # if reach
            if self.goToGoal(self.action_goal):
                if self.ACTION == 0:
                    # update center point to True
                    distances = np.linalg.norm(self.center_line[:, :3] - self.action_goal, axis=1)
                    closest_idx = np.argmin(distances)
                    self.center_line[closest_idx][3] = True
                if self.ACTION == 1:
                    # update center point to True
                    distances = np.linalg.norm(self.center_line[:, :3] - self.action_goal, axis=1)
                    self.center_line[distances <= 0.25, 3] = True
                # get reward
                self.reward += -self.small_r * (self.gamma **self.timestep )
                print("Reward: ", -self.small_r * (self.gamma **self.timestep ))
                print("Total Reward: ", self.reward)
                # update path
                self.update_path()
                # change counter for new action
                self.ACTION = -1
                self.new_action = True
            else:
                if self.print_goal_action:
                    print(f'\r {self.action_goal}')
                    self.print_goal_action = False
        elif self.ACTION == 2:
            while True:
                r = input("Please enter the reward: ")
                try:
                    # Try to convert the reward to an integer
                    number = int(r)
                    print(f"Reward entered: {number}")
                    self.reward += number * (self.gamma **self.timestep)
                    # mark as visited
                    not_visited_def = [def_ for def_ in self.OOBelief if not def_._visited]
                    if not not_visited_def:
                        print("Error, no unvisited poses")
                    """TODO: code relies too much on closest, need more robust """
                    distances = np.linalg.norm([pose._pose - self.robot[:3] for pose in not_visited_def], axis=1)
                    closest_idx = np.argmin(distances)
                    closest_pose = not_visited_def[closest_idx]
                    closest_pose.visited()
                    # if self.new_action:
                    print("Reward: ", number * (self.gamma **self.timestep ))
                    print("Total Reward: ", self.reward)
                    self.ACTION = -1
                    self.new_action = True
                    break  # Exit the loop once a valid integer is entered
                except ValueError:
                    # If conversion fails, prompt the user to try again
                    print("That's not a valid integer. Please try again.")
        elif self.ACTION == 3:
            # shutdown when exist
            print("Total Reward: ", self.reward)
            rospy.signal_shutdown("Task Done.")
        else:
            print("Unknown action: ", self.ACTION)

        if self.new_action:
            print("Observation: ", self.observations.shape if self.observations is not None else "None")
        # update
        self.update_visited()
        self.update_belief()
        if self.new_action:
            self.new_action = False
        # marker
        self.publish_markers()
    def planner(self):
        # belief is empty
        if len(self.OOBelief) < 1:
            # self.ACTION = 0
            return self.coverage()
        else:
            not_visited_def = [def_ for def_ in self.OOBelief if not def_._visited]
            # all beliefs are visited
            if not not_visited_def:
                # exit or coverage
                self.ACTION = 0
                return self.coverage()
            else:
                # sample belief                
                b = np.array(self.sample_beliefs(not_visited_def))
                # closest def sample
                distances_def = np.linalg.norm(b - self.robot[:3], axis=1)
                closest_def = b[np.argmin(distances_def)]
                distances_robot_to_neighbor = np.linalg.norm(self.robot_state - self.robot[:3], axis=1)
                # neighbors that are not itself
                neighbors = self.robot_state[(distances_robot_to_neighbor > 0.1) & (distances_robot_to_neighbor <=0.35)]
                # distances_neighbors_to_goal = np.linalg.norm(neighbors - closest_def, axis=1)
                distance_robot_to_goal = np.linalg.norm(self.robot[:2] - closest_def[:2])
                # no other closer
                # print(distance_robot_to_goal, "<", np.min(distances_neighbors_to_goal))
                if distance_robot_to_goal < self.alpha * self.arm_l:
                    self.ACTION = 2
                    return "declare"
                # heuristics
                g = np.linalg.norm(neighbors - self.robot[:3], axis=1)
                h = np.linalg.norm(neighbors - closest_def, axis=1)
                f = g+h
                best_neighbor_idx = np.argmin(f)
                self.ACTION = 1
                return neighbors[best_neighbor_idx]

    def sample_beliefs(self, not_visited_def):
        b = []
        for belif in not_visited_def:
            b.append(belif.sample_belief())
        return b        

    def goToGoal(self, goal):
        """ goal: np.array([x,y,z])"""
        self.visualizeTarget(goal[0], goal[1])
        lin_vel = 0
        ang_vel = 0  
        angleToGoal, distance = compute_angle_distance(goal, self.robot)
        """ if close enough """
        if distance < self.lin_tol:
            self.publish_velocity(lin_vel, ang_vel)
            print("Goal Reached")
            return True

        angle_diff = abs(shortest_angular_difference(self.robot[3], angleToGoal)) 
        # print(goal)
        # print(self.robot[3], "->", angleToGoal, distance)
        if angle_diff < self.ang_tol:
            lin_vel = self.lin
            ang_vel = 0
        else:
            lin_vel = 0
            ang_vel = turning(self.robot[3], angleToGoal, self.ang)
        self.publish_velocity(lin_vel, ang_vel)
        return False

    def update_belief(self, threshold=0.3):
        if self.observations is None:
            for def_ in self.OOBelief:
                # o^i = null
                o = 1 - self.TP
                bel = def_._belief
                bel[:, 3] *= o 
                total_b = np.sum(bel[:, 3])
                if total_b > 0:  # Avoid division by zero
                    bel[:, 3] /= total_b
            return
        if self.observations.size > 0:
            # print(self.observations)
            obs = self.observations[:, :3]
            # add new belief if there's a new one
            if len(self.OOBelief) == 0:
                to_add = self.observations
            else: 
                existing_poses = np.array([belief._pose for belief in self.OOBelief])
                distances = np.linalg.norm(obs[:, np.newaxis] - existing_poses, axis=2)
                # Check if the minimum distance to any existing belief is greater than the threshold
                min_distances = np.min(distances, axis=1)
                to_add = self.observations[min_distances >= threshold]
            for point in to_add:
                distances = np.linalg.norm(self.culvert_points[:, :3] - np.array([point[0], point[1], point[2]]), axis=1)
                closest_idx = np.argmin(distances)
                self.OOBelief.append(defect_belief(self.culvert_points[closest_idx][:3], self.def_id, point[3], self.culvert_points[:, :3]))
                self.def_id += 1
                
            for def_ in self.OOBelief:
                # Filter points with the same class using a boolean mask
                same_class_mask = self.observations[:, 3] == def_._cls
                same_class_points = self.observations[same_class_mask]
                if same_class_points.size == 0:
                    # o^i = null
                    o = 1 - self.TP
                    bel = def_._belief
                    bel[:, 3] *= o 
                    total_b = np.sum(bel[:, 3])
                    if total_b > 0:  # Avoid division by zero
                        bel[:, 3] /= total_b
                else:
                    # Compute the Euclidean distance to the reference point using vectorized operations
                    ref_coords = np.array(def_._pose)
                    distances = np.linalg.norm(same_class_points[:, :3] - ref_coords, axis=1)
                    # Find the index of the closest point
                    closest_idx = np.argmin(distances)
                    obs_pose = same_class_points[closest_idx][:3]
                    """ \omega calculation"""
                    distance = np.linalg.norm(self.robot[:3] - obs_pose)
                    omega = np.exp(w * (distance - self.d_threshold)) if distance >= self.d_threshold else 1
                    # |o - s_d| <= 1.5 \delta
                    if distances[closest_idx] <= 1.5 * self.delta:
                        """ F calculation   """
                        # We need the inverse of the covariance matrix for the PDF calculation
                        inv_covariance = np.linalg.inv(self.covariance)
                        # Compute the difference between each point and the "pose"
                        diff = def_._belief[:, :3] - obs_pose 
                        # Calculate the exponent term in the multivariate normal distribution
                        exponent = -0.5 * np.sum(diff @ inv_covariance * diff, axis=1)
                        # Calculate the normalization constant for the multivariate normal distribution
                        normalization_const = 1 / np.sqrt((2 * np.pi) ** 3 * np.linalg.det(self.covariance))
                        # Compute the PDF for all points at once
                        pdf_values = normalization_const * np.exp(exponent)
                        # Reshape the result to n x 1
                        pdf_values = pdf_values.reshape(-1, 1)
                        # element-wise multiplication
                        new_belief = def_._belief[:, 3] * pdf_values.flatten() * omega
                        # normalize
                        total_sum = np.sum(new_belief)
                        new_belief /= total_sum
                        # update
                        def_._belief[:, 3] = new_belief
                    # |o - s_d| > 1.5 \delta
                    else:
                        new_belief = def_._belief[:, 3] * omega / self.FOV_vol
                        # normalize
                        total_sum = np.sum(new_belief)
                        new_belief /= total_sum
                        # update
                        def_._belief[:, 3] = new_belief
        else: 
            for def_ in self.OOBelief:
                # o^i = null
                o = 1 - self.TP
                bel = def_._belief
                bel[:, 3] *= o 
                total_b = np.sum(bel[:, 3])
                if total_b > 0:  # Avoid division by zero
                    bel[:, 3] /= total_b
            return

    def coverage(self):
        # no more to cover
        if np.all(self.culvert_points[:, 3]):
            self.ACTION = 3
            return "exit"
        robot = self.robot[:3]
        unvisited_center = self.center_line[self.center_line[:, 3] == False, :3]
        distances = np.linalg.norm(unvisited_center - robot, axis=1)
        closest_idx = np.argmin(distances)
        goal =  unvisited_center[closest_idx]
        distances_robot_to_neighbor = np.linalg.norm(self.robot_state - robot, axis=1)
        # neighbors that are not itself
        neighbors = self.robot_state[(distances_robot_to_neighbor > 0.1) & (distances_robot_to_neighbor <=0.35)]
        # distances_neighbors_to_goal = np.linalg.norm(neighbors - goal[:3], axis=1)
        distance_robot_to_goal = np.linalg.norm(robot - goal[:3])
        # heuristics
        g = np.linalg.norm(neighbors - robot, axis=1)
        h = np.linalg.norm(neighbors - goal[:3], axis=1)
        f = g+h
        best_neighbor_idx = np.argmin(f)
        self.ACTION = 0
        return neighbors[best_neighbor_idx]
        
    def update_path(self):
        # update path
        pose_stamped = PoseStamped()
        pose_stamped.header.stamp = rospy.Time.now()
        pose_stamped.header.frame_id = self.global_frame
        pose_stamped.pose.position.x = self.robot[0]
        pose_stamped.pose.position.y = self.robot[1]
        pose_stamped.pose.position.z = self.robot[2]
        self.path_poses.append(pose_stamped)
        self.path_var.poses = self.path_poses
        self.path_var.header.stamp = rospy.Time.now()
        self.path_pub.publish(self.path_var)

    def publish_velocity(self, linear_x=0.0, angular_z=0.0):
        """Publishes velocity commands to cmd_vel."""
        self.cmd_vel.linear.x = linear_x
        self.cmd_vel.angular.z = angular_z
        self.cmd_pub.publish(self.cmd_vel)

    def update_visited(self):
        # lol = []
        for tf in self.frames:
            tf_ = self.tf_buffer.can_transform(self.global_frame, self.frames[0], rospy.Time(0), rospy.Duration(2.0))
            trans = self.tf_buffer.lookup_transform(self.global_frame, self.frames[0], rospy.Time(0), rospy.Duration(0.3))
            transformation_matrix = get_transformation_matrix(trans)
            transformed_points = transform_points_batch(self.frostum_fov, transformation_matrix)
            # lol.append(transformed_points)
            # update culvert coverage
            update_points_in_frustum_scipy(self.culvert_points, transformed_points)
            # update center point coverage
            update_points_in_frustum_scipy(self.center_line, transformed_points)
        # Calculate the convex hull for the combined points (intersection)
        # combined_points = np.vstack(lol)        
        # hull = ConvexHull(combined_points)
        # print(hull.volume)

    def odomCb(self, odom):
        x = odom.pose.pose.position.x
        y = odom.pose.pose.position.y
        z = odom.pose.pose.position.z
        # Extract the orientation (quaternion: x, y, z, w)
        orientation_q = odom.pose.pose.orientation
        quaternion = [orientation_q.x, orientation_q.y, orientation_q.z, orientation_q.w]
        # Convert quaternion to Euler angles (roll, pitch, yaw)
        _, _, yaw = tft.euler_from_quaternion(quaternion)

        # Store x, y, z, and yaw into the array
        self.robot = np.array([x, y, z, yaw, 1.0])

    def transform_point(self, x, y, z, child_frame, parent_frame):
        p = Point()
        p.x = x 
        p.y = y
        p.z = z                
        self.tf_buffer.can_transform(parent_frame, child_frame, rospy.Time(0), rospy.Duration(2.0))
        trans = self.tf_buffer.lookup_transform(parent_frame, child_frame,
                            rospy.get_rostime() - rospy.Duration(0.2),
                            rospy.Duration(0.1))
        # for bag
        # trans = tf_buffer.lookup_transform("odom", "zed_left_camera_frame",
        #                     header.stamp,
        #                     rospy.Duration(0.1))

        point_stamp = PointStamped()
        point_stamp.point = p
        point_stamp.header = parent_frame
        p_tf = do_transform_point(point_stamp, trans)

        return p_tf.point.x, p_tf.point.y, p_tf.point.z
    
    def publish_culvert_points_marker(self):
        """
        Create a Marker message with the given points.

        Parameters:
        unique_points (list of tuple): List of points to include in the marker.
        
        Returns:
        Marker: The constructed Marker message.
        """
        marker = Marker()
        marker.header.frame_id = self.global_frame
        marker.header.stamp = rospy.Time.now()
        marker.ns = "culvert_points"
        marker.id = 10
        marker.type = Marker.POINTS  # Use POINTS instead of SPHERE_LIST
        marker.action = Marker.ADD
        marker.pose.orientation.w = 1.0
        marker.scale.x = 0.02  # Adjust size of points as needed
        marker.color.r = 1.0
        marker.color.a = 1.0

        # Convert points to Marker format
        for point in self.culvert_points:
            p = Point()
            p.x, p.y, p.z = point[:3]
            marker.points.append(p)
            color = ColorRGBA()
            if point[-1]:
                color.r, color.g, color.b, color.a = 0.0, 1.0, 0.0, 1.0
            else:
                color.r, color.g, color.b, color.a = 1.0, 0.0, 0.0, 1.0
            marker.colors.append(color)        
        self.culvert_points_pub.publish(marker)
        # pose_array = PoseArray()
        # pose_array.header.frame_id = "odom"
        # pose_array.header.stamp = rospy.Time.now()

        # point = self.culvert_points[111]
        # pose = Pose()
        # pose.position.x = point[0]
        # pose.position.y = point[1]
        # pose.position.z = point[2]
        # pose.orientation.x = point[3]
        # pose.orientation.y = point[4]
        # pose.orientation.z = point[5]
        # pose.orientation.w = point[6]
        # pose_array.poses.append(pose)
        # self.culvert_pose_pub.publish(pose_array)
    
    def publish_robot_points_marker(self):
        """
        Create a Marker message with the given points.

        Parameters:
        unique_points (list of tuple): List of points to include in the marker.
        
        Returns:
        Marker: The constructed Marker message.
        """
        marker = Marker()
        marker.header.frame_id = "odom"
        marker.header.stamp = rospy.Time.now()
        marker.ns = "robot_points"
        marker.id = 100
        marker.type = Marker.POINTS  # Use POINTS instead of SPHERE_LIST
        marker.action = Marker.ADD
        marker.pose.orientation.w = 1.0
        marker.scale.x = 0.02  # Adjust size of points as needed
        marker.color.b = 1.0
        marker.color.a = 1.0
        # Convert points to Marker format
        for point in self.robot_state:
            p = Point()
            p.x = point[0]
            p.y = point[1]
            p.z = point[2]
            marker.points.append(p)
            color = ColorRGBA()
            color.r, color.g, color.b, color.a = 0.0, 0.0, 1.0, 1.0
            marker.colors.append(color)  
                
        for point in self.center_line:
            p = Point()
            p.x, p.y, p.z = point[:3]
            marker.points.append(p)
            color = ColorRGBA()
            if point[-1]:
                color.r, color.g, color.b, color.a = 0.0, 1.0, 0.0, 1.0
            else:
                color.r, color.g, color.b, color.a = 1.0, 0.0, 0.0, 1.0
            marker.colors.append(color)

        self.robot_points_pub.publish(marker)

    def publish_max_b_marker(self):
        marker = Marker()
        marker.header.frame_id = self.global_frame
        marker.header.stamp = rospy.Time.now()
        marker.ns = "culvert_points"
        marker.id = 15
        marker.type = Marker.POINTS  # Use POINTS instead of SPHERE_LIST
        marker.action = Marker.ADD
        marker.pose.orientation.w = 1.0
        marker.scale.x = 0.05  # Adjust size of points as needed

        # Convert points to Marker format
        for bef in self.OOBelief:
            b = bef.sample_belief(True)
            p = Point()
            p.x, p.y, p.z = b[:3]
            marker.points.append(p)
            color = ColorRGBA()
            # spalls
            if bef._cls == 1.0:
                color.r, color.g, color.b, color.a = 150, 150, 150, 1.0
            # cracks
            else:
                color.r, color.g, color.b, color.a = 173, 216, 230, 1.0
            marker.colors.append(color)       
        self.observations_pub.publish(marker)

    def publish_frustum_marker(self, frame_id, id):
        """
        Publish a frustum marker to RViz.

        Parameters:
        pub_marker (rospy.Publisher): Publisher object for publishing Marker messages.
        vertices (ndarray): Array of shape (8, 3) containing the vertices of the frustum pyramid.
        """
        # Create Marker message
        # marker = Marker()
        # marker.header.frame_id = "camera_0_link" 
        # marker.type = Marker.LINE_LIST
        # marker.action = Marker.ADD
        # marker.scale.x = 0.01  # Line width
        # marker.color.r = 1.0
        # marker.color.g = 0.0
        # marker.color.b = 0.0
        # marker.color.a = 1.0
        
        # # Add vertices to Marker
        # for i in range(4):
        #     p1 = Point()
        #     p1.x, p1.y, p1.z = vertices[i % 4]
        #     p2 = Point()
        #     p2.x, p2.y, p2.z = vertices[(i + 1) % 4]
        #     marker.points.append(p1)
        #     marker.points.append(p2)
            
        #     p3 = Point()
        #     p3.x, p3.y, p3.z = vertices[i + 4]
        #     marker.points.append(p1)
        #     marker.points.append(p3)
            
        #     p4 = Point()
        #     p4.x, p4.y, p4.z = vertices[(i + 1) % 4 + 4]
        #     marker.points.append(p2)
        #     marker.points.append(p4)
            
        #     marker.points.append(p3)
        #     marker.points.append(p4)
        
        # # Publish Marker
        # self.fov_pub.publish(marker)
        
        marker = Marker()
        marker.header.frame_id = frame_id  
        marker.type = Marker.TRIANGLE_LIST
        marker.action = Marker.ADD
        marker.id = id
        marker.scale.x = 1.0
        marker.scale.y = 1.0
        marker.scale.z = 1.0
        marker.color.r = 0.56
        marker.color.g = 0.93
        marker.color.b = 0.56
        marker.color.a = 0.4  # Adjust transparency as needed

        # Define the triangles for the frustum faces
        faces = [
            # Near plane
            (0, 1, 2),
            (0, 2, 3),
            # Far plane
            (4, 5, 6),
            (4, 6, 7),
            # Side faces
            (0, 1, 5),
            (0, 5, 4),
            (1, 2, 6),
            (1, 6, 5),
            (2, 3, 7),
            (2, 7, 6),
            (3, 0, 4),
            (3, 4, 7),
        ]
        
        # Add vertices to Marker as triangles
        for face in faces:
            for i in face:
                p = Point()
                p.x, p.y, p.z = vertices[i]
                marker.points.append(p)
        # Publish Marker
        self.fov_pub.publish(marker)

    def visualizeTarget(self, x, y):
        marker = Marker()
        marker.header.frame_id = self.global_frame  # Adjust according to your TF frames
        # marker.header.stamp = rospy.Time.now()
        marker.ns = "arrow_marker"
        marker.id = 10000
        marker.type = Marker.ARROW
        marker.action = Marker.ADD
        
        # Set the pose of the marker
        # The arrow points up from the point (x, y) to (x, y, 1)
        marker.points.append(Point(x, y, 0))  # Start point
        marker.points.append(Point(x, y, 0.3))  # End point - pointing straight up
        
        # Set the scale of the arrow
        marker.scale.x = 0.02  # Shaft diameter
        marker.scale.y = 0.05  # Head diameter
        marker.scale.z = 0.05  # Head length
        
        # Set the color of the marker
        marker.color.r = 0.0
        marker.color.g = 0.0
        marker.color.b = 1.0
        marker.color.a = 1.0  # Make sure to set the alpha to something non-zero!
        
        # Publish the Marker
        self.target_pub.publish(marker)

    def publish_markers(self):
        self.publish_culvert_points_marker()
        self.publish_robot_points_marker()
        self.publish_max_b_marker()
        for i in range(0,3):
            self.publish_frustum_marker(self.frames[i], self.frames_id[i])

def get_transformation_matrix(transform):
    # Extract translation and rotation (quaternion) from the TransformStamped message
    t = transform.transform.translation
    q = transform.transform.rotation
    # Create a 4x4 translation matrix
    translation_matrix = np.array([
        [1, 0, 0, t.x],
        [0, 1, 0, t.y],
        [0, 0, 1, t.z],
        [0, 0, 0, 1]
    ])
    # Create a 4x4 rotation matrix from the quaternion
    rotation_matrix = quaternion_matrix([q.x, q.y, q.z, q.w])
    # Combine the translation and rotation into one 4x4 transformation matrix
    transformation_matrix = np.dot(translation_matrix, rotation_matrix)
    return transformation_matrix

def transform_points_batch(points, transformation_matrix):
    # Add the 4th homogeneous coordinate (1) to all points
    points_homogeneous = np.hstack((points, np.ones((points.shape[0], 1))))

    # Apply the transformation matrix to all points
    transformed_points_homogeneous = transformation_matrix.dot(points_homogeneous.T).T

    # Convert back to 3D by removing the homogeneous coordinate
    transformed_points = transformed_points_homogeneous[:, :3]
    return transformed_points

if __name__ == '__main__':
    try:

        # Retrieve parameters from the ROS parameter server or use defaults
        w = rospy.get_param('~w', 3.27)
        l = rospy.get_param('~l', 2.4)
        h = rospy.get_param('~h', 1.0)
        scale = rospy.get_param('~scale', 3)
        offset_x = rospy.get_param('~offset_x', 0.0)
        offset_y = rospy.get_param('~offset_y', -1.5 + 0.013)
        offset_z = rospy.get_param('~offset_z', 0.0)

        rospy.loginfo(offset_x, offset_y, offset_z)

        h_fov = rospy.get_param('~h_fov', 61)
        v_fov = rospy.get_param('~v_fov', 49)
        near = rospy.get_param('~near', 0.2)
        far = rospy.get_param('~far', 3.0)

        robot_l = 1.0
        robot_offset_x = rospy.get_param('~robot_offset_x', offset_x-1.0)
        robot_offset_y = rospy.get_param('~robot_offset_y', offset_y+0.38)
        robot_offset_z = rospy.get_param('~robot_offset_y', offset_z)
        robot_x_scale = rospy.get_param('~robot_x_scale', 2.1)
        robot_y_scale = rospy.get_param('~robot_y_scale', 2.1)

        # generate culvert points & rgbd frostum
        unique_points = generate_uniform_grid(w, l, h, offset_x=offset_x, offset_y=offset_y, offset_z=offset_z, scale=scale)
        vertices = frostum_vertices(near, far,h_fov, v_fov)
        robot_state, center_line = generate_robot_state(w*2/3, l+robot_l, offset_x=robot_offset_x, offset_y=robot_offset_y, offset_z=robot_offset_z, scale_x=robot_x_scale, scale_y=robot_y_scale)

        # sim
        sim = Sim(unique_points, robot_state, center_line, vertices)
        # sim.publish_markers()
        rospy.spin()
    except rospy.ROSInterruptException:
        pass
