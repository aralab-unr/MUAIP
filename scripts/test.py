#!/usr/bin/env python3

import rospy
from visualization_msgs.msg import Marker
from geometry_msgs.msg import Point
from utils import generate_uniform_grid, frostum_vertices

def publish_markers(unique_points):
    # Initialize ROS node
    rospy.init_node('marker_publisher', anonymous=True)

    # Publisher for Marker
    marker_pub = rospy.Publisher('visualization_marker', Marker, queue_size=10)

    # Create Marker message
    marker = Marker()
    marker.header.frame_id = "odom"  # Adjust frame_id as needed
    marker.header.stamp = rospy.Time.now()
    marker.ns = "basic_shapes"
    marker.id = 0
    marker.type = Marker.POINTS  # Use POINTS instead of SPHERE_LIST
    marker.action = Marker.ADD
    marker.pose.orientation.w = 1.0
    marker.scale.x = 0.01  # Adjust size of points as needed
    marker.scale.y = 0.01
    marker.scale.z = 0.01
    marker.color.r = 1.0
    marker.color.a = 1.0

    # Convert points to Marker format
    for point in unique_points:
        p = Point()
        p.x = point[0]
        p.y = point[1]
        p.z = point[2]
        marker.points.append(p)

    # Publish Marker
    rate = rospy.Rate(10)  # 10 Hz
    while not rospy.is_shutdown():
        marker.header.stamp = rospy.Time.now()
        marker_pub.publish(marker)
        rate.sleep()

def publish_frustum_marker(pub_marker, vertices):
    """
    Publish a frustum marker to RViz.

    Parameters:
    pub_marker (rospy.Publisher): Publisher object for publishing Marker messages.
    vertices (ndarray): Array of shape (8, 3) containing the vertices of the frustum pyramid.
    """
    # Create Marker message
    marker = Marker()
    marker.header.frame_id = "oak"  # Replace with your frame_id
    marker.type = Marker.LINE_LIST
    marker.action = Marker.ADD
    marker.scale.x = 0.01  # Line width
    marker.color.r = 1.0
    marker.color.g = 0.0
    marker.color.b = 0.0
    marker.color.a = 1.0
    
    # Add vertices to Marker
    for i in range(4):
        p1 = Point()
        p1.x, p1.y, p1.z = vertices[i % 4]
        p2 = Point()
        p2.x, p2.y, p2.z = vertices[(i + 1) % 4]
        marker.points.append(p1)
        marker.points.append(p2)
        
        p3 = Point()
        p3.x, p3.y, p3.z = vertices[i + 4]
        marker.points.append(p1)
        marker.points.append(p3)
        
        p4 = Point()
        p4.x, p4.y, p4.z = vertices[(i + 1) % 4 + 4]
        marker.points.append(p2)
        marker.points.append(p4)
        
        marker.points.append(p3)
        marker.points.append(p4)
    
    # Publish Marker
    pub_marker.publish(marker)
    
    # rospy.loginfo("Frustum marker published.")

    marker = Marker()
    marker.header.frame_id = "oak"  # Replace with your frame_id
    marker.type = Marker.TRIANGLE_LIST
    marker.action = Marker.ADD
    marker.scale.x = 1.0
    marker.scale.y = 1.0
    marker.scale.z = 1.0
    marker.color.r = 0.0
    marker.color.g = 1.0
    marker.color.b = 0.0
    marker.color.a = 0.5  # Adjust transparency as needed

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
    pub_marker.publish(marker)

def frustum_visualizer_node():
    rospy.init_node('frustum_visualizer_node')
    pub_marker = rospy.Publisher('frustum_marker', Marker, queue_size=10)
    
    rate = rospy.Rate(1)  # 1 Hz
    while not rospy.is_shutdown():
        # Parameters for frustum
        h_fov = 62
        v_fov = 37.5
        near = 0.72
        far = 5.0
        vertices = frostum_vertices(near, far,h_fov, v_fov)
        
        # Publish frustum marker
        publish_frustum_marker(pub_marker, vertices)
        
        rate.sleep()

if __name__ == '__main__':
    try:
        # grid_test
        w = 1.2
        l = 4
        h = 0.7
        scale = 11
        offset_y = -w/2 +0.013
        unique_points = generate_uniform_grid(w, l, h, offset_y=offset_y, scale=scale)
        publish_markers(unique_points)

        # frustum_visualizer_node()
    except rospy.ROSInterruptException:
        pass