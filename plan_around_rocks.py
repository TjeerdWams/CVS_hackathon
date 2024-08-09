import cv2
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image

from blob_detector import blob_detector
from path_planner.rrt_star_dubins import RRTStarDubins
from path_planner.rrt_star import RRTStar
import matplotlib.pyplot as plt
from utils.plot import plot_arrow

show_animation = True

# Specify the path to a specific image
image_path = '/home/cvs2024l8/Pictures/BEV_blob.png'


# Load the image
image = cv2.imread(image_path)

# Check if the image was loaded successfully
if image is None:
    print(f"Failed to load image at {image_path}. Check the file path.")
else:
    # Detect blobs
    thresh, contour_image, keypoints, blob_image = blob_detector(image)

    print("Number of blobs detected: ", len(keypoints))

    obstacles = []
    for count, blob in enumerate(keypoints):
        size = blob.size
        x, y = blob.pt
        y = 668 - y
        print(f'Size of blob {count+1}: {round(size)}')
        print(f'Location of blob {count+1}: {round(x), round(y)}')

        # Append the (x, y, size) tuple to the obstacles list
        obstacle = (x, y, size/2)
        obstacles.append(obstacle)

        
#############################

print("Start rrt star planning")

# ====Search Path with RRT====
obstacleList = obstacles  # [x,y,size(radius)]

# Set Initial parameters
start = [250.0, 0.0, np.deg2rad(0.0)]
goal = [700.0, 600.0, np.deg2rad(0.0)]
rand_area=[0, 700.0]


rrt_star = RRTStar(
    start=start,
    goal=goal,
    rand_area=rand_area,
    obstacle_list=obstacleList,
    expand_dis=40,
    max_iter=150,
    robot_radius=10.0)
path = rrt_star.planning(animation=show_animation)

if path is None:
    print("Cannot find path")
else:
    print("found path!!")

    # Draw final path
    if show_animation:
        rrt_star.draw_graph()
        plt.plot([x for (x, y) in path], [y for (x, y) in path], 'r--')
        plt.grid(True)
        plt.show()

# Prepare your x, y coordinates
path_print = np.array(path)
x_coords = path_print[:,0]  # Example x-coordinates
y_coords = 668 - path_print[:,1]  # Example y-coordinates

# Create a figure and axis
fig, ax = plt.subplots()

# Display the image
ax.imshow(image)

# Plot the path
ax.plot(x_coords, y_coords, color='red', linewidth=2, marker='o', markersize=5)

# Display the plot
plt.show()
