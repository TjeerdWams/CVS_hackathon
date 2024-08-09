import cv2
import numpy as np
import matplotlib.pyplot as plt

from blob_detector import blob_detector
from path_planner.rrt_star_dubins import RRTStarDubins
from path_planner.rrt_star import RRTStar
import matplotlib.pyplot as plt
from utils.plot import plot_arrow

show_animation = True

# Specify the path to a specific image
# image_path = '/home/cvs2024l8/Hackathon/Data/bunker_data/Bunker_lights_on/frame_0000.jpg'
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
        y = 720 - y
        print(f'Size of blob {count+1}: {round(size)}')
        print(f'Location of blob {count+1}: {round(x), round(y)}')

        # Append the (x, y, size) tuple to the obstacles list
        obstacle = (x, y, size/2)
        obstacles.append(obstacle)

        print(f'obstacles: {obstacles}')

        


#############################

print("Start rrt star with dubins planning")

# ====Search Path with RRT====
obstacleList = obstacles  # [x,y,size(radius)]

# Set Initial parameters
start = [0.0, 0.0, np.deg2rad(0.0)]
goal = [1000.0, 600.0, np.deg2rad(0.0)]
rand_area=[-2.0, 1000.0]

# rrtstar_dubins = RRTStarDubins(start, goal, rand_area, obstacle_list=obstacleList)
# path = rrtstar_dubins.planning(animation=show_animation)

rrt_star = RRTStar(
    start=start,
    goal=goal,
    rand_area=rand_area,
    obstacle_list=obstacleList,
    expand_dis=50,
    max_iter=200,
    robot_radius=0.8)
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


