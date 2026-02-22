import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap

from src.lidar_generator import generate_lidar
from src.occupancy_grid import create_occupancy_grid
from src.clustering import detect_clusters
from src.thermal_processing import process_thermal
from src.fusion import apply_heat_to_grid
from src.planner import astar

print("Generating synthetic LiDAR...")
df = generate_lidar()

print("Creating occupancy grid...")
grid = create_occupancy_grid(df)

print("Detecting obstacle clusters...")
df = detect_clusters(df)

print("Generating synthetic thermal data...")
thermal = np.random.normal(28, 2, (200, 200))  # room temperature

# Add fire blobs manually
fire_spots = [(120,120), (80,40), (150,60), (60,150), (40,60)]

for (x,y) in fire_spots:
    for i in range(-10, 10):
        for j in range(-10, 10):
            if i**2 + j**2 <= 100:  # circular fire
                gx = x + i
                gy = y + j
                if 0 <= gx < 200 and 0 <= gy < 200:
                    thermal[gx][gy] = 95  # fire temperature

heat_centers = process_thermal(thermal)
print("Fusing thermal with LiDAR grid...")
grid = apply_heat_to_grid(grid, heat_centers)

start = (10, 10)
goal = (150, 150)

print("Planning safest path...")
path = astar(grid, start, goal)

print("Path length:", len(path))

# Visualization
plt.figure(figsize=(8,8))

cmap = ListedColormap(["white", "black", "red"])
plt.imshow(grid.T, origin='lower', cmap=cmap)

if path:
    xs = [p[0] for p in path]
    ys = [p[1] for p in path]
    plt.plot(xs, ys, color='blue', linewidth=2)

plt.scatter(start[0], start[1], color='green', s=100, label='Start')
plt.scatter(goal[0], goal[1], color='purple', s=100, label='Goal')

plt.legend()
plt.title("DRONY - Heat-Aware Autonomous Navigation")
plt.show()
