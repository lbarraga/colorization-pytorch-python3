import matplotlib.pyplot as plt
import numpy as np

# Set up the figure and axis for a square image
fig, ax = plt.subplots(figsize=(6, 6))  # Make the figure square
ax.set_xlim(0, 10)
ax.set_ylim(0, 10)

# Colors
sky_color = "lightblue"
ground_color = "lightgreen"  # Light green color for the ground
sun_color = "yellow"
tree_trunk_color = "saddlebrown"
tree_leaves_color = "green"
cloud_color = "white"

# Draw the sky (occupying top 2/3 of the canvas)
ax.add_patch(plt.Rectangle((0, 4), 10, 6, color=sky_color, ec="none"))

# Draw the ground (light green, occupying the bottom 1/3 of the canvas)
ax.add_patch(plt.Rectangle((0, 0), 10, 4, color=ground_color, ec="none"))

# Draw the sun (in the upper right)
sun = plt.Circle((8, 8), 1, color=sun_color, ec="none")
ax.add_patch(sun)

# Function to draw a palm tree with rounder leaves
def draw_palm_tree(x, y):
    # Trunk of the tree (larger)
    ax.add_patch(plt.Rectangle((x - 0.2, y), 0.4, 2.5, color=tree_trunk_color, ec="none"))

    # Leaves of the tree (larger and more circular)
    leaf_positions = [(x, y + 2), (x + 0.8, y + 2.3), (x - 0.8, y + 2.3),
                      (x, y + 3), (x + 1, y + 2.6), (x - 1, y + 2.6)]

    # Increase the size of the leaves to make them rounder
    for pos in leaf_positions:
        ax.add_patch(plt.Circle(pos, 0.8, color=tree_leaves_color, ec="none"))

# Draw two palm trees
draw_palm_tree(2, 1)
draw_palm_tree(7, 1)

# Function to draw a cloud
def draw_cloud(x, y):
    cloud_sizes = [0.8, 1, 0.8]  # Sizes of the cloud parts
    cloud_positions = [(x, y), (x + 0.7, y), (x - 0.7, y),
                       (x + 0.3, y + 0.5), (x - 0.3, y + 0.5)]  # Positions of the cloud parts

    for i, pos in enumerate(cloud_positions):
        ax.add_patch(plt.Circle(pos, cloud_sizes[i % len(cloud_sizes)], color=cloud_color, ec="none"))

# Draw a cloud in the sky
draw_cloud(4, 8)

# Hide axes, remove extra space, and display the scene
ax.axis('off')  # Hide axes
plt.subplots_adjust(left=0, right=1, top=1, bottom=0)  # Remove the padding around the plot
plt.margins(0, 0)  # Remove any margins around the plot
plt.show()
