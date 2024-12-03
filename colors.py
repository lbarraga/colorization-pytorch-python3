import matplotlib.pyplot as plt
from PIL import Image
from collections import Counter
import numpy as np


# Function to get distinct colors from an image
def get_distinct_colors(image_path):
    # Open the image
    img = Image.open(image_path)

    # Convert image to RGB if it's not already in that mode
    img = img.convert('RGB')

    # Get the colors from the image
    pixels = list(img.getdata())

    # Count the distinct colors
    color_counts = Counter(pixels)

    # List of distinct colors
    distinct_colors = list(color_counts.keys())

    return distinct_colors


# Function to create a circular (pie chart) plot of colors
def plot_color_circle(colors):
    # Number of colors
    num_colors = len(colors)

    # Create the plot
    fig, ax = plt.subplots(figsize=(6, 6))

    # Create a pie chart using the distinct colors
    wedges, texts = ax.pie([1] * num_colors, colors=[np.array(color) / 255 for color in colors],
                           wedgeprops={'edgecolor': 'black'}, startangle=90)

    # Equal aspect ratio ensures that pie is drawn as a circle.
    ax.axis('equal')

    # Show the plot
    plt.show()


# Path to your image in the Downloads folder
image_path = "/home/lukasbt/Downloads/trees.png"  # Change if necessary

# Get distinct colors
distinct_colors = get_distinct_colors(image_path)

# Create and show the circular color diagram
plot_color_circle(distinct_colors)
