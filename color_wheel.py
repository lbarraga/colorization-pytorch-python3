import numpy as np
import matplotlib.pyplot as plt

def create_color_image(n, width=256, height=256):
    # Create an empty image
    image = np.zeros((height, width, 3), dtype=np.uint8)

    # Calculate the width of each stripe
    stripe_width = width // n

    # Generate n evenly spaced colors
    colors = plt.cm.get_cmap('hsv', n)

    for i in range(n):
        color = colors(i)[:3]  # Get the RGB values
        color = (np.array(color) * 255).astype(np.uint8)  # Convert to 0-255 range
        image[:, i*stripe_width:(i+1)*stripe_width, :] = color

    # Fill the remaining part with the last color
    if width % n != 0:
        image[:, n*stripe_width:, :] = color

    return image

# Parameters
n = 10  # Number of colors
width = 256
height = 256

# Create the image
image = create_color_image(n, width, height)

# Display the image
plt.imshow(image)
plt.axis('off')
plt.show()