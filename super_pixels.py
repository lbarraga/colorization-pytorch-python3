import cv2

def generate_superpixels(input_image_path, compactness, region_size, output_image_path="output_superpixels4.png"):
    """
    Generate superpixels on an input image using OpenCV's SLIC algorithm.

    Args:
        input_image_path (str): Path to the input image.
        num_superpixels (int): Approximate number of superpixels.
        compactness (float): Balances color proximity and space proximity. Higher values mean more compact clusters.
        output_image_path (str): Path to save the output image with superpixel boundaries.
    """
    # Load the input image
    image = cv2.imread(input_image_path)
    if image is None:
        raise ValueError(f"Image at path '{input_image_path}' could not be loaded.")

    # Convert to Lab color space for better segmentation
    lab_image = cv2.cvtColor(image, cv2.COLOR_BGR2Lab)

    # Create SLIC superpixel object
    slic = cv2.ximgproc.createSuperpixelSLIC(
        lab_image,
        algorithm=cv2.ximgproc.SLIC,
        region_size=region_size,
        ruler=compactness,
    )

    # Run the SLIC algorithm
    slic.iterate(10)  # Number of iterations

    # Get the labels and draw the superpixel boundaries
    mask = slic.getLabelContourMask(thick_line=True)
    labels = slic.getLabels()
    segmented_image = image.copy()
    segmented_image[mask == 255] = [0, 0, 255]  # Red boundaries for superpixels

    # Optionally save the output image
    cv2.imwrite(output_image_path, segmented_image)

    print(f"Superpixel segmentation completed. Output saved to '{output_image_path}'.")

# Example usage
input_image_path = "/home/lukasbt/Pictures/n02085620_242_gray.JPEG"  # Replace with your image path
generate_superpixels(
    input_image_path=input_image_path,
    compactness=60,
    region_size=30,
)
