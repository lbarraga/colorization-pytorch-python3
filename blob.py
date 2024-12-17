import cv2
import numpy as np


def blob_detection(input_image_path, output_image_path):
    # Load the image
    image = cv2.imread(input_image_path, cv2.IMREAD_GRAYSCALE)
    if image is None:
        print("Error: Unable to load the image.")
        return

    # Set up the SimpleBlobDetector parameters
    params = cv2.SimpleBlobDetector_Params()

    # Filter by area
    params.filterByArea = True
    params.minArea = 100

    # Filter by circularity
    params.filterByCircularity = True
    params.minCircularity = 0.1

    # Filter by convexity
    params.filterByConvexity = True
    params.minConvexity = 0.87

    # Filter by inertia
    params.filterByInertia = True
    params.minInertiaRatio = 0.01

    # Create a blob detector with the parameters
    detector = cv2.SimpleBlobDetector_create(params)

    # Detect blobs in the image
    keypoints = detector.detect(image)

    # Draw detected blobs as red circles
    # cv2.DRAW_MATCHES_FLAGS_DRAW_RICH_KEYPOINTS ensures the size of the circle corresponds to the size of the blob
    output_image = cv2.drawKeypoints(image, keypoints, np.array([]), (0, 0, 255),
                                     cv2.DRAW_MATCHES_FLAGS_DRAW_RICH_KEYPOINTS)

    # Write the output image
    cv2.imwrite(output_image_path, output_image)

    print(f"Blob detection completed. Output saved to {output_image_path}")


# Example usage
input_image_path = "img.png"  # Replace with your input image path
output_image_path = "output.jpg"  # Replace with your desired output image path
blob_detection(input_image_path, output_image_path)
