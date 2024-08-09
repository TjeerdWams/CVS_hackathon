import cv2
import numpy as np
import matplotlib.pyplot as plt


def blob_detector(image):
    # Convert the image to grayscale
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    # Parameters for thresholding
    min_thresh = 90
    max_thresh = 255

    # Apply a binary threshold
    _, thresh = cv2.threshold(gray, min_thresh, max_thresh, cv2.THRESH_BINARY)

    # Find contours
    contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    # Draw contours on the original image
    contour_image = image.copy()
    cv2.drawContours(contour_image, contours, -1, (0, 255, 0), 2)

    # Initialize blob detector parameters
    params = cv2.SimpleBlobDetector_Params()
    params.filterByColor = True
    params.blobColor = 0  # Detect white blobs
    params.filterByArea = True
    params.minArea = 500    # Minimum area of blobs (adjust as needed)
    params.maxArea = 20000   # Maximum area of blobs (adjust as needed)
    params.filterByCircularity = False
    params.filterByInertia = False
    params.filterByConvexity = False

    detector = cv2.SimpleBlobDetector_create(params)

    # Detect blobs
    keypoints = detector.detect(thresh)
    keypoints = sorted(keypoints, key=lambda k: -k.size)  # Adjust the number of keypoints to keep

    # Draw detected blobs as red circles
    blob_image = cv2.drawKeypoints(image, keypoints, np.array([]), (0, 0, 255),
                                   cv2.DRAW_MATCHES_FLAGS_DRAW_RICH_KEYPOINTS)
    

    return thresh, contour_image, keypoints, blob_image

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

    for count, blob in enumerate(keypoints):
        size = blob.size
        x, y = blob.pt
        print(f'Size of blob {count+1}: {round(size)}')
        print(f'location of blob {count+1}: {round(x), round(y)}')


    # Display the results
    plt.figure(figsize=(24, 12))

    plt.subplot(1, 3, 1)
    plt.imshow(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
    plt.title('Original Image')
    plt.axis('off')

    plt.subplot(1, 3, 2)
    plt.imshow(thresh, cmap='gray')
    plt.title('Thresholded Image')
    plt.axis('off')

    plt.subplot(1, 3, 3)
    plt.imshow(cv2.cvtColor(contour_image, cv2.COLOR_BGR2RGB))
    plt.title('Contours')
    plt.axis('off')

    plt.figure(figsize=(12, 12))
    plt.imshow(cv2.cvtColor(blob_image, cv2.COLOR_BGR2RGB))
    plt.title('Detected Blobs')
    plt.axis('off')

    plt.show()

    # # Optionally display the result in a window (useful if running locally)
    # cv2.imshow('Blob Detection', blob_image)
    # if cv2.waitKey(0) & 0xFF == ord('q'):
    #     cv2.destroyAllWindows()

    # cv2.destroyAllWindows()
