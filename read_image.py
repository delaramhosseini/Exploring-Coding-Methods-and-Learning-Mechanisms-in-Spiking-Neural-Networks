import cv2
import os
import numpy as np
import matplotlib.pyplot as plt


def GetImage(
    image_address,
    n=10,
    prefix="./images/",
    show_image=True,
    flatten=True,
):
    # Read the grayscale image
    effective_address = prefix + image_address
    effective_address = effective_address.replace(" ", "\\")
    try:
        image = cv2.imread(effective_address, cv2.IMREAD_GRAYSCALE)
        if image is None:
            raise ValueError(f"Wrong effective_address: {effective_address}")

        # Resize the image to a square size n*n
        resized_image = cv2.resize(image, (n, n))

        # Display the resized image
        if show_image:
            plt.imshow(image, cmap="gray")
            plt.axis("off")  # Remove axis
            plt.show()

        return list(np.array(resized_image).reshape(-1)) if flatten else resized_image

    except Exception as e:
        pass
