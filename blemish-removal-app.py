#######################################################
# HOWTO USE:
#  1. Manual mode: select a circle with the left mouse button,
#     then click on the spot from where to copy.
#  2. Automatic mode: select a circle with the left mouse button,
#     then press the 'c' key.
#  3. Save modified image: press the 's' key.
#  4. Undo (infinite number of steps): press the 'u' key.
#  5. Quit: press the 'q' key or Esc.
#######################################################

import cv2
import math
import numpy as np

# ----------------------------
# Global Configuration
# ----------------------------
filename = "blemish.webp"
window_name = "blemish removal"

# Flags and state variables for mouse interaction
mouse_pressed = False
area_selected = False

# Coordinates and radius for the selected area
start_x, start_y, radius = -1, -1, 0

# Load the image in color mode
image = cv2.imread(filename, cv2.IMREAD_COLOR)
# Copy of the image for displaying temporary frames
image_for_frame = image.copy()
# Stack for undo functionality
undo_stack = []


# --------------------------------------------------
# Function: find_best_match_area
# Description: Searches around the blemish area for a patch that best
# matches the target area for seamless cloning.
# Note: Detailed comments provided on color processing:
#       - Converting from BGR to grayscale for gradient calculation.
#       - Using color differences computed on the BGR channels.
# --------------------------------------------------
def find_best_match_area(x, y, radius):
    # Expand the analyzed region by a factor of 1.5 around the blemish.
    analyze_radius = int(radius * 1.5)
    # Ensure that the patch does not exceed the image boundaries.
    safe_radius = min(
        analyze_radius, x, y, image.shape[1] - x - 1, image.shape[0] - y - 1
    )
    # Extract the blemish area patch.
    blemish_area = image[
        y - safe_radius : y + safe_radius, x - safe_radius : x + safe_radius
    ]
    # Create a mask over the patch. Initially, the mask is filled with ones.
    # Then we draw a black circle (zero values) to cover the blemish.
    mask = np.ones((2 * safe_radius, 2 * safe_radius), dtype=np.uint8)
    cv2.circle(mask, (safe_radius, safe_radius), safe_radius, (0, 0, 0), -1)
    # Expand the mask to three channels for color operations.
    mask_3ch = np.stack([mask, mask, mask], axis=2)

    # Convert the blemish area to grayscale. This is needed for the Sobel operator.
    # Color processing: conversion from BGR to Gray reduces color detail.
    blemish_gray = cv2.cvtColor(blemish_area, cv2.COLOR_BGR2GRAY)
    # Calculate gradients in the x and y directions.
    blemish_gx = cv2.Sobel(blemish_gray, cv2.CV_32F, 1, 0, ksize=3) * mask
    blemish_gy = cv2.Sobel(blemish_gray, cv2.CV_32F, 0, 1, ksize=3) * mask

    # Set the search radius for candidate patches.
    search_radius = 2 * safe_radius
    best_score = float("inf")
    best_match_x, best_match_y = -1, -1
    num_points = 16  # Number of points (angles) to consider around the blemish.

    # Iterate in two radial steps (multipliers 1 and 2) to cover nearby patches.
    for add_point in range(2):
        add_point_mult = add_point + 1
        for i in range(num_points * add_point_mult):
            angle = 2 * np.pi * i / num_points
            # Compute candidate coordinates based on the angle and multiplier.
            search_x = int(x + search_radius * np.cos(angle) * add_point_mult)
            search_y = int(y + search_radius * np.sin(angle) * add_point_mult)

            # Check boundaries to ensure the candidate patch is fully inside the image.
            if (
                search_x - safe_radius < 0
                or search_y - safe_radius < 0
                or search_x + safe_radius >= image.shape[1]
                or search_y + safe_radius >= image.shape[0]
            ):
                continue

            # Extract the candidate patch.
            compare_area = image[
                search_y - safe_radius : search_y + safe_radius,
                search_x - safe_radius : search_x + safe_radius,
            ]
            # Convert the candidate patch to grayscale for gradient calculation.
            # Color processing: despite working with colors later, gradients are computed on grayscale.
            compare_gray = cv2.cvtColor(compare_area, cv2.COLOR_BGR2GRAY)
            compare_gx = cv2.Sobel(compare_gray, cv2.CV_32F, 1, 0, ksize=3) * mask
            compare_gy = cv2.Sobel(compare_gray, cv2.CV_32F, 0, 1, ksize=3) * mask

            # Calculate the absolute color difference.
            # Color processing: using cv2.absdiff on BGR images and applying the mask on each channel.
            color_diff = np.sum(
                cv2.absdiff(blemish_area, compare_area) * mask_3ch
            ) / np.sum(mask)
            gx_diff = np.sum(np.abs(blemish_gx - compare_gx)) / np.sum(mask)
            gy_diff = np.sum(np.abs(blemish_gy - compare_gy)) / np.sum(mask)

            # Combine the color and gradient differences as a weighted score.
            score = color_diff * 0.6 + gx_diff * 0.2 + gy_diff * 0.2

            print(f"add_point: {add_point_mult}, angle: {angle}, score: {score}")
            # Update the best match if a lower score is found.
            if score < best_score:
                best_score = score
                best_match_x = search_x
                best_match_y = search_y

    print(f"best score: {best_score}")
    return best_match_x, best_match_y, best_score


# --------------------------------------------------
# Function: heal_spot
# Description: Repairs the blemish by copying a patch from a source area into the blemish area
# using seamless cloning. Provides visual feedback and employs undo functionality.
# --------------------------------------------------
def heal_spot(copy_from_x, copy_from_y):
    global start_x, start_y, radius
    # Ensure that the safe radius is within image bounds for both the source patch and target area.
    safe_radius = min(
        radius,
        min(
            copy_from_x,
            copy_from_y,
            image.shape[1] - copy_from_x,
            image.shape[0] - copy_from_y,
            start_x,
            start_y,
            image.shape[1] - start_x,
            image.shape[0] - start_y,
        ),
    )
    # Extract the patch from the source area.
    src_patch = image[
        copy_from_y - safe_radius : copy_from_y + safe_radius,
        copy_from_x - safe_radius : copy_from_x + safe_radius,
    ]
    # Create a mask for the source patch.
    # Color processing: the mask is in full color (BGR) with white (255,255,255)
    # representing the region to be cloned.
    src_mask = np.zeros(src_patch.shape, src_patch.dtype)
    cv2.circle(src_mask, (safe_radius, safe_radius), safe_radius, (255, 255, 255), -1)

    # Save the current state of the image to allow undos.
    undo_stack.append(image.copy())

    # Provide visual feedback by drawing a temporary magenta circle.
    temp_image = image.copy()
    cv2.circle(
        temp_image, (copy_from_x, copy_from_y), radius, (255, 0, 255), 3
    )  # Magenta (BGR: 255, 0, 255)
    cv2.imshow(window_name, temp_image)
    cv2.waitKey(500)

    # Perform seamless cloning to blend the source patch into the blemish area.
    # Color processing: seamlessClone blends the colors of the source and destination.
    image[:] = cv2.seamlessClone(
        src_patch, image, src_mask, (start_x, start_y), cv2.NORMAL_CLONE
    )
    image_for_frame[:] = image


# --------------------------------------------------
# Function: auto_heal_spot
# Description: Automatically searches for the best matching patch and
# initiates the healing operation.
# --------------------------------------------------
def auto_heal_spot(_, __):
    global start_x, start_y, radius
    # Find the best matching source area based on the patch similarity.
    best_src_x, best_src_y, score = find_best_match_area(start_x, start_y, radius)
    heal_spot(best_src_x, best_src_y)


# --------------------------------------------------
# Function: mouse_callback
# Description: Handles mouse events for selecting the blemish area and triggering the heal.
# --------------------------------------------------
def mouse_callback(event, x, y, flags, userdata):
    global mouse_pressed, area_selected
    global start_x, start_y, radius

    if event == cv2.EVENT_LBUTTONDOWN:
        # If an area is already selected, the click triggers the healing operation.
        if area_selected:
            heal_spot(x, y)
            area_selected = False
            return
        # Start a new selection for the blemish area.
        mouse_pressed = True
        start_x, start_y = x, y
        area_selected = False
        radius = 0
        image_for_frame[:] = image

    elif event == cv2.EVENT_MOUSEMOVE:
        if mouse_pressed:
            area_selected = True
            temp_image = image.copy()
            # Update the radius based on mouse movement.
            dx = start_x - x
            dy = start_y - y
            radius = int(math.ceil(math.hypot(dx, dy)))
            # Draw a green circle (BGR: 0, 255, 0) to show the current selection.
            cv2.circle(temp_image, (start_x, start_y), radius, (0, 255, 0), 1)
            image_for_frame[:] = temp_image
        else:
            return

    elif event == cv2.EVENT_LBUTTONUP:
        mouse_pressed = False

    else:
        return

    cv2.imshow(window_name, image_for_frame)


# ----------------------------
# Main Window and Event Loop Setup
# ----------------------------
cv2.namedWindow(window_name)
cv2.imshow(window_name, image)
cv2.setMouseCallback(window_name, mouse_callback)

while True:
    key = cv2.waitKey(20)
    # Undo: revert changes if 'u' is pressed and undo_stack is not empty.
    if key == ord("u") and len(undo_stack) > 0:
        restored = undo_stack.pop()
        image[:] = restored
        cv2.imshow(window_name, image)
    # Automatic healing: trigger if 'c' is pressed while an area is selected.
    if key == ord("c") and area_selected:
        area_selected = False
        auto_heal_spot(start_x, start_y)
        cv2.imshow(window_name, image)
    # Save the modified image when 's' is pressed.
    if key == ord("s"):
        cv2.imwrite("modified.jpg", image)
    # Quit if 'q' or Esc key is pressed.
    if key == 27 or key == ord("q"):
        break

cv2.destroyAllWindows()
