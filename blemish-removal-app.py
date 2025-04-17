#######################################################
# HOWTO USE:
# 1. manual mode: select circle with left mouse btn,
#    then click on spot from where to copy
# 2. automatic mode: select circle with left mouse btn,
#    then press 'c' key
# 3. save modified image - press 's' key
# 4. Undo (infinite number of steps) - press 'u' key
# 5. Quit - press 'q' or esc key
#######################################################

import cv2
import math
import numpy as np

filename = "blemish.webp"
window_name = "blemish removal"
mousePressed,  areaSelected = False, False
startX, startY, radius = -1, -1, 0

image = cv2.imread(filename, cv2.IMREAD_COLOR)
image_for_frame = image.copy()
undo_stack = []

def find_best_match_area(x, y, radius):
    # get bigger radius near blemish
    analyze_radius = int(radius * 1.5)
    # if radius is near image borders - make it smaller
    safe_radius = min(
        analyze_radius,
        x, y,
        image.shape[1] - x - 1,
        image.shape[0] - y - 1
    )
    # get enlarged area with blemish
    blemish_area = image[
                        y - safe_radius : y + safe_radius,
                        x - safe_radius:x + safe_radius
                    ]
    # prepare mask to cover blemish
    mask = np.ones((2 * safe_radius, 2 * safe_radius), dtype=np.uint8)
    cv2.circle(mask, (safe_radius, safe_radius), safe_radius, (0,0,0), -1)
    mask_3ch = np.stack([mask, mask, mask], axis=2)
    # blemish gradient
    blemish_gray = cv2.cvtColor(blemish_area, cv2.COLOR_BGR2GRAY)
    blemish_gx = cv2.Sobel(blemish_gray, cv2.CV_32F, 1, 0, ksize=3) * mask
    blemish_gy = cv2.Sobel(blemish_gray, cv2.CV_32F, 0, 1, ksize=3) * mask

    # we will look for src image around blemish
    search_radius = 2 * safe_radius
    # initialize score to compare patches
    best_score = float('inf')
    best_match_x, best_match_y = -1, -1
    # it will be 16 (or more) points around blemish to check
    num_points = 16
    for add_point in range(2):
        add_point_mult = add_point + 1
        for i in range(num_points * add_point_mult):
            angle = 2 * np.pi * i / num_points
            search_x = int(x + search_radius * np.cos(angle) * add_point_mult)
            search_y = int(y + search_radius * np.sin(angle) * add_point_mult)

            if (search_x - safe_radius < 0 or
                search_y - safe_radius < 0 or
                search_x + safe_radius >= image.shape[1] or
                search_y + safe_radius >= image.shape[0]):
                continue
            # get patch to compare
            compare_area = image[
                                search_y - safe_radius : search_y + safe_radius,
                                search_x - safe_radius : search_x + safe_radius
                           ]
            # calculate score
            compare_gray = cv2.cvtColor(compare_area, cv2.COLOR_BGR2GRAY)
            compare_gx = cv2.Sobel(compare_gray, cv2.CV_32F, 1, 0, ksize=3) * mask
            compare_gy = cv2.Sobel(compare_gray, cv2.CV_32F, 0, 1, ksize=3) * mask

            color_diff = np.sum(cv2.absdiff(blemish_area, compare_area) * mask_3ch) / np.sum(mask)
            gx_diff = np.sum(np.abs(blemish_gx - compare_gx)) / np.sum(mask)
            gy_diff = np.sum(np.abs(blemish_gy - compare_gy)) / np.sum(mask)

            score = color_diff * 0.6 + gx_diff * 0.2 + gy_diff * 0.2

            print(f"add_point: {add_point_mult}, angle: {angle}, score: {score}")
            # store best score (minimum)
            if score < best_score:
                best_score = score
                best_match_x = search_x
                best_match_y = search_y

    print(f"best score: {best_score}")
    return best_match_x, best_match_y, best_score

def heal_spot(x,y):
    global startX, startY, radius
    # if radius is near image borders - make it smaller
    safe_radius = min(radius,
        min(x, y,
            image.shape[1] - x,
            image.shape[0] - y,
            startX, startY,
            image.shape[1] - startX,
            image.shape[0] - startY))
    # get spot to copy from
    src_spot = image[
                    y - safe_radius : y + safe_radius,
                    x - safe_radius : x + safe_radius
               ]
    # prepare mask for source spot
    src_mask = np.zeros(src_spot.shape, src_spot.dtype)
    cv2.circle(src_mask, (safe_radius, safe_radius), safe_radius, (255,255,255), -1)
    # prepare undo
    undo_stack.append(image.copy())
    temp_image = image.copy()
    cv2.circle(temp_image, (x,y), radius, (255, 0, 255), 3)  # Червоне коло
    cv2.imshow(window_name, temp_image)
    cv2.waitKey(500)
    # heal blemish
    image[:] = cv2.seamlessClone(src_spot, image, src_mask, (startX, startY), cv2.NORMAL_CLONE)
    image_for_frame[:] = image

def auto_heal_spot(x, y):
    global startX, startY, radius
    # auto-search for spot to copy from
    best_src_x, best_src_y, score = find_best_match_area(startX, startY, radius)
    heal_spot(best_src_x, best_src_y)

def mouse_callback(action, x, y, flags, userdata):
    global mousePressed,  areaSelected
    global startX, startY, radius
    if action == cv2.EVENT_LBUTTONDOWN:
        if areaSelected:
            heal_spot(x,y)
            areaSelected = False
            return
        mousePressed, startX, startY, areaSelected, radius = True, x, y, False, 0
        image_for_frame[:] = image
    elif action == cv2.EVENT_MOUSEMOVE:
        if mousePressed:
            areaSelected = True
            temp_image = image.copy()
            dx = startX - x
            dy = startY - y
            radius = int(math.ceil(math.hypot(dx, dy)))
            cv2.circle(temp_image, (startX, startY), radius, (0,255,0), 1)
            image_for_frame[:] = temp_image
        else:
            return
    elif action == cv2.EVENT_LBUTTONUP:
        mousePressed = False
    else:
        return
    cv2.imshow(window_name, image_for_frame)

cv2.namedWindow(window_name)
cv2.imshow(window_name, image)
cv2.setMouseCallback(window_name, mouse_callback)

while True:
    k = cv2.waitKey(20)
    if k == ord('u') and len(undo_stack) > 0 :
        restored = undo_stack.pop()
        image[:] = restored
        cv2.imshow(window_name, image)
    if k == ord('c') and areaSelected:
        areaSelected = False
        auto_heal_spot(startX, startY)
        cv2.imshow(window_name, image)
    if k == ord('s'):
        cv2.imwrite("modified.jpg", image)
    if k == 27 or k == ord('q'):
        break

cv2.destroyAllWindows()