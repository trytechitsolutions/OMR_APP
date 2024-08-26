import cv2
import numpy as np

def biggest_contour(contours):
    biggest = np.array([])
    max_area = 0
    for contour in contours:
        area = cv2.contourArea(contour)
        if area > 1000:
            peri = cv2.arcLength(contour, True)
            approx = cv2.approxPolyDP(contour, 0.015 * peri, True)
            
            if area > max_area and len(approx) == 4:
                biggest = approx
                max_area = area     
    return biggest

def find_paper(image):
    '''
    Find an answer sheet in the image and auto crop it
    '''
    # define readed answersheet image output size
    (max_width, max_height) = (1000, 1300)
    
    img_original = image.copy()
    
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    gray = cv2.bilateralFilter(gray, 10, 75, 75)  # Add bilateral filtering to reduce noise
    edged = cv2.Canny(gray, 10, 20)

    (contours, _) = cv2.findContours(edged.copy(), cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
    contours = sorted(contours, key=cv2.contourArea, reverse=True)[:10]

    biggest = biggest_contour(contours)

    cv2.drawContours(image, [biggest], -1, (0, 255, 0), 3)

    # Pixel values in the original image
    points = biggest.reshape(4, 2)
    input_points = np.zeros((4, 2), dtype="float32")

    points_sum = points.sum(axis=1)
    input_points[0] = points[np.argmin(points_sum)]
    input_points[3] = points[np.argmax(points_sum)]

    points_diff = np.diff(points, axis=1)
    input_points[1] = points[np.argmin(points_diff)]
    input_points[2] = points[np.argmax(points_diff)]

    # Desired points values in the output image
    converted_points = np.float32([[0, 0], [max_width, 0], [0, max_height], [max_width, max_height]])

    # Perspective transformation
    matrix = cv2.getPerspectiveTransform(input_points, converted_points)
    img_output = cv2.warpPerspective(img_original, matrix, (max_width, max_height))
    
    return img_output

    
def read_id(img,no_of_ids):

    no_of_id_choice=10
    # no_of_ids=10
    rs_img=cv2.resize(img,((img.shape[1]//no_of_ids)*no_of_ids,
                        (img.shape[0]//no_of_id_choice)*no_of_id_choice))
    grey = cv2.cvtColor(rs_img, cv2.COLOR_BGR2GRAY)
    blurred = cv2.GaussianBlur(grey, (3, 3), 0)  # Apply Gaussian blur to reduce noise
    res = cv2.threshold(blurred, 170, 255, cv2.THRESH_BINARY_INV)[1]
    kernel = np.ones((5, 5), np.uint8)
    morphed = cv2.morphologyEx(res, cv2.MORPH_GRADIENT, kernel)
    rows = np.hsplit(morphed,no_of_ids)
    boxes=[]
    for r in rows: 
        cols= np.vsplit(r,no_of_id_choice)
        for box in cols:
            boxes.append(box)
    pxl = [cv2.countNonZero(box) for box in boxes]
    val = [1 if 200 <= cv2.countNonZero(box) < 300 else 0 for box in boxes]
    val = np.array(val).reshape((no_of_ids, no_of_id_choice))
    bubbled = [np.argmax(val) if np.sum(val) == 1 else 'invalid' for val in val]
    ids="".join([str(i) for i in bubbled])
    if len(ids)==no_of_ids:
        return ids
    else:
        return 'invalid id'

def get_ans_200(img):
#img=q1
    no_of_choice=4
    no_of_qsn=40
    
    rs_img=cv2.resize(img,((img.shape[1]//no_of_choice)*no_of_choice,
                   (img.shape[0]//no_of_qsn)*no_of_qsn))
    grey = cv2.cvtColor(rs_img, cv2.COLOR_BGR2GRAY)
    blurred = cv2.GaussianBlur(grey, (3, 3), 0)  # Apply Gaussian blur to reduce noise
    res = cv2.threshold(blurred, 170, 255, cv2.THRESH_BINARY_INV)[1]
    kernel = np.ones((5, 5), np.uint8)
    morphed = cv2.morphologyEx(res, cv2.MORPH_GRADIENT, kernel)
    rows = np.vsplit(morphed,no_of_qsn)
    boxes=[]
    for r in rows:
        cols= np.hsplit(r,no_of_choice)
        for box in cols:
            boxes.append(box)
    '''
    (200 pxls,330 pxls) are takng as a treshold values
    '''
    # Create a binary list based on pixel count
    # pxls = [cv2.countNonZero(box) for box in boxes]
    l = [1 if 200 <= cv2.countNonZero(box) < 330 else 0 for box in boxes]
    # Reshape the list into a 2D numpy array
    A = np.array(l).reshape((no_of_qsn, no_of_choice))
    
    bubbled = [np.argmax(val) + 1 if np.sum(val) == 1 else 'invalid' for val in A]
    return bubbled

def get_ans_100(img):
#img=q1
    no_of_choice=4
    no_of_qsn=20
    
    rs_img=cv2.resize(img,((img.shape[1]//no_of_choice)*no_of_choice,
                   (img.shape[0]//no_of_qsn)*no_of_qsn))
    grey = cv2.cvtColor(rs_img, cv2.COLOR_BGR2GRAY)
    blurred = cv2.GaussianBlur(grey, (3, 3), 0)  # Apply Gaussian blur to reduce noise
    res = cv2.threshold(blurred, 170, 255, cv2.THRESH_BINARY_INV)[1]
    kernel = np.ones((5, 5), np.uint8)
    morphed = cv2.morphologyEx(res, cv2.MORPH_GRADIENT, kernel)
    rows = np.vsplit(morphed,no_of_qsn)
    boxes=[]
    for r in rows:
        cols= np.hsplit(r,no_of_choice)
        for box in cols:
            boxes.append(box)
    '''
    (250 pxls,380 pxls) are treshold values
    '''
    # Create a binary list based on pixel count
    l = [1 if 250 <= cv2.countNonZero(box) < 380 else 0 for box in boxes]
    # Reshape the list into a 2D numpy array
    A = np.array(l).reshape((no_of_qsn, no_of_choice))
    bubbled = [np.argmax(val) + 1 if np.sum(val) == 1 else 'invalid' for val in A]
    return bubbled