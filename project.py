import base64
import gc
import io
import os
import pathlib
from typing import TypeAlias, cast


import PIL
import cv2
import numpy as np
import torch
import torchvision.transforms as torchvision_T
from PIL import Image
from torchvision.models.segmentation import (
    DeepLabV3,
    deeplabv3_mobilenet_v3_large,
    deeplabv3_resnet50,
)

Mat: TypeAlias = np.ndarray[int, np.dtype[np.generic]]


def load_model(num_classes=2, device=torch.device("cpu")):
    
    model = deeplabv3_mobilenet_v3_large(num_classes=num_classes, aux_loss=True)
    checkpoint_path = os.path.join(os.getcwd(), "SCANNER_WEIGHTS.pth")

    model.to(device)
    checkpoints = torch.load(checkpoint_path, map_location=device)
    model.load_state_dict(checkpoints, strict=False)
    model.eval()

    _ = model(torch.randn((1, 3, 384, 384)))

    return model


def image_preprocess_transforms(
    mean=(0.4611, 0.4359, 0.3905), std=(0.2193, 0.2150, 0.2109)
):
    common_transforms = torchvision_T.Compose(
        [
            torchvision_T.ToTensor(),
            torchvision_T.Normalize(mean, std),
        ]
    )
    return common_transforms


def order_points(pts):

    rect = np.zeros((4, 2), dtype="float32")
    pts = np.array(pts)
    s = pts.sum(axis=1)

    rect[0] = pts[np.argmin(s)]
    rect[2] = pts[np.argmax(s)]

    diff = np.diff(pts, axis=1)

    rect[1] = pts[np.argmin(diff)]
    rect[3] = pts[np.argmax(diff)]
    
    return rect.astype("int").tolist()


def find_dest(pts):
    (tl, tr, br, bl) = pts

    widthA = np.sqrt(((br[0] - bl[0]) ** 2) + ((br[1] - bl[1]) ** 2))
    widthB = np.sqrt(((tr[0] - tl[0]) ** 2) + ((tr[1] - tl[1]) ** 2))
    maxWidth = max(int(widthA), int(widthB))


    heightA = np.sqrt(((tr[0] - br[0]) ** 2) + ((tr[1] - br[1]) ** 2))
    heightB = np.sqrt(((tl[0] - bl[0]) ** 2) + ((tl[1] - bl[1]) ** 2))
    maxHeight = max(int(heightA), int(heightB))

    destination_corners = [[0, 0], [maxWidth, 0], [maxWidth, maxHeight], [0, maxHeight]]

    return order_points(destination_corners)


def process_image(scan_mat: np.ndarray, trained_model: DeepLabV3, image_size=384, buffer=10):
    global preprocess_transforms

    TARGET_SIZE = image_size
    half = TARGET_SIZE // 2
    original_height, original_width, channels = scan_mat.shape
    resized_mat = cv2.resize(scan_mat, (TARGET_SIZE, TARGET_SIZE), interpolation=cv2.INTER_NEAREST)

    scale_x = original_width / TARGET_SIZE
    scale_y = original_height / TARGET_SIZE

    preprocessed_mat = preprocess_transforms(resized_mat)
    model_input = torch.unsqueeze(torch.tensor(preprocessed_mat), dim=0)

    with torch.no_grad():
        output = trained_model(model_input)["out"].cpu()

    del model_input
    gc.collect()

    output = torch.argmax(output, dim=1, keepdim=True).permute(0, 2, 3, 1)[0].numpy().squeeze().astype(np.int32)
    r_height, r_width = output.shape

    extended_output = np.zeros((TARGET_SIZE + r_height, TARGET_SIZE + r_width), dtype=output.dtype)
    extended_output[half:half + TARGET_SIZE, half:half + TARGET_SIZE] = output * 255
    output = extended_output.copy()

    del extended_output
    gc.collect()

    canny_edge_map = cv2.Canny(output.astype(np.uint8), 225, 255)
    canny_edge_map = cv2.dilate(canny_edge_map, cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5)))
    contours, _ = cv2.findContours(canny_edge_map, cv2.RETR_LIST, cv2.CHAIN_APPROX_NONE)
    page_contour = sorted(contours, key=cv2.contourArea, reverse=True)[0]

    epsilon = 0.02 * cv2.arcLength(page_contour, True)
    corners = cv2.approxPolyDP(page_contour, epsilon, True)

    corners = np.concatenate(corners).astype(np.float32)

    corners[:, 0] -= half
    corners[:, 1] -= half

    corners[:, 0] *= scale_x
    corners[:, 1] *= scale_y

    if not (
        np.all(corners.min(axis=0) >= (0, 0)) and np.all(corners.max(axis=0) <= (original_width, original_height))
    ):

        left_pad, top_pad, right_pad, bottom_pad = 0, 0, 0, 0

        rect = cv2.minAreaRect(corners.reshape((-1, 1, 2)))
        box_corners = np.int32(cv2.boxPoints(rect))

        box_x_min = np.min(box_corners[:, 0])
        box_x_max = np.max(box_corners[:, 0])
        box_y_min = np.min(box_corners[:, 1])
        box_y_max = np.max(box_corners[:, 1])

        if box_x_min <= 0:
            left_pad += abs(box_x_min) + buffer

        if box_x_max >= original_width:
            right_pad += box_x_max - original_width + buffer

        if box_y_min <= 0:
            top_pad += abs(box_y_min) + buffer

        if box_y_max >= original_height:
            bottom_pad += box_y_max - original_height + buffer

        image_extended = np.zeros(
            (top_pad + bottom_pad + original_height, left_pad + right_pad + original_width, channels),
            dtype=scan_mat.dtype,
        )

        image_extended[top_pad:top_pad+original_height,left_pad:left_pad+original_width,:] = scan_mat
        image_extended = image_extended.astype(np.float32)

        box_corners[:, 0] += left_pad
        box_corners[:, 1] += top_pad

        corners = box_corners
        scan_mat = image_extended

    corners_sorted = sorted(map(tuple, corners.tolist()))
    corners_sorted_ordered = order_points(corners_sorted)
    destination_corners_sorted_ordered = find_dest(corners_sorted_ordered)
    transform_matrix_sorted_ordered = cv2.getPerspectiveTransform(np.float32(corners_sorted_ordered), np.float32(destination_corners_sorted_ordered))

    output_warped_sorted_ordered = cv2.warpPerspective(
        scan_mat.astype(np.float32),
        transform_matrix_sorted_ordered,
        (destination_corners_sorted_ordered[2][0], destination_corners_sorted_ordered[2][1]),
        flags=cv2.INTER_LANCZOS4,
    )
    
    output_warped_sorted_ordered.clip(0.0)
    
    return output_warped_sorted_ordered.astype(np.uint8)

IMAGE_SIZE = 384
preprocess_transforms = image_preprocess_transforms()


model = load_model()

threshold = 11
c = 4

directory_path = "inputs"
file_list = [f for f in os.listdir(directory_path) if os.path.isfile(os.path.join(directory_path, f))]

i = 1
for fadd in file_list:
    image = cv2.imread(directory_path + "/" + fadd, 1)
    h, w = image.shape[:2]
    final = process_image(scan_mat=image, trained_model=model, image_size=IMAGE_SIZE)
    final = cv2.cvtColor(final, cv2.COLOR_BGR2GRAY)
    final = cv2.adaptiveThreshold(
            final, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, threshold, c
        )

    cv2.imwrite("out" + str(i) + ".jpg", final)
    i = i + 1