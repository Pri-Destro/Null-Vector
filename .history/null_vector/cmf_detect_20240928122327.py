import argparse
from quant_matrix import QuantizationMatrix
from utils.helper_utils import (
    read_img, create_quantize_dct, lexographic_sort, shift_vector_thresh, display_results
)
import cv2  # Import OpenCV for image resizing

if __name__ == "__main__":

    # Create the parser
    parser = argparse.ArgumentParser()

    # Add an argument
    parser.add_argument('--img', required=True, help="Path of the image on which operation needs to be performed")
    parser.add_argument('--block_size', type=int, default=8, help="Block Size")
    parser.add_argument('--qf', type=float, default=0.75, help="Quality Factor")
    parser.add_argument('--shift_thresh', type=int, default=10, help="Threshold for shift vector count")
    parser.add_argument('--stride', type=int, default=1, help="Sliding window stride count / overlap")
    parser.add_argument('--scale_percent', type=int, default=100, help="Downscale image by percentage (e.g., 50 for 50%)")

    # Parse the argument
    args = parser.parse_args()

    img_path = args.img
    block_size = args.block_size
    qf = args.qf
    shift_thresh = args.shift_thresh
    stride = args.stride
    scale_percent = args.scale_percent  # Get the scaling percentage from arguments

    # 8x8 quantization matrix based on QF
    Q_8x8 = QuantizationMatrix().get_qm(qf)

    # Read img
    img, original_image, overlay, width, height = read_img(img_path)

    # Downscale image if scale_percent is less than 100%
    if scale_percent < 100:
        img = cv2.resize(img, (int(width * scale_percent / 100), int(height * scale_percent / 100)), interpolation=cv2.INTER_AREA)
        original_image = cv2.resize(original_image, (int(width * scale_percent / 100), int(height * scale_percent / 100)), interpolation=cv2.INTER_AREA)
        overlay = cv2.resize(overlay, (int(width * scale_percent / 100), int(height * scale_percent / 100)), interpolation=cv2.INTER_AREA)
        width = int(width * scale_percent / 100)
        height = int(height * scale_percent / 100)

    # DCT
    quant_row_matrices = create_quantize_dct(img, width, height, block_size, stride, Q_8x8)

    # Lexographic sort
    shift_vec_count, matched_blocks = lexographic_sort(quant_row_matrices)

    # Shift vector thresholding
    matched_pixels_start = shift_vector_thresh(shift_vec_count, matched_blocks, shift_thresh)

    # Displaying output
    display_results(overlay, original_image, matched_pixels_start, block_size)
