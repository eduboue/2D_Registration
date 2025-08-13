#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import argparse
import os
import json
import h5py
import cv2
import numpy as np
import tifffile as tiff
from scipy.signal import savgol_filter

### --- Instruction Page ---
description = """"
This is a program that allows a user to register
time-series image data aquired from confocal or two-photon data.

The function embedded in the program are:

    load_multitiff_image(file_path)
    compute_average_reference(images, num_frames=100)
    smooth_registration_offsets
    register_images(img_stack, template, downsample_rates, max_movement)
    apply_registration(img_stack, reg_offsets)
    save_stack(images, save_path)

The general stratagy for register images is to downsamnple the images
heirarchially (starting a small number and working your way up to 1:1), then
assessing correlations at each interval. It then assesses the best correaltion
and uses that best correlation to make a x-displacmeent, y-displacmeent. Sometimes
jitter can corrupt the registration, so this allows you to both interate
registrations (i.e., allow you to register once, and then re-register the
registered image, however many times you want). There is also a stratagy
to smooth the transforamtion matrix before applying - this essentailly
nessecitates (essentailly because you dont have to, but for functionality
you should) iterating registerations.

The general workflow would be to laod the tiff, make a average stack,
view image (both static and time-lapse), generate trasformation matrix and
funally apply it. You can view your results also to ensure drift has (mostly)
been corrected for.

Inputs:
    -i, --input, (required=True)                        Sets path to input TIFF file
    -o, --output (required=True)                        Sets path to save registered TIFF file
    --iterate    (type=int, default=1)                  Sets number of times to iterate the registration process
    -s, --smooth (type=int, choices=[0, 1], default=1)  Apply smoothing to registration offsets (1=Yes, 0=No)
    --num_frames (type=int, default=100)                Number of frames to use for computing the average reference image (default: 100)")
    --downsample_rates, (type=parse_vector, \
          default=[1/16, 3/32, 1/8, 3/16, 1/4, 1/2, 1]) Downsample rates as a vector, e.g., [1/16, 3/32, 1/8, 3/16, 1/4, 1/2, 1]
    --max_movement (type=float, default=0.5)            Maximum movement for registration (default: 0.5)
    --save_as_tif (type=int, choices [0,1], default=0)  Allows user to spefic saving output as a tiff without metadata.

Output:
    Saves a registered image stack (multitiff) to the output directory

Example:
    python RegistrationScriptFINAL_hd5.py -i /path/to/input.tif -o /path/to/output.tif --iterate 2 -s 1 --num_frames 50 --downsample_rates [1/16,3/32,1/8,3/16,1/4,1/2,1] --max_movement 0.5 --save_as_tif 0

    NOTE: If you are specifying downsample_rates, make sure this is written in brackets ("[]") and that there are no spaces. The code can find the comma and seperate that way.

Written: Erik R. Duboué, Florida Atlantic University
www.erikduboue.com
Spring 2025

"""
### --- Function to Load HDF5 Data ---
def load_hdf5_data(file_path):
    images, metadata = None, {}

    try:
        with h5py.File(file_path, "r") as file:
            if "image_data" not in file:
                raise ValueError("No image data found in the HDF5 file.")
            images = file["image_data"][:]

            if "metadata" in file:
                metadata_group = file["metadata"]
                for key, value in metadata_group.attrs.items():
                    try:
                        metadata[key] = json.loads(value) if isinstance(value, str) and value.strip() else value
                    except json.JSONDecodeError:
                        metadata[key] = value  # Store as-is if not JSON

    except Exception as e:
        print(f"Error loading HDF5 file: {e}")
        return None, {}

    return images, metadata

### --- Function to Save HDF5 Data ---
def save_hdf5_data(images, metadata, output_path):
    with h5py.File(output_path, "w") as file:
        file.create_dataset("image_data", data=images, compression="gzip")
        metadata_group = file.create_group("metadata")
        for key, value in metadata.items():
            metadata_group.attrs[key] = json.dumps(value) if isinstance(value, (dict, list)) else str(value)

    print(f"Registered images saved as HDF5: {output_path}")

def load_multitiff_image(file_path):
    if not file_path.lower().endswith(('.tif', '.tiff')):
        raise ValueError("File must be a .tif or .tiff format")
    try:
        with tiff.TiffFile(file_path) as tif:
            images = tif.asarray()
    except Exception as e:
        raise RuntimeError(f"Error loading TIFF file: {e}")
    if images.ndim == 2:
        print("Warning: You have loaded a single image. A time-series dataset is required.")
        return None
    elif images.ndim == 3 and images.shape[0] > images.shape[1]:
        print(f"Successfully loaded a {images.shape[1]}x{images.shape[2]} image with {images.shape[0]} frames.")
        return images
    elif images.ndim == 4:
        print("Warning: Cannot accept 3D volumetric images. Load a time-series dataset from a single focal plane.")
        return None
    raise ValueError

def compute_average_reference(images, num_frames=100):
    if images is None:
        print("No valid time-series data loaded. Please check your input file.")
        return None
    num_frames = min(num_frames, images.shape[0])
    avg_image = np.mean(images[:num_frames], axis=0)
    print(f"Computed averaged reference image using the first {num_frames} frames.")
    return avg_image

def smooth_registration_offsets(reg_offsets, window_length=11, polyorder=3):
    if len(reg_offsets) < window_length:
        print("Warning: Too few frames for smoothing. Returning original offsets.")
        return reg_offsets
    smoothed_offsets = np.zeros_like(reg_offsets)
    smoothed_offsets[:, 0] = savgol_filter(reg_offsets[:, 0], window_length, polyorder)
    smoothed_offsets[:, 1] = savgol_filter(reg_offsets[:, 1], window_length, polyorder)
    return smoothed_offsets

def register_images(img_stack, template, downsample_rates, max_movement):

    """
    Perform hierarchical image registration using correlation-based alignment.

    Parameters:
    - img_stack (np.array): 3D NumPy array (frames, height, width) in uint16.
    - template (np.array): First image as reference (height, width).
    - downsample_rates (list): List of downsampling factors.
    - max_movement (float): Maximum expected movement in the images.

    Returns:
    - reg_offsets (np.array): Array of (Y, X) registration offsets for each frame, returned as uint16.
    """

    depth, height, width = img_stack.shape
    reg_offsets = np.zeros((depth, 2), dtype=np.float32)  # Store (Y, X) offsets as float32 during processing

    for r, downsample in enumerate(downsample_rates):
        print(f"Registering images, iteration {r+1}/{len(downsample_rates)}...")

        # Downsample template and convert to float32 for OpenCV processing
        down_height, down_width = int(height * downsample), int(width * downsample)
        template_img = cv2.resize(template, (down_width, down_height), interpolation=cv2.INTER_LINEAR).astype(np.float32)

        for d in range(1, depth):  # Start from second image (index 1)
            reg_img = cv2.resize(img_stack[d], (down_width, down_height), interpolation=cv2.INTER_LINEAR).astype(np.float32)

            if r == 0:
                # Initial search bounds
                min_offset_y = -int(round(max_movement * down_height / 2))
                max_offset_y = int(round(max_movement * down_height / 2))
                min_offset_x = -int(round(max_movement * down_width / 2))
                max_offset_x = int(round(max_movement * down_width / 2))
            else:
                # Refining previous offsets
                prev_y, prev_x = reg_offsets[d] * downsample_rates[r]
                refine_window = downsample / downsample_rates[r-1] / 2
                min_offset_y, max_offset_y = int(prev_y - refine_window), int(prev_y + refine_window)
                min_offset_x, max_offset_x = int(prev_x - refine_window), int(prev_x + refine_window)

            best_corr_value = -1
            best_corr_x, best_corr_y = 0, 0

            # Brute-force search within offset range
            for y in range(min_offset_y, max_offset_y + 1):
                for x in range(min_offset_x, max_offset_x + 1):
                    # Extract overlapping regions
                    sub_template = template_img[max(y, 0):down_height+min(y, 0), max(x, 0):down_width+min(x, 0)]
                    sub_reg = reg_img[max(-y, 0):down_height+min(-y, 0), max(-x, 0):down_width+min(-x, 0)]

                    # Ensure images are the same size
                    if sub_template.shape != sub_reg.shape:
                        continue

                    # Compute correlation (fixing the 16-bit issue)
                    correlation = cv2.matchTemplate(sub_reg, sub_template, method=cv2.TM_CCOEFF_NORMED)
                    corr_value = np.max(correlation)

                    # Track best match
                    if corr_value > best_corr_value:
                        best_corr_x, best_corr_y = x, y
                        best_corr_value = corr_value

            # Store offsets, scaled back to original resolution
            reg_offsets[d] = [best_corr_y / downsample, best_corr_x / downsample]


    print("Done! Registration offsets (Y, X):")
    print(reg_offsets)
    return reg_offsets

def apply_registration(img_stack, reg_offsets):
    depth, height, width = img_stack.shape
    registered_stack = np.zeros_like(img_stack, dtype=np.uint16)
    for d in range(depth):
        img_float32 = img_stack[d].astype(np.float32)
        offset_y, offset_x = reg_offsets[d]
        transformation_matrix = np.array([[1, 0, offset_x], [0, 1, offset_y]], dtype=np.float32)
        registered_img = cv2.warpAffine(img_float32, transformation_matrix, (width, height), flags=cv2.INTER_LINEAR)
        registered_stack[d] = np.clip(registered_img, 0, 65535).astype(np.uint16)
    print("Registration applied to all frames.")
    return registered_stack

def save_stack(images, save_path):
    if images is None:
        print("No image data to save.")
        return
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    tiff.imwrite(save_path, images, dtype=images.dtype)
    print(f"Saved stack as TIFF: {save_path}")

def parse_vector(arg):
    try:
        return [eval(x) for x in arg.strip('[]').split(',')]
    except Exception:
        raise argparse.ArgumentTypeError("Downsample rates must be a list of numbers, e.g., [1/16, 1/8, 1/4, 1/2, 1]")

def main():
    parser = argparse.ArgumentParser(description="Time-Series Image Registration CLI Tool")
    parser.add_argument("-i", "--input", required=True, help="Path to input HDF5 (.h5, .hd5) or TIFF (.tif) file")
    parser.add_argument("-o", "--output", required=True, help="Path to save registered output")
    parser.add_argument("--save_as_tif", type=int, choices=[0, 1], default=0, help="Save as TIFF (default: HDF5)")
    parser.add_argument("--iterate", type=int, default=1, help="Number of times to iterate the registration process (default: 1)")
    parser.add_argument("-s", "--smooth", type=int, choices=[0, 1], default=0, help="Apply smoothing to registration offsets (1=Yes, 0=No, default: 1)")
    parser.add_argument("--num_frames", type=int, default=100, help="Number of frames to use for computing the average reference image (default: 100)")
    parser.add_argument("--downsample_rates", type=parse_vector, default=[1/8,1/4,1/2,1], help="Downsample rates as a list, e.g., [1/16,1/8,1/4,1/2,1]")
    parser.add_argument("--max_movement", type=float, default=0.5, help="Maximum movement for registration (default: 0.5)")

    args = parser.parse_args()

    print(f"Loading input file: {args.input}")
    images, metadata = load_hdf5_data(args.input) if args.input.endswith((".h5", ".hd5")) else (load_multitiff_image(args.input), {})

    if images is None:
        print("Failed to load input file. Exiting.")
        return

    print("Computing average reference image...")
    AVG_image = compute_average_reference(images, num_frames=args.num_frames)

    print("Setting up registration parameters.")
    downsample_rates = args.downsample_rates
    max_movement = args.max_movement

    for iteration in range(args.iterate):
        print(f"Entering registration level {iteration+1}")
        reg_offsets = register_images(images, AVG_image, downsample_rates, max_movement)

        if args.smooth:
            reg_offsets = smooth_registration_offsets(reg_offsets)
            print("Smoothing applied to registration offsets.")

        images = apply_registration(images, reg_offsets)
        print(f"Registration level {iteration+1} completed.")
        print("")
        print("")

    print(f"Saving results as: {args.output}")
    save_hdf5_data(images, metadata, args.output) if not args.save_as_tif else tiff.imwrite(args.output, images)

if __name__ == "__main__":
    main()
