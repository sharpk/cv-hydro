import cv2
import numpy as np
import matplotlib.pyplot as plt
import sys # Import sys for command-line arguments

def find_and_read_hydrometers(image_path,
                                    adaptive_block_size=5,        # Changed to 5
                                    adaptive_C=7,                 # Changed to 7
                                    dilate_iterations=5,          # Changed to 5
                                    min_hydrometer_area_ratio=1/20, # Changed to 1/20
                                    min_aspect_ratio_filter=3,    # Changed to 3
                                    min_height_filter=100,        # Changed to 100
                                    # --- Meniscus detection specific parameters ---
                                    canny_threshold1=50, canny_threshold2=150, # Canny edge detection thresholds
                                    horizontal_edge_sum_multiplier=10, # Multiplier for edge strength check
                                    liquid_avg_intensity_threshold=100, # Max average intensity for liquid (darker)
                                    air_avg_intensity_threshold=160,    # Min average intensity for air/glass (brighter)
                                    min_contrast_difference=60):        # Min difference between air/liquid for meniscus
                                    

    img_orig = cv2.imread(image_path)
    if img_orig is None:
        print(f"Error: Could not load image from {image_path}")
        return []

    img_display = img_orig.copy()
    gray = cv2.cvtColor(img_orig, cv2.COLOR_BGR2GRAY)
    blurred = cv2.GaussianBlur(gray, (5, 5), 0)

    # --- Hydrometer Localization (using your tuned parameters) ---
    thresh = cv2.adaptiveThreshold(blurred, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
                                   cv2.THRESH_BINARY_INV, adaptive_block_size, adaptive_C)
    kernel = np.ones((5,5), np.uint8)
    dilated = cv2.dilate(thresh, kernel, iterations=dilate_iterations)

    contours, _ = cv2.findContours(dilated, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    hydrometer_regions = []
    img_total_area = img_orig.shape[0] * img_orig.shape[1]
    min_hydrometer_area = img_total_area * min_hydrometer_area_ratio 

    for contour in contours:
        area = cv2.contourArea(contour)
        if area < min_hydrometer_area:
            continue

        x, y, w, h = cv2.boundingRect(contour)
        if w == 0: aspect_ratio = 0
        else: aspect_ratio = float(h) / w

        if aspect_ratio > min_aspect_ratio_filter and h > min_height_filter:
            hydrometer_regions.append({'bbox': (x, y, w, h), 'contour': contour})
            cv2.rectangle(img_display, (x, y), (x + w, y + h), (0, 255, 255), 2) # Yellow bounding box

    if not hydrometer_regions:
        print("No hydrometers found after filtering. Please refine hydrometer isolation parameters if this is unexpected.")
        return [] # Exit if no hydrometers found

    plt.figure(figsize=(10, 8))
    plt.imshow(cv2.cvtColor(img_display, cv2.COLOR_BGR2RGB))
    plt.title("Filtered Hydrometer Bounding Boxes (Yellow BBoxes)")
    plt.show()

    results = []
    # --- Process Each Detected Hydrometer ---
    for i, hydro_region in enumerate(hydrometer_regions):
        x, y, w, h = hydro_region['bbox']
        
        # Define a tighter ROI for the scale reading part
        scale_roi_x1 = x + int(w * 0.2)
        scale_roi_x2 = x + int(w * 0.8)
        scale_roi_y1 = y 
        scale_roi_y2 = y + int(h * 1.05)

        # Ensure ROI is within image bounds
        scale_roi_x1 = max(0, scale_roi_x1)
        scale_roi_x2 = min(img_orig.shape[1], scale_roi_x2)
        scale_roi_y1 = max(0, scale_roi_y1)
        scale_roi_y2 = min(img_orig.shape[0], scale_roi_y2)
        
        hydro_gray_roi = gray[scale_roi_y1:scale_roi_y2, scale_roi_x1:scale_roi_x2]
        
        # --- DEBUG VISUALIZATION: Intensity Profile ---
        row_intensities = np.mean(hydro_gray_roi, axis=1) # Average intensity for each row
        
        plt.figure(figsize=(10, 5))
        plt.subplot(1, 2, 1)
        plt.imshow(hydro_gray_roi, cmap='gray')
        plt.title(f"Hydrometer {i+1} Scale ROI")
        plt.subplot(1, 2, 2)
        plt.plot(row_intensities, range(len(row_intensities)))
        plt.ylim(len(row_intensities) - 1, 0) # Invert Y-axis for plot to match image orientation
        plt.title("Average Row Intensity Profile (Top is Y=0 in ROI)")
        plt.xlabel("Average Intensity")
        plt.ylabel("Row Index (Y-coordinate in ROI)")
        plt.grid(True)
        plt.show()

        print("\n--- INSTRUCTIONS for Meniscus Detection Tuning (Step 1 of 3) ---")
        print("1. **Examine the 'Average Row Intensity Profile' plot.**")
        print("   - Look for a distinct, rapid change in average intensity as you move from top to bottom (or bottom to top).")
        print("   - The liquid is usually darker, so you'll likely see a drop in intensity as you enter the liquid.")
        print("   - Note the approximate intensity value *above* the meniscus (air/glass) and *below* the meniscus (liquid).")
        print("     e.g., if air is ~180-200 and liquid is ~50-80.")
        print("-----------------------------------------------------------------\n")


        # --- MENISCUS DETECTION STRATEGY ---
        meniscus_y_in_roi = -1
        meniscus_found = False
        
        # Apply Canny to the ROI to find prominent edges
        hydro_edges_roi = cv2.Canny(hydro_gray_roi, canny_threshold1, canny_threshold2) 

        # Display edges within ROI for debugging
        plt.figure(figsize=(4, 6))
        plt.imshow(hydro_edges_roi, cmap='gray')
        plt.title(f"Hydrometer {i+1} ROI Edges")
        plt.show()

        print("\n--- INSTRUCTIONS for Meniscus Detection Tuning (Step 2 of 3) ---")
        print("1. **Examine the 'Hydrometer X ROI Edges' plot.**")
        print("   - Do you see a clear, distinct horizontal edge at the level where the liquid meets the air?")
        print("   - If the meniscus edge is faint or missing: **DECREASE `current_canny_threshold1` and `current_canny_threshold2`** in the `find_and_read_hydrometers` call (e.g., 30, 90).")
        print("   - If there's too much noise (many irrelevant edges): **INCREASE `current_canny_threshold1` and `current_canny_threshold2`** (e.g., 70, 200).")
        print("2. Now, consider the *strength* of the meniscus edge. Is it a long, continuous line or a very short, broken one?")
        print("   - This relates to `current_horizontal_edge_sum_multiplier` below.")
        print("-----------------------------------------------------------------\n")


        # Iterate from the bottom of the ROI upwards, looking for the highest strong horizontal edge
        scan_height = hydro_edges_roi.shape[0]

        for current_y_in_roi in range(scan_height - 5, int(scan_height * 0.1), -1): 
            # Check for a strong horizontal line in the edge image at this Y-coordinate
            if np.sum(hydro_edges_roi[current_y_in_roi, :]) > (hydro_edges_roi.shape[1] * horizontal_edge_sum_multiplier): 
                
                # Now, confirm it's likely the meniscus by checking intensity above and below it
                if current_y_in_roi + 10 < scan_height and current_y_in_roi - 10 >= 0: 
                    avg_intensity_below_edge = np.mean(hydro_gray_roi[current_y_in_roi + 5 : current_y_in_roi + 10, :])
                    avg_intensity_above_edge = np.mean(hydro_gray_roi[current_y_in_roi - 10 : current_y_in_roi - 5, :])

                    # Heuristic: Liquid (below) is dark, air/glass (above) is bright, and there's a good contrast
                    if avg_intensity_below_edge < liquid_avg_intensity_threshold and \
                       avg_intensity_above_edge > air_avg_intensity_threshold and \
                       (avg_intensity_above_edge - avg_intensity_below_edge) > min_contrast_difference:
                        
                        meniscus_y_in_roi = current_y_in_roi
                        meniscus_found = True
                        break 
        
        # Convert meniscus Y-coordinate back to original image coordinates
        meniscus_y_global = meniscus_y_in_roi + scale_roi_y1 if meniscus_found else -1

        if meniscus_y_global != -1:
            cv2.line(img_display, (scale_roi_x1, meniscus_y_global), (scale_roi_x2, meniscus_y_global),
                     (0, 0, 255), 2, cv2.LINE_AA) # Red line for meniscus
            print(f"Hydrometer {i+1}: Identified Meniscus Y-coordinate: {meniscus_y_global}")

            # --- Calibrate and Map to Specific Gravity ---
            h_scale_roi = scale_roi_y2 - scale_roi_y1
            
            # These relative positions (0.10, 0.25, 0.65) are still *estimates* based on the hydrometer's height.
            # For accurate specific gravity, these marks should ideally be located precisely using OCR or template matching.
            mark_0_990_relative_y = 0.10  # Roughly top of the blue band
            mark_1_000_relative_y = 0.25  # Roughly top of the green band (where "1.000" text is)
            mark_1_020_relative_y = 0.65  # Roughly line labeled '20'

            known_mark_top_y = scale_roi_y1 + int(h_scale_roi * mark_0_990_relative_y)
            known_mark_top_value = 0.990

            known_mark_bottom_y = scale_roi_y1 + int(h_scale_roi * mark_1_020_relative_y)
            known_mark_bottom_value = 1.020
            
            # Draw calibration marks for visualization
            cv2.line(img_display, (scale_roi_x1-5, known_mark_top_y), (scale_roi_x2+5, known_mark_top_y), (0, 255, 0), 1)
            cv2.putText(img_display, "0.990", (scale_roi_x2 + 10, known_mark_top_y), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1, cv2.LINE_AA)
            
            cv2.line(img_display, (scale_roi_x1-5, known_mark_bottom_y), (scale_roi_x2+5, known_mark_bottom_y), (0, 255, 0), 1)
            cv2.putText(img_display, "1.020", (scale_roi_x2 + 10, known_mark_bottom_y), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1, cv2.LINE_AA)

            specific_gravity = None
            if known_mark_top_y <= meniscus_y_global <= known_mark_bottom_y:
                proportion = (meniscus_y_global - known_mark_top_y) / (known_mark_bottom_y - known_mark_top_y)
                specific_gravity = known_mark_top_value + proportion * (known_mark_bottom_value - known_mark_top_value)
                
                print(f"Hydrometer {i+1}: Estimated Specific Gravity: {specific_gravity:.3f}")
                cv2.putText(img_display, f"SG: {specific_gravity:.3f}", (x, y - 10),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2, cv2.LINE_AA)
            else:
                print(f"Hydrometer {i+1}: Meniscus ({meniscus_y_global}) is outside the defined calibration range ({known_mark_top_y}-{known_mark_bottom_y}).")
            
            results.append({
                'id': i+1,
                'bbox': hydro_region['bbox'],
                'meniscus_y': meniscus_y_global,
                'specific_gravity': specific_gravity
            })
        else:
            print(f"Hydrometer {i+1}: Could not detect meniscus. Try adjusting `canny_threshold1/2`, `horizontal_edge_sum_multiplier`, or intensity thresholds (`liquid_avg_intensity_threshold`, `air_avg_intensity_threshold`, `min_contrast_difference`).")
            results.append({
                'id': i+1,
                'bbox': hydro_region['bbox'],
                'meniscus_y': None,
                'specific_gravity': None
            })

    plt.figure(figsize=(10, 8))
    plt.imshow(cv2.cvtColor(img_display, cv2.COLOR_BGR2RGB))
    plt.title("Final Result with Detected Meniscus and SG")
    plt.show()

    return results

# --- How to use it ---
# Get image path from command-line argument
if len(sys.argv) < 2:
    print("Usage: python your_script_name.py <path_to_image>")
    sys.exit(1)
image_path = sys.argv[1]

# --- TUNABLE MENISCUS DETECTION PARAMETERS ---
# Adjust these based on the visual debugging plots and console output instructions.
# Hydrometer isolation parameters are now set as default in the function definition.
current_canny_threshold1 = 70          # Canny lower threshold
current_canny_threshold2 = 200         # Canny upper threshold
current_horizontal_edge_sum_multiplier = 50 # Adjust for strength of horizontal edges
current_liquid_avg_intensity_threshold = 160 # Max average intensity for liquid region (e.g., 0-100)
current_air_avg_intensity_threshold = 120    # Min average intensity for air/glass region (e.g., 160-255)
current_min_contrast_difference = 20         # Min difference between air & liquid intensities

analysis_results = find_and_read_hydrometers(
    image_path,
    canny_threshold1=current_canny_threshold1,
    canny_threshold2=current_canny_threshold2,
    horizontal_edge_sum_multiplier=current_horizontal_edge_sum_multiplier,
    liquid_avg_intensity_threshold=current_liquid_avg_intensity_threshold,
    air_avg_intensity_threshold=current_air_avg_intensity_threshold,
    min_contrast_difference=current_min_contrast_difference
)

print("\n--- Summary of Results ---")
for res in analysis_results:
    print(f"Hydrometer {res['id']}:")
    print(f"  Bounding Box: {res['bbox']}")
    print(f"  Meniscus Y-coord: {res['meniscus_y']}")
    print(f"  Specific Gravity: {res['specific_gravity'] if res['specific_gravity'] is not None else 'N/A'}")