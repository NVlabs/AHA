import os
import re
import argparse
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec
from PIL import Image

def generate_sequential_visualizations(input_folder, output_folder, viewpoints=None):
    """
    Generate a sequence of visualizations showing incremental frame filling.
    For N frames (starting from 1), creates N-1 images (skipping the 1st) with progressively more filled cells.
    
    The file visualization_1_filled.png (if exists) is removed and subsequent visualizations
    (from filled count 2 onwards) are saved as 0.png, 1.png, 2.png, ... etc.
    
    Args:
        input_folder: Folder containing images named as viewpoint_frame.png.
        output_folder: Folder to save the output visualizations.
        viewpoints: List of viewpoint names (e.g., ['front', 'overhead', 'wrist']).
                   If None, defaults to ['front', 'overhead', 'wrist'].
    """
    # Use default viewpoints if none provided
    if viewpoints is None:
        viewpoints = ['front', 'overhead', 'wrist']
    
    # Create output folder if it doesn't exist
    os.makedirs(output_folder, exist_ok=True)
    
    # Remove visualization_1_filled.png if it exists
    file_to_remove = os.path.join(output_folder, "visualization_1_filled.png")
    if os.path.exists(file_to_remove):
        os.remove(file_to_remove)
        print(f"Removed: {file_to_remove}")
    
    # Get all image files
    all_files = [f for f in os.listdir(input_folder) if f.endswith(('.png', '.jpg', '.jpeg'))]
    
    print(f"Using viewpoints: {viewpoints}")
    
    # Determine the total frames for each viewpoint (skipping frame 0)
    max_frames = {}
    frame_files = {}
    
    for viewpoint in viewpoints:
        # Initialize dictionary to store files for each frame number
        frame_files[viewpoint] = {}
        
        # Find all files for this viewpoint and determine max frame number (ignoring frame 0)
        max_frame_num = -1
        for file in all_files:
            match = re.match(rf'{viewpoint}_(\d+)\.(png|jpg|jpeg)', file)
            if match:
                frame_num = int(match.group(1))
                if frame_num < 1:
                    continue  # ignore frame 0
                max_frame_num = max(max_frame_num, frame_num)
                frame_files[viewpoint][frame_num] = os.path.join(input_folder, file)
        
        # Store the total frames (max frame number, since we start from 1)
        max_frames[viewpoint] = max_frame_num
    
    print(f"Detected max frames per viewpoint: {max_frames}")
    
    # Determine the overall max frames for grid sizing
    overall_max_frames = max(max_frames.values()) if max_frames else 0
    
    if overall_max_frames == 0:
        print("No valid frames detected. Check your folder structure.")
        return
    
    rows = len(viewpoints)
    cols = overall_max_frames
    
    # Function to determine text color based on image brightness at a location
    def get_contrasting_color(image, x_ratio=0.05, y_ratio=0.05, sample_size=30):
        h, w = image.shape[:2]
        x, y = int(x_ratio * w), int(y_ratio * h)
        
        # Get a small sample area around the position
        x_start = max(0, x - sample_size//2)
        x_end = min(w, x + sample_size//2)
        y_start = max(0, y - sample_size//2)
        y_end = min(h, y + sample_size//2)
        
        if len(image.shape) == 3:  # Color image
            sample = image[y_start:y_end, x_start:x_end]
            # Convert to grayscale for brightness calculation
            if sample.shape[2] == 3:  # RGB
                brightness = np.mean(sample, axis=(0, 1)).mean()
            else:  # RGBA or other
                brightness = np.mean(sample[:, :, :3], axis=(0, 1)).mean() 
        else:  # Grayscale
            sample = image[y_start:y_end, x_start:x_end]
            brightness = np.mean(sample)
        
        # Return white text for dark backgrounds, black for light
        return 'white' if brightness < 128 else 'black'
    
    # For each sequential step (starting from 2 filled frames onward)
    for filled_count in range(2, overall_max_frames + 1):
        # New index: subtract 2 so that the first generated image is 0.png
        new_index = filled_count - 2
        print(f"Generating visualization (index {new_index}) with {filled_count} frames filled")
        
        # Create figure with grid layout and black background
        fig = plt.figure(figsize=(cols*4, rows*4), facecolor='black')
        gs = GridSpec(rows, cols, figure=fig, wspace=0.02, hspace=0.02)
        
        # For each viewpoint (row)
        for row, viewpoint in enumerate(viewpoints):
            total_frames = max_frames[viewpoint]
            
            # For each column (possible frame)
            for col in range(cols):
                ax = fig.add_subplot(gs[row, col])
                
                # Calculate the frame number; now starting from 1
                frame_num = col + 1
                
                # Check if this frame should be filled in this iteration
                should_be_filled = col < filled_count and frame_num <= total_frames and frame_num in frame_files[viewpoint]
                
                if should_be_filled:
                    # Load the image
                    image_path = frame_files[viewpoint][frame_num]
                    img = np.array(Image.open(image_path))
                    ax.imshow(img)
                    
                    # Determine text color that contrasts with the image
                    text_color = get_contrasting_color(img)
                    
                    # Add frame number as text at the top left (since images now start from 1)
                    ax.text(0.05, 0.95, f"{frame_num}", transform=ax.transAxes, 
                            fontsize=28, color=text_color, fontweight='bold',
                            verticalalignment='top', horizontalalignment='left')
                else:
                    # Create empty placeholder with "0" on white background in same position
                    ax.set_facecolor('white')
                    ax.text(0.05, 0.95, "0", fontsize=28, 
                           color='black', fontweight='bold',
                           verticalalignment='top', horizontalalignment='left')
                
                # Remove axis ticks and set border color to black
                ax.set_xticks([])
                ax.set_yticks([])
                for spine in ax.spines.values():
                    spine.set_color('black')
                    spine.set_linewidth(2)
        
        # Set tight layout with black background for grid lines
        plt.subplots_adjust(left=0, right=1, top=1, bottom=0, wspace=0.02, hspace=0.02)
        fig.patch.set_facecolor('black')
        
        # Save this sequential visualization with new sequential naming: 0.png, 1.png, 2.png, ...
        output_path = os.path.join(output_folder, f"{new_index}.png")
        plt.savefig(output_path, dpi=150, bbox_inches='tight', 
                    facecolor=fig.get_facecolor(), edgecolor='none')
        plt.close()
        
        print(f"Saved: {output_path}")
    
    print(f"All {overall_max_frames - 1} sequential visualizations generated in {output_folder}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Generate sequential visualizations from input images.")
    parser.add_argument("input_folder", type=str, 
                        help="Folder containing frame images (named as viewpoint_frame.png)")
    parser.add_argument("output_folder", type=str, 
                        help="Folder to save the sequential visualizations")
    parser.add_argument("--viewpoints", nargs='*', default=['front', 'overhead', 'wrist'],
                        help="Optional: list of viewpoint names (e.g., front overhead wrist)")
    
    args = parser.parse_args()
    
    generate_sequential_visualizations(
        input_folder=args.input_folder,
        output_folder=args.output_folder,
        viewpoints=args.viewpoints
    )
