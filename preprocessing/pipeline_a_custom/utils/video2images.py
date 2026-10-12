import cv2
import os
import yaml

def extract_frames_from_video(video_path, output_folder, frame_rate=1, scale_factor=None):
    """
    Extract video frames and save them as image files.

    Args:
        video_path (str): path to the input video file.
        output_folder (str): path to the output image folder.
        frame_rate (int): save one image every N frames. 1 saves every frame,
                          30 saves one image per 30 frames.
        scale_factor (float, optional): downsampling ratio. 
                                        e.g. 0.5 halves both width and height.
    """
    # Check that the video file exists
    if not os.path.exists(video_path):
        print(f"Error: Video file not found at '{video_path}'")
        return

    # Create the output folder (skipped if it exists)
    if not os.path.exists(output_folder):
        os.makedirs(output_folder)
        print(f"Created output folder: '{output_folder}'")

    # Open the video
    video_capture = cv2.VideoCapture(video_path)

    # Check that the video opened correctly
    if not video_capture.isOpened():
        print(f"Error: Could not open video file at '{video_path}'")
        return

    frame_count = 0
    saved_frame_count = 0

    while True:
        # Read a frame from the video
        success, frame = video_capture.read()

        # Stop when no more frames can be read
        if not success:
            break

        # Save an image every frame_rate frames
        if frame_count % frame_rate == 0:
            # Resize if a downsampling ratio is set
            if scale_factor is not None and scale_factor != 1.0:
                # Get the original frame size
                height, width = frame.shape[:2]
                
                # Compute the new resolution from the ratio (cast to int)
                new_width = int(width * scale_factor)
                new_height = int(height * scale_factor)
                
                # Resize with LANCZOS4 interpolation
                frame = cv2.resize(frame, (new_width, new_height), interpolation=cv2.INTER_LANCZOS4)

            frame_filename = os.path.join(output_folder, f"{saved_frame_count:05d}.png")
            cv2.imwrite(frame_filename, frame)
            saved_frame_count += 1
            print(f"Saved {frame_filename}")

        frame_count += 1

    # Release the video
    video_capture.release()
    print(f"\nDone! A total of {saved_frame_count} frames were saved in '{output_folder}'.")

if __name__ == "__main__":

    path2config =  os.path.join('configs','legacy','SMPLX_dataloader','CUSTOM','scarf.yaml')
    with open(path2config, 'r') as f:
        params = yaml.load(f, Loader=yaml.FullLoader)

    video_file = params['Path']['raw_video']
    output_dir = os.path.join(params['Path']['training_folder'], 'images') 
    
    # Downsampling ratio (e.g. 0.5 = half size)
    # Set to None or 1.0 to keep the original size.
    downsample_ratio = 0.5 

    # Run frame extraction
    extract_frames_from_video(video_file, output_dir, frame_rate=2, scale_factor=downsample_ratio)