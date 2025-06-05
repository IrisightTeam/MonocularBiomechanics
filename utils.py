import cv2
import numpy as np
from tqdm import tqdm
import tensorflow_hub as hub

def video_reader(filename: str, batch_size: int = 8, width: int | None = None):
    """
    Read a video file and yield frames in batches.

    In theory, tensorflow_io has tools for this but they don't seem to work for me. That
    is probably more efficient if it works as they can prefetch. This also will optionally
    downsample the video if compute is a limit.

    Args:
        filename: (str) The path to the video file.
        batch_size: (int) The number of frames to yield at once.
        width: (int | None) The width to downsample to. If None, the original width is used.

    Returns:
        A tuple of (generator, n_frames) where generator yields batches and n_frames is total frame count
    """

    cap = cv2.VideoCapture(filename)
    
    # Get total frame count
    n_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    
    def frame_generator():
        cap = cv2.VideoCapture(filename)  # Create new capture object for generator
        frames = []
        while True:
            ret, frame = cap.read()

            if ret is False:
                if len(frames) > 0:
                    frames = np.array(frames)
                    yield frames
                cap.release()
                return
            else:
                frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

                if width is not None:
                    # downsample to keep the aspect ratio and output the specified width
                    scale = width / frame.shape[1]
                    height = int(frame.shape[0] * scale)
                    frame = cv2.resize(frame, (width, height))

                frames.append(frame)

                if len(frames) >= batch_size:
                    frames = np.array(frames)
                    yield frames
                    frames = []
    
    cap.release()  # Release the initial capture object
    return frame_generator(), n_frames

def load_metrabs():
    if load_metrabs.model is not None:
        return load_metrabs.model
    load_metrabs.model = hub.load('https://bit.ly/metrabs_l')  # Takes about 3 minutes
    return load_metrabs.model

load_metrabs.model = None