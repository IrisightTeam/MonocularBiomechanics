import cv2
import numpy as np
from tqdm import tqdm
import tensorflow_hub as hub
import os

def jax_memory_limit():
    import os
    os.environ["XLA_PYTHON_CLIENT_PREALLOCATE"] = "false"

def tensorflow_memory_limit():
    # limit tensorflow memory. there are also other approaches
    # https://www.tensorflow.org/guide/gpu#limiting_gpu_memory_growth
    import tensorflow as tf

    gpus = tf.config.list_physical_devices("GPU")
    if gpus:
        try:
            # Currently, memory growth needs to be the same across GPUs
            for gpu in gpus:
                tf.config.experimental.set_memory_growth(gpu, True)
            logical_gpus = tf.config.list_logical_devices("GPU")
            print(len(gpus), "Physical GPUs,", len(logical_gpus), "Logical GPUs")
        except RuntimeError as e:
            # Memory growth must be set before GPUs have been initialized
            print(e)

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

def load_metrabs(path:str | None = None):
    if load_metrabs.model is not None:
        return load_metrabs.model
    if path is not None and os.path.exists(path):
        load_metrabs.model = hub.load(path)
        return load_metrabs.model
    load_metrabs.model = hub.load('https://bit.ly/metrabs_l')  # Takes about 3 minutes
    if path is not None:
        # save a local copy to keep from waiting as long
        tf.saved_model.save(load_metrabs.model, path)
    return load_metrabs.model

load_metrabs.model = None

joint_names = [
    "backneck",
    "upperback",
    "clavicle",
    "sternum",
    "umbilicus",
    "lfronthead",
    "lbackhead",
    "lback",
    "lshom",
    "lupperarm",
    "lelbm",
    "lforearm",
    "lwrithumbside",
    "lwripinkieside",
    "lfin",
    "lasis",
    "lpsis",
    "lfrontthigh",
    "lthigh",
    "lknem",
    "lankm",
    "LHeel",
    "lfifthmetatarsal",
    "LBigToe",
    "lcheek",
    "lbreast",
    "lelbinner",
    "lwaist",
    "lthumb",
    "lfrontinnerthigh",
    "linnerknee",
    "lshin",
    "lfirstmetatarsal",
    "lfourthtoe",
    "lscapula",
    "lbum",
    "rfronthead",
    "rbackhead",
    "rback",
    "rshom",
    "rupperarm",
    "relbm",
    "rforearm",
    "rwrithumbside",
    "rwripinkieside",
    "rfin",
    "rasis",
    "rpsis",
    "rfrontthigh",
    "rthigh",
    "rknem",
    "rankm",
    "RHeel",
    "rfifthmetatarsal",
    "RBigToe",
    "rcheek",
    "rbreast",
    "relbinner",
    "rwaist",
    "rthumb",
    "rfrontinnerthigh",
    "rinnerknee",
    "rshin",
    "rfirstmetatarsal",
    "rfourthtoe",
    "rscapula",
    "rbum",
    "Head",
    "mhip",
    "CHip",
    "Neck",
    "LAnkle",
    "LElbow",
    "LHip",
    "LHand",
    "LKnee",
    "LShoulder",
    "LWrist",
    "LFoot",
    "RAnkle",
    "RElbow",
    "RHip",
    "RHand",
    "RKnee",
    "RShoulder",
    "RWrist",
    "RFoot",
]
