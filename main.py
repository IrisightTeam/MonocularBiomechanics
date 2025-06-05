from pose_pipeline.env import tensorflow_memory_limit, jax_memory_limit
tensorflow_memory_limit()
jax_memory_limit()
import gradio as gr
import jax
import tensorflow as tf
import tensorflow_hub as hub
import numpy as np
import jax.numpy as jnp
from body_models.datajoint.monocular_dj import get_samsung_calibration, MonocularDataset, get_model, fit_model
import os
from typing import List
from utils import video_reader,load_metrabs
from pose_pipeline.wrappers.bridging import get_model as get_metrabs_model

jax.config.update("jax_compilation_cache_dir", "./jax_cache")
jax.config.update("jax_persistent_cache_min_entry_size_bytes", -1)
jax.config.update("jax_persistent_cache_min_compile_time_secs", 0)


joint_names = ['backneck', 'upperback', 'clavicle', 'sternum', 'umbilicus', 'lfronthead', 'lbackhead', 'lback',
 'lshom', 'lupperarm', 'lelbm', 'lforearm', 'lwrithumbside', 'lwripinkieside', 'lfin', 'lasis', 'lpsis', 'lfrontthigh', 'lthigh',
 'lknem', 'lankm', 'LHeel', 'lfifthmetatarsal', 'LBigToe', 'lcheek', 'lbreast', 'lelbinner', 'lwaist', 'lthumb', 'lfrontinnerthigh',
 'linnerknee', 'lshin', 'lfirstmetatarsal', 'lfourthtoe', 'lscapula', 'lbum', 'rfronthead', 'rbackhead', 'rback', 'rshom', 'rupperarm',
 'relbm', 'rforearm', 'rwrithumbside', 'rwripinkieside', 'rfin', 'rasis', 'rpsis', 'rfrontthigh', 'rthigh', 'rknem', 'rankm', 'RHeel',
 'rfifthmetatarsal', 'RBigToe', 'rcheek', 'rbreast', 'relbinner', 'rwaist', 'rthumb', 'rfrontinnerthigh', 'rinnerknee', 'rshin', 'rfirstmetatarsal',
 'rfourthtoe', 'rscapula', 'rbum', 'Head', 'mhip', 'CHip', 'Neck', 'LAnkle', 'LElbow', 'LHip', 'LHand', 'LKnee', 'LShoulder', 'LWrist', 'LFoot',
 'RAnkle', 'RElbow', 'RHip', 'RHand', 'RKnee', 'RShoulder', 'RWrist', 'RFoot']

def save_metrabs_data(accumulated, video_name):
    fname = video_name.split('/')[-1].split('.')[0]
    boxes, pose3d, pose2d, confs = [], [], [], []
    for i,(box, p3d, p2d)in enumerate(zip(accumulated['boxes'], accumulated['poses3d'], accumulated['poses2d'])):
        # TODO: write logic for better box tracking
        if len(box) == 0:
            boxes.append(np.zeros((5)))
            pose3d.append(np.zeros((87, 3)))
            pose2d.append(np.zeros((87, 2)))
            confs.append(np.zeros((87)))
            print("no boxes")
            continue
        boxes.append(box[0].numpy())
        pose3d.append(p3d[0].numpy())
        pose2d.append(p2d[0].numpy())
        confs.append(np.ones((87)))

        with open(f'{fname}_keypoints.npz', 'wb') as f:
            np.savez(f, pose3d=pose3d, pose2d=pose2d, boxes=boxes, confs=confs)

def get_framerate(video_path):
    """
    Get the framerate of a video file.
    """
    import cv2
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        raise ValueError(f"Could not open video file: {video_path}")
    fps = cap.get(cv2.CAP_PROP_FPS)
    cap.release()
    return fps

def load_metrabs_data(video_name):
    fname = video_name.split('/')[-1].split('.')[0]
    try:
        with open(f'{fname}_keypoints.npz', 'rb') as f:
            data = np.load(f, allow_pickle=True)
            boxes = data['boxes']
            keypoints2d = data['keypoints2d']
            keypoints3d = data['keypoints3d']
            confs = data['confs']


        return boxes, keypoints2d, keypoints3d, confs
    except FileNotFoundError:
        print("No saved data found for this video.")
        return None, None, None, None


def process_videos_with_metrabs(video_files: List[str], participant_id: str = "test", session_date: str = "2025-06-03", progress=gr.Progress()) -> str:
    """
    Process the uploaded videos. Replace this with your actual processing logic.
    """
    if not video_files:
        return "No videos uploaded."

    progress(0, desc="Loading model (takes 3 minutes)...")
    model = get_metrabs_model()
    progress(0.1, desc="Model loaded successfully.")
    skeleton = 'bml_movi_87'
    
    video_count = 0
    for video_idx, video_path in enumerate(video_files):
        if video_path is not None:
            vid, n_frames = video_reader(video_path)
            accumulated = None
            for frame_idx, frame_batch in enumerate(vid):
                progress(frame_idx * 8 / n_frames, desc=f"Processing video {video_idx+1}")

                pred = model.detect_poses_batched(frame_batch, skeleton=skeleton)

                if accumulated is None:
                    accumulated = pred

                else:
                    # concatenate the ragged tensor along the batch for each element in the dictionary
                    for key in accumulated.keys():
                        accumulated[key] = tf.concat([accumulated[key], pred[key]], axis=0)
            save_metrabs_data(accumulated, video_path)
            video_count += 1
    
    return f"Successfully processed {video_count} videos with Metrabs."


def process_videos_with_biomechanics(video_files: List[str], progress=gr.Progress()) -> str:
    """
    Process the uploaded videos with biomechanics fitting. Replace this with your actual processing logic.
    """

    max_iters = 6000 
    def step_callback(step, model, dataset, metrics_dict, **kwargs):
        if step % 500 == 0:
            progress(step / max_iters, desc=f"Fitting model: Step {step}/{max_iters}")
    
    if not video_files:
        return "No videos uploaded."
    
    timestamps_list = []
    keypoints2d_list = []
    keypoints3d_list = []
    confs_list = []
    for i, video_path in enumerate(video_files):
        if video_path is not None:
            boxes, keypoints2d, keypoints3d, confs = load_metrabs_data(video_path)
            if boxes is None:
                print(f"Video {video_path}: No Metrabs data found.")
                continue

            fps = get_framerate(video_path)
            timestamps = np.arange(0, len(keypoints2d)) / fps
            timestamps_list.append(timestamps)
            keypoints2d_list.append(keypoints2d[jnp.newaxis])  # Add camera dimension
            keypoints3d_list.append(keypoints3d[jnp.newaxis])  # Add camera dimension
            confs_list.append(jnp.ones_like(keypoints2d[...,0])[jnp.newaxis])  # fake confidences

    dataset = MonocularDataset(
        timestamps=timestamps_list,
        keypoints_2d=keypoints2d_list,
        keypoints_3d=keypoints3d_list,
        keypoint_confidence=confs_list,
        camera_params=get_samsung_calibration(),
        phone_attitude=None,
    )

    progress(0, desc="Building biomechanics model...")
    model = get_model(dataset, xml_path='humanoid/humanoid_torque.xml', joint_names=joint_names) # might need to change the site names
    model, metrics = fit_model(model, dataset, lr_init_value=1e-3, max_iters=max_iters, step_callback=step_callback)
    progress(1.0, desc="Biomechanics model fit successfully.")

    for i, video_path in enumerate(video_files):
        progress(i / len(video_files), desc=f"Processing video {i+1}/{len(video_files)} for biomechanics fitting")
        timestamps = dataset.get_all_timestamps(i)

        (state, _, _), (qpos, qvel, _), rnc = model(
            timestamps, trajectory_selection=i, steps=0, skip_action=True, fast_inference=True, check_constraints=False
        )

        # save zip archive
        fname = video_path.split('/')[-1].split('.')[0]
        with open(f'{fname}_fitted_model.npz', 'wb') as f:
            np.savez(f, qpos=np.array(qpos), qvel=np.array(qvel), rnc=np.array(rnc), sites=np.array(state.site_xpos), joints=np.array(state.xpos))

    
    return f"Successfully processed {len(dataset)} videos with biomechanics fitting."

def clear_videos():
    """Clear the video upload component"""
    return None

# Create the Gradio interface
with gr.Blocks(title="Open Portable Biomechanics Lab") as demo:
    gr.Markdown("# Open Portable Biomechanics Lab")
    gr.Markdown("Upload multiple videos for processing. Supported formats: MP4, AVI, MOV, MKV")
    
    with gr.Row():
        with gr.Column():
            video_input = gr.File(
                label="Upload Videos",
                file_count="multiple",
                file_types=["video"],
                height=200
            )

            participant_id_input = gr.Textbox(
                label="Participant ID",
                placeholder="Type something here...",
                lines=1,
            )

            date_input = gr.Textbox(
                label="Session Date",
                placeholder="Type something here...",
                lines=1,
            )
            
            with gr.Row():
                metrabs_btn = gr.Button("1. Keypoint Detection", variant="primary")
                mjx_btn = gr.Button("2. Biomechanical Fitting", variant="primary")
        
        with gr.Column():
            output_text = gr.Textbox(
                label="Processing Results",
                lines=10,
                max_lines=20,
                interactive=False
            )
    
    # Event handlers
    metrabs_btn.click(
        fn=process_videos_with_metrabs,
        inputs=[video_input, participant_id_input, date_input],
        outputs=[output_text]
    )

    mjx_btn.click(
        fn=process_videos_with_biomechanics,
        inputs=[video_input],
        outputs=[output_text]
    )
    
    # Also process when files are uploaded
    video_input.change(
        fn=lambda files: f"Uploaded {len(files) if files else 0} videos. Click 'Process Videos' to continue.",
        inputs=[video_input],
        outputs=[output_text]
    )

demo.launch(share=True)
