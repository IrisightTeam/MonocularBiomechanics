import gradio as gr
import tensorflow as tf
import tensorflow_hub as hub
import os
from typing import List
from utils import video_reader,load_metrabs

def process_videos_with_metrabs(video_files: List[str], participant_id: str = "test", session_date: str = "2025-06-03", progress=gr.Progress()) -> str:
    """
    Process the uploaded videos. Replace this with your actual processing logic.
    """
    if not video_files:
        return "No videos uploaded."

    progress(0, desc="Loading model (takes 3 minutes)...")
    model = load_metrabs()
    model = hub.load('https://bit.ly/metrabs_l')  # Takes about 3 minutes
    progress(0.1, desc="Model loaded successfully.")
    skeleton = 'bml_movi_87'
    joint_names = model.per_skeleton_joint_names[skeleton].numpy().astype(str)
    joint_edges = model.per_skeleton_joint_edges[skeleton].numpy()
    
    results = []
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

    
    return f"Successfully processed {len(results)} videos:\n" + "\n".join(results)

def process_videos_with_biomechanics(video_files: List[str]) -> str:
    """
    Process the uploaded videos with biomechanics fitting. Replace this with your actual processing logic.
    """
    if not video_files:
        return "No videos uploaded."
    
    results = []
    for i, video_path in enumerate(video_files):
        if video_path is not None:
            filename = os.path.basename(video_path)
            file_size = os.path.getsize(video_path)
            results.append(f"Video {i+1}: {filename} (Size: {file_size:,} bytes)")
    
    return f"Successfully processed {len(results)} videos with biomechanics fitting:\n" + "\n".join(results)

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
