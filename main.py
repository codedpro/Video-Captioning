from fastapi import FastAPI, File, UploadFile
from fastapi.responses import JSONResponse
import av
import numpy as np
from PIL import Image
from huggingface_hub import hf_hub_download
from transformers import AutoProcessor, AutoModelForCausalLM

app = FastAPI()

processor = AutoProcessor.from_pretrained("microsoft/git-base-vatex")
model = AutoModelForCausalLM.from_pretrained("microsoft/git-base-vatex")

# set seed for reproducability
np.random.seed(45)


def read_video_pyav(container, indices):
    '''
    Decode the video with PyAV decoder.
    Args:
        container (`av.container.input.InputContainer`): PyAV container.
        indices (`List[int]`): List of frame indices to decode.
    Returns:
        result (np.ndarray): np array of decoded frames of shape (num_frames, height, width, 3).
    '''
    frames = []
    container.seek(0)
    start_index = indices[0]
    end_index = indices[-1]
    for i, frame in enumerate(container.decode(video=0)):
        if i > end_index:
            break
        if i >= start_index and i in indices:
            frames.append(frame)
    return np.stack([x.to_ndarray(format="rgb24") for x in frames])


def sample_frame_indices(clip_len, frame_sample_rate, seg_len):
    '''
    Sample a given number of frame indices from the video.
    Args:
        clip_len (`int`): Total number of frames to sample.
        frame_sample_rate (`int`): Sample every n-th frame.
        seg_len (`int`): Maximum allowed index of sample's last frame.
    Returns:
        indices (`List[int]`): List of sampled frame indices
    '''
    converted_len = int(clip_len * frame_sample_rate)
    end_idx = np.random.randint(converted_len, seg_len)
    start_idx = end_idx - converted_len
    indices = np.linspace(start_idx, end_idx, num=clip_len)
    indices = np.clip(indices, start_idx, end_idx - 1).astype(np.int64)
    return indices


@app.post("/caption_video/")
async def caption_video(file: UploadFile = File(...)):
    try:
        # save video file
        with open(file.filename, "wb") as buffer:
            buffer.write(await file.read())

        # load video
        container = av.open(file.filename)

        # sample frames
        num_frames = model.config.num_image_with_embedding
        indices = sample_frame_indices(
            clip_len=num_frames, frame_sample_rate=4, seg_len=container.streams.video[0].frames
        )
        frames = read_video_pyav(container, indices)

        pixel_values = processor(images=list(frames), return_tensors="pt").pixel_values

        generated_ids = model.generate(pixel_values=pixel_values, max_length=50)

        captions = processor.batch_decode(generated_ids, skip_special_tokens=True)

        return JSONResponse(content={"captions": captions})

    except Exception as e:
        return JSONResponse(content={"error": str(e)})
