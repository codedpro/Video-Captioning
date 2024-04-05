# Video Captioning API

This project is a FastAPI-based API for generating captions for videos. It utilizes a pre-trained model from Hugging Face's model hub to generate captions for uploaded videos.

## Installation

1. Clone the repository:

    ```bash
    git clone https://github.com/codedpro/video-captioning-api.git
    ```

2. Navigate to the project directory:

    ```bash
    cd video-captioning-api
    ```

3. Install the required dependencies:

    ```bash
    pip install -r requirements.txt
    ```

## About

This API generate captions for uploaded videos. It follows a simple workflow:

1. **Video Upload**: Users can upload a video file.
2. **Video Processing**: The uploaded video is processed to extract frames.
3. **Caption Generation**: The extracted frames are used as input to a pre-trained model to generate captions.
4. **Response**: The generated captions are returned as JSON response.

The model used for caption generation is `microsoft/git-base-vatex` from the Hugging Face model hub.

## Usage

To use this API, send a POST request to the `/caption_video/` endpoint with the video file as form data. Here's an example using `curl`:

```bash
curl -X 'POST' \
  'http://localhost:8000/caption_video/' \
  -H 'accept: application/json' \
  -H 'Content-Type: multipart/form-data' \
  -F 'file=@/path/to/your/video.mp4'
```

Replace `/path/to/your/video.mp4` with the path to your video file.

## API Endpoints

### `/caption_video/` [POST]

- **Description**: Generate captions for the uploaded video.
- **Request Body**: Form data with a single field `file` containing the video file.
- **Response**: JSON response containing the generated captions.

## Credits

- [FastAPI](https://fastapi.tiangolo.com/)
- [PyAV](https://github.com/mikeboers/PyAV)
- [Hugging Face Transformers](https://huggingface.co/transformers)

## Dev

This Project Developed by [@Coded_Pro](t.me/coded_pro)
