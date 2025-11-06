Video Object Matcher - Streamlit App
Components: YOLOv8 (ultralytics), ByteTrack (via YOLOv8 tracker), CLIP (OpenAI), cosine similarity, LangChain (simple summarization)
Single-file app that: accepts two videos, detects & tracks objects (shoes), computes tracklet embeddings with CLIP, compares tracklets, and outputs whether videos share the same object.

Instructions
-----------
1. Create a Python 3.8+ virtualenv.
2. Install dependencies (GPU recommended for speed):

pip install streamlit ultralytics opencv-python-headless torch torchvision git+https://github.com/openai/CLIP.git scikit-learn faiss-cpu langchain sentence_transformers

Note: If you have a CUDA GPU and want faiss-gpu and torch with CUDA, install appropriate wheels.

3. Run the app:

streamlit run video_object_matcher_streamlit_app.py

Notes/Warnings
--------------
- ultralytics' YOLOv8 provides a Python track API that can use ByteTrack (tracker configs). This script uses model.track(). If you encounter tracker config issues, make sure your ultralytics version is up-to-date.
- ByteTrack and heavy models are faster on GPU. CPU will work but be slow.
