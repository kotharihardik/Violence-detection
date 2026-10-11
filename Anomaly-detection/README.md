# Vision Transformers for Violence Detection

An end-to-end, highly optimized PyTorch pipeline for training and deploying Violence Detection models using Vision Transformers (specifically **Meta's DINOv2**).

This repository was completely refactored to support blazing-fast feature extraction, rigorous data-leakage prevention, and real-time inference on raw video files.

## Model Architecture
1. **Spatial Feature Extractor:** Meta's `DINOv2` (`vitb14`) is used to extract 768-dimensional embeddings from video frames.
2. **Temporal Classifier:** A custom 2-layer `Temporal Transformer` processes sliding windows (32 frames) of DINOv2 embeddings to classify the sequence as **Normal (0)** or **Violence (1)**.

## Project Structure
- `config.py` - Centralized settings for hyperparameters, batch sizes, and directory paths.
- `utils/preprocessing/` - Scripts to safely extract frames, remove identical duplicates via pHash, and batch-extract DINOv2 features on the GPU.
- `models/` - PyTorch definitions for the Spatial Extractor and Temporal Transformer.
- `train/` - PyTorch training loop utilizing DataLoaders, Early Stopping, and best-model checkpointing.
- `inference.py` - Standalone script to run violence detection on raw `.mp4` video files in real-time, outputting annotated bounding-box videos.

## How to Run

### 1. Setup
Install the dependencies:
```bash
pip install -r requirements_pytorch.txt
```

### 2. Prepare Data
Download the UCF-Crime dataset and place the videos inside the `Data/` folder.

### 3. Build Dataset & Extract Features
Run the preprocessing pipeline in order:
```bash
python utils/preprocessing/conv_video_to_img.py
python utils/preprocessing/conv_img_to_npy.py
python utils/preprocessing/forward_pass_extr.py
```

### 4. Train the Model
```bash
python train/train.py
```
*The best model weights will be automatically saved to `train_output/models/best_model.pt`.*

### 5. Run Inference
To test your trained model on a brand new video, use the inference script:
```bash
python inference.py Data/my_test_video.mp4
```
*An annotated video (`output_annotated.mp4`) with bounding boxes and violence probabilities will be generated!*
