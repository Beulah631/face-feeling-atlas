# Model Weights Directory

Place your trained model weight files in this directory:

1. **YOLOv8 Face Detection:**
   - File name: `yolov8n-face.pt`
   - Purpose: Locates face bounding boxes in the input frame.
   - Pre-trained/fine-tuned YOLOv8 face weights.

2. **Swin Transformer FER Classifier (Proposed Model):**
   - File name: `swin_transformer_fer.pt`
   - Purpose: Extracts windowed self-attention features and classifies face into 7 emotions (224x224 RGB).

3. **CNN Baseline Classifier (Comparison Baseline):**
   - File name: `cnn_baseline_fer.pt`
   - Purpose: Base paper comparison model (e.g. 48x48 FER2013 baseline).

> **Note:** The backend does NOT automatically download unknown model weights from the internet. When weights are placed in this folder and the backend restarts, the models are automatically detected and loaded into memory.
