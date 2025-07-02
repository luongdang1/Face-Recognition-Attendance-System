AI-Powered Facial Recognition Attendance System
📅 February 2025 – March 2025
🔧 Technologies: Python, C/C++, MTCNN, MobileFaceNet, SQLite, Flask, Raspberry Pi 4

# 📌 Overview
This is a lightweight and efficient AI-powered attendance system built to run on Raspberry Pi 4, combining real-time face detection, recognition, and local attendance logging with a user-friendly dashboard. The system is designed for offline use, low power, and limited-resource environments like classrooms or small office setups.

# 🧠 Architecture
  ## 1. Face Detection

[1] ![image](https://github.com/user-attachments/assets/16b6fe2b-5b3c-4160-a135-c0583097e31b)

The MTCNN (Multi-task Cascaded Convolutional Neural Network) processes an input image through three sequential stages—P-Net, R-Net, and O-Net—to accurately detect faces and facial landmarks. First, the image is resized into an image pyramid to detect faces at multiple scales. The P-Net (Proposal Network) scans each scaled image and generates candidate face regions along with bounding box regression offsets. These candidates are then filtered using non-maximum suppression (NMS) to remove overlapping boxes. Next, the cropped regions from the original image are resized to 24×24 pixels and passed through the R-Net (Refine Network), which further eliminates false positives and refines the bounding boxes with greater accuracy. After another round of NMS, the surviving regions are resized to 48×48 pixels and passed through the O-Net (Output Network). O-Net performs final face classification, more precise bounding box regression, and predicts five facial landmarks (e.g., eyes, nose, mouth corners). The final output consists of well-localized face bounding boxes and their corresponding landmark positions, making MTCNN a powerful and efficient face detection pipeline, particularly well-suited for real-time and embedded applications.

![image](https://github.com/user-attachments/assets/db179110-054c-4a73-99e4-71229899ac82)

  ## 2. Face Recognition
To utilize the pre-trained model, you can use [model_mobilefacenet.pth]

[2] ![image](https://github.com/user-attachments/assets/e3b8ce42-8682-4d38-9456-01c46228275f)

After the face has been detected and aligned by MTCNN, the cropped facial region is resized (typically to 112×112 pixels) and passed to MobileFaceNet, a lightweight convolutional neural network specifically designed for real-time face recognition on mobile and embedded devices. MobileFaceNet processes the input image and outputs a 128-dimensional embedding vector that compactly represents the identity-specific features of the face. This vector is then L2-normalized to ensure that all embeddings lie on a unit hypersphere, making distance comparisons more stable and consistent.

To recognize the person, the generated embedding is compared with a database of stored embeddings representing known individuals. The comparison is typically done using cosine similarity, which measures the angle between vectors: the closer the value is to 1, the more similar the faces are. If the similarity exceeds a predefined threshold, the system classifies the face as belonging to a known individual; otherwise, it is marked as "unknown." This embedding-based matching process is robust to variations in lighting, facial expression, and partial occlusions, enabling accurate real-time face recognition even on resource-constrained hardware like the Raspberry Pi 4.
Matches embeddings via cosine similarity with fusion techniques for robustness against lighting and pose variations.

[3] ![image](https://github.com/user-attachments/assets/f0154b83-5b87-4b33-8644-30e93d7b2ffd)

To improve the accuracy of face recognition, the system uses ArcFace loss during the training of MobileFaceNet. ArcFace introduces an angular margin to the softmax loss, forcing embeddings of the same identity to cluster closely while separating different identities more distinctly on a hypersphere. This leads to highly discriminative and robust embeddings, making the system more reliable under variations in lighting, pose, and expression.


  ## 3. Hardware Control (C/C++)
Handles GPIO and camera module integration using C/C++ for low-level operations.

# Quick start
You can refer to [this notebook](https://github.com/luongdang1/ASR-in-Smart-Home/blob/main/asr_speech_recognition.ipynb) to easily learn how to use it
# References 
"Some of the papers I referred to include:"

Optimizing Speech Recognition for the Edge — [arXiv:1909.12408](https://arxiv.org/abs/1909.12408)

Tiny Transducer: A Highly Efficient Speech Recognition Model for Edge Devices — [arXiv:2101.06856](https://arxiv.org/pdf/2101.06856)

Conformer-Based Speech Recognition on Extreme Edge-Computing Devices — [arXiv:2312.10359](https://arxiv.org/pdf/2312.10359)

https://nvidia.github.io/OpenSeq2Seq/html/speech-recognition/deepspeech2.html [1]

https://www.researchgate.net/figure/DeepSpeech-2-architecture41_fig23_348706070 [2]

https://deepspeech.readthedocs.io/en/v0.6.1/DeepSpeech.html [3]

https://ieeexplore.ieee.org/stamp/stamp.jsp?tp=&arnumber=10966154 [4]

https://www.kaggle.com/datasets/tommyngx/fluent-speech-corpus [5]

https://github.com/mozilla/DeepSpeech 
# License
The project is released under the MIT License.

