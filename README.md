# AI-Powered Facial Recognition Attendance System

# Overview
This is a lightweight and efficient AI-powered attendance system built to run on Raspberry Pi 4, combining real-time face detection, recognition, and local attendance logging with a user-friendly dashboard. The system is designed for offline use, low power, and limited-resource environments like classrooms or small office setups.

# Architecture
## 1. Face Detection

![image](https://github.com/user-attachments/assets/16b6fe2b-5b3c-4160-a135-c0583097e31b)

The MTCNN (Multi-task Cascaded Convolutional Neural Network) processes an input image through three sequential stages—P-Net, R-Net, and O-Net—to accurately detect faces and facial landmarks. First, the image is resized into an image pyramid to detect faces at multiple scales. The P-Net (Proposal Network) scans each scaled image and generates candidate face regions along with bounding box regression offsets. These candidates are then filtered using non-maximum suppression (NMS) to remove overlapping boxes. Next, the cropped regions from the original image are resized to 24×24 pixels and passed through the R-Net (Refine Network), which further eliminates false positives and refines the bounding boxes with greater accuracy. After another round of NMS, the surviving regions are resized to 48×48 pixels and passed through the O-Net (Output Network). O-Net performs final face classification, more precise bounding box regression, and predicts five facial landmarks (e.g., eyes, nose, mouth corners). The final output consists of well-localized face bounding boxes and their corresponding landmark positions, making MTCNN a powerful and efficient face detection pipeline, particularly well-suited for real-time and embedded applications.

![image](https://github.com/user-attachments/assets/db179110-054c-4a73-99e4-71229899ac82)

Dataset : [face-detection-dataset](https://www.kaggle.com/datasets/fareselmenshawii/face-detection-dataset) - Author : Fares Elmenshawii
## 2. Face Recognition
To utilize the pre-trained model, you can use [model_mobilefacenet.pth](https://github.com/luongdang1/Face-Recognition-Attendance-System/blob/main/model_mobilefacenet.pth)

![image](https://github.com/user-attachments/assets/e3b8ce42-8682-4d38-9456-01c46228275f)

After the face has been detected and aligned by MTCNN, the cropped facial region is resized (typically to 112×112 pixels) and passed to MobileFaceNet, a lightweight convolutional neural network specifically designed for real-time face recognition on mobile and embedded devices. MobileFaceNet processes the input image and outputs a 128-dimensional embedding vector that compactly represents the identity-specific features of the face. This vector is then L2-normalized to ensure that all embeddings lie on a unit hypersphere, making distance comparisons more stable and consistent.

To recognize the person, the generated embedding is compared with a database of stored embeddings representing known individuals. The comparison is typically done using cosine similarity, which measures the angle between vectors: the closer the value is to 1, the more similar the faces are. If the similarity exceeds a predefined threshold, the system classifies the face as belonging to a known individual; otherwise, it is marked as "unknown." This embedding-based matching process is robust to variations in lighting, facial expression, and partial occlusions, enabling accurate real-time face recognition even on resource-constrained hardware like the Raspberry Pi 4.
Matches embeddings via cosine similarity with fusion techniques for robustness against lighting and pose variations.

![image](https://github.com/user-attachments/assets/f0154b83-5b87-4b33-8644-30e93d7b2ffd)

To improve the accuracy of face recognition, the system uses ArcFace loss during the training of MobileFaceNet. ArcFace introduces an angular margin to the softmax loss, forcing embeddings of the same identity to cluster closely while separating different identities more distinctly on a hypersphere. This leads to highly discriminative and robust embeddings, making the system more reliable under variations in lighting, pose, and expression.

Dataset : [VN-celeb](https://www.kaggle.com/datasets/duypok/vn-celeb) - Author : DuyNguyễnDươngHoàng
## 3. Hardware Control (C/C++)
Handles GPIO and camera module integration using C/C++ for low-level operations.

# Quick start
You can refer to [this notebook](https://github.com/luongdang1/Face-Recognition-Attendance-System/blob/main/face_recognitionn.ipynb) to easily learn how to use it
# References 
[1] [Sheng Chen1,2, Yang Liu2, Xiang Gao2, and Zhen Han1 “ MobileFaceNets: “ Efficient 
CNNs for Accurate Real- Time Face Verification on Mobile Devices”](https://arxiv.org/pdf/1804.07573)

[2] [Ziping Yu1 , Hongbo Huang∗2 , Weijun Chen3 , Yongxin Su4 , Yahui Liu5 , and 
Xiuying Wang2 “YOLO-FaceV2: A Scale and Occlusion Aware Face Detector”](https://arxiv.org/pdf/2208.02019)

[3] [Luis Vilaca1,2, Paula Viana1,2, Pedro Calvaho1,2, and Maria Terasa Andrade1,3 
“Improving Efficiency in Facial Recognition Tasks Through a Dataset Optimization 
Approach “](https://ieeexplore.ieee.org/stamp/stamp.jsp?arnumber=10452341)

[4] [Anjith George, Member, Christophe Ecabert, Member “EdgeFace: Efficient Face 
Recognition Model for Edge Devices”](https://arxiv.org/pdf/2307.01838)

[5] [Muhammad Zeeshan Khan, Saad Harous, Saleet Ul Hassan, Muhammad Usman 
Ghani Khan, Razi Iqba “Deep Unified Model For Face Recognition Based on 
Convolution Neural Network and Edge Computing”](https://ieeexplore.ieee.org/stamp/stamp.jsp?arnumber=8721062)

[6] [Jiankang Deng, Jia Guo, Jing Yang, Niannan Xue, Irene Kotsia, and Stefanos 
Zafeiriou “ArcFace: Additive Angular Margin Loss for Deep Face Recognition”](https://arxiv.org/pdf/1801.07698)
# License
The project is released under the MIT License.

