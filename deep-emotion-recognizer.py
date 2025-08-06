# %%
import cv2 
import numpy as np
import sys
import os
import argparse
import albumentations as A
import torch
from torchvision import transforms
from torchvision.datasets import ImageFolder
from torch.utils.data import DataLoader
import torch.nn as nn
import torch.optim as optim
from PIL import Image
import torch.multiprocessing as mp


# %%
from facenet_pytorch import MTCNN, InceptionResnetV1
from PIL import Image
from deepface import DeepFace
from torch.utils.data import Dataset
import numpy as np
from deepface import DeepFace
import cv2


# define a global errorcount variable 
global errorcount
errorcount = 0

# %%
class EmotionDataset(Dataset):
    def __init__(self, root_dir,embedder, transform=None):
        self.root_dir = root_dir
        self.transform = transform
        self.classes = sorted(os.listdir(root_dir))
        self.image_paths = []
        self.labels = []
        self.errorcount = 0
        for idx, cls in enumerate(self.classes):
            cls_dir = os.path.join(root_dir, cls)
            for fname in os.listdir(cls_dir):
                if fname.lower().endswith(('png', 'jpg', 'jpeg')):
                    self.image_paths.append(os.path.join(cls_dir, fname))
                    self.labels.append(idx)
       
        self.embedder = embedder  # Keras model

    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, idx):

        # Add this in a TRy Except block to handle errors

        try:
            img_path = self.image_paths[idx]
            label = self.labels[idx]
            #print(f"Processing image {img_path} with label {label}")
            # Load grayscale then convert to RGB
            img = cv2.imread(img_path)

            if img is None:
                print("Error: Image not found or could not be loaded.")
            else:
                # Get the embedding using DeepFace.represent()
                # The 'model_name' parameter specifies which model to use.
                # 'VGG-Face' is the default and a good choice.
                # The 'detector_backend' parameter can be changed if you want a different face detector.
                # 'opencv', 'ssd', 'dlib', 'mtcnn', 'retinaface', 'mediapipe' are available.
                representations = DeepFace.represent(
                    img_path=img,
                    model_name='VGG-Face',
                    detector_backend='mtcnn', # A robust face detector
                )

                # The result is a list of dictionaries, one for each face detected in the image.
                # We'll assume there is only one face for this example.
                if representations:
                    embedding = representations[0]['embedding']
                  
                    # The size of the VGG-Face embedding is 4096
                    # This is a fixed-size vector representing the face.
                   # print("Embedding size:", len(embedding))
                else:
                    print("No face detected in the image.")
            # Convert to torch tensor
            embedding = torch.tensor(embedding, dtype=torch.float)
            label = torch.tensor(label, dtype=torch.long)
        
        except Exception as e:
           # print(f"Error processing image {img_path}: {e}")
            embedding = torch.zeros(4096, dtype=torch.float)  # Assuming 2622 is the embedding size
            label = 4 # Use 4 for Nuetral class or any other fallback label
            label = torch.tensor(label, dtype=torch.long)
            self.errorcount += 1
        
        # print(worker_info := f"Worker ID: {os.getpid()}, Image Path: {img_path}, Label: {label.item()}")
        print("Error count:", self.errorcount)
        return embedding, label
       


# %%
class EmotionClassifier(nn.Module):
    def __init__(self, embedding_size, num_classes):
        super().__init__()
        self.classifier = nn.Sequential(
            nn.Linear(embedding_size, 256),
            nn.ReLU(inplace=True),
            nn.Dropout(0.5),
            nn.Linear(256, num_classes)
        )

    def forward(self, x):
        return self.classifier(x)

# %%
def train_model(data_dir,
                num_epochs=3,
                batch_size=500,
                lr=1e-3,
                save_path='emotion_detector_deepface.pth'):
    # MTCNN for face detection
    # mtcnn = MTCNN(image_size=160, margin=0)
    # Load DeepFace embedding model
    print("Loading DeepFace embedding model...")

    embedder = DeepFace.build_model('VGG-Face') 
    # Dataset + DataLoader
    dataset = EmotionDataset(root_dir=data_dir,
                            #  mtcnn=mtcnn,
                             embedder=embedder)
    dataloader = DataLoader(dataset,
                            batch_size=batch_size,
                            shuffle=True,
                            num_workers=5)
    
    print(f"Dataset loaded with {len(dataset)} samples.")

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = EmotionClassifier(embedding_size=4096,  # VGG-Face embedding size
                              num_classes=len(dataset.classes))
    model.to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=lr)

    for epoch in range(num_epochs):
        model.train()
        running_loss = 0.0
        correct = 0
        total = 0
        for embeddings, labels in dataloader:
            embeddings = embeddings.to(device)
            labels = labels.to(device)

            optimizer.zero_grad()
            outputs = model(embeddings)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

            running_loss += loss.item() * embeddings.size(0)
            _, preds = torch.max(outputs, 1)
            correct += (preds == labels).sum().item()
            total += labels.size(0)

        epoch_loss = running_loss / total
        epoch_acc = correct / total
        print(f'Epoch {epoch+1}/{num_epochs} - Loss: {epoch_loss:.4f} - Acc: {epoch_acc:.4f}')

    # Save classifier state and classes
    torch.save({
        'model_state_dict': model.state_dict(),
        'class_names': dataset.classes
    }, save_path)
    print(f'Model saved to {save_path}')


# %%

if __name__ == "__main__":
        

        
        mp.set_start_method('spawn', force=True)  # Set multiprocessing start method
        data_dir = 'Dataset/images/train'
        num_epochs = 3
        batch_size = 1000
        lr = 1e-3   
        save_path = 'emotion_detector_deepface.pth'

        print(f"Training model with parameters:\n")
        train_model(
                data_dir=data_dir,
                num_epochs=num_epochs,
                batch_size=batch_size,
                lr=lr,
                save_path=save_path
            )

# %%


# # %%
# import torch
# import gc

# # 1. Delete all tensors and models from the GPU
# del EmotionClassifier
# del EmotionDataset
# del embedder
# del dataloader
# del model
# del criterion
# del optimizer


# # 2. Run the garbage collector to clean up references
# gc.collect()

# # 3. Clear the CUDA memory cache
# torch.cuda.empty_cache()

# torch.cuda.empty_cache()

# # %%
# dataset = EmotionDataset(
#     root_dir='Dataset/images/train',
#     embedder=DeepFace.build_model('VGG-Face')
# )

# # find the set of labels in the dataset
# labels = set(dataset.labels)
# print(f"Unique labels in the dataset: {labels}")

# # %%
# import numpy as np
# from deepface import DeepFace
# import cv2

# # Load an image using OpenCV
# # You can replace 'path/to/your/image.jpg' with the actual path to your image
# img = cv2.imread('Dataset/images/train/fear/50.jpg')  # Example image path

# # Check if the image was loaded successfully
# if img is None:
#     print("Error: Image not found or could not be loaded.")
# else:
#     # Get the embedding using DeepFace.represent()
#     # The 'model_name' parameter specifies which model to use.
#     # 'VGG-Face' is the default and a good choice.
#     # The 'detector_backend' parameter can be changed if you want a different face detector.
#     # 'opencv', 'ssd', 'dlib', 'mtcnn', 'retinaface', 'mediapipe' are available.
#     representations = DeepFace.represent(
#         img_path=img,
#         model_name='VGG-Face',
#         detector_backend='mtcnn'  # A robust face detector
#     )

#     # The result is a list of dictionaries, one for each face detected in the image.
#     # We'll assume there is only one face for this example.
#     if representations:
#         embedding = representations[0]['embedding']
#         print("Embedding shape:", np.array(embedding).shape)
#         print("Embedding (first 10 values):", embedding[:10])

#         # The size of the VGG-Face embedding is 2622
#         # This is a fixed-size vector representing the face.
#         print("Embedding size:", len(embedding))
#     else:
#         print("No face detected in the image.")


