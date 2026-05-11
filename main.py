import torch
import torch.nn as nn
from torchvision import models, transforms
from fastapi import FastAPI, UploadFile, File
from PIL import Image
import io

app = FastAPI()

# 1. Recreate the Model Architecture
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = models.resnet18(weights=None)
model.fc = nn.Linear(model.fc.in_features, 7)
model.load_state_dict(torch.load("emotion_model_optimized.pth", map_location=device))
model.to(device)
model.eval()

class_names = ['angry', 'disgust', 'fear', 'happy', 'sad', 'surprise', 'neutral']

# 2. Image Preprocessing
def preprocess_image(image_bytes):
    transform = transforms.Compose([
        transforms.Grayscale(num_output_channels=1),
        transforms.Resize((48, 48)),
        transforms.ToTensor(),
        transforms.Lambda(lambda x: x.repeat(3, 1, 1)),
        transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
    ])
    image = Image.open(io.BytesIO(image_bytes))
    return transform(image).unsqueeze(0).to(device)

# 3. The /predict Endpoint
@app.post("/predict")
async def predict(file: UploadFile = File(...)):
    # Read file
    image_bytes = await file.read()
    input_tensor = preprocess_image(image_bytes)
    
    # Inference
    with torch.no_grad():
        outputs = model(input_tensor)
        probabilities = torch.nn.functional.softmax(outputs, dim=1)[0]
    
    # Format Response
    all_probs = {class_names[i]: float(probabilities[i]) for i in range(len(class_names))}
    dominant_emotion = class_names[torch.argmax(probabilities).item()]
    
    return {
        "dominant_emotion": dominant_emotion,
        "all_probabilities": all_probs
    }