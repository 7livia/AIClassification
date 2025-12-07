import os
import cv2
from PIL import Image
import torch
from transformers import AutoImageProcessor, ViTForImageClassification
import warnings
warnings.filterwarnings("ignore")

# Put your image path here 
IMAGE_PATH = r"C:\Users\PC\Desktop\path\path\test_images\test1.jpg"  

# If the file doesn't exist, try common variations
if not os.path.exists(IMAGE_PATH):
    possibilities = [
        IMAGE_PATH.replace("\\", "/"),
        os.path.join("test_images", "test1.jpg"),
        os.path.join("dataset", "test_images", "test1.jpg"),
        "test1.jpg"
    ]
    for p in possibilities:
        if os.path.exists(p):
            IMAGE_PATH = p
            print(f"Found image at: {p}")
            break
    else:
        print("Image not found anywhere! Please check the path.")
        print("Example: IMAGE_PATH = r'C:\\path\\to\\your\\image.jpg'")
        exit()


MODEL_NAME = "prithivMLmods/Deep-Fake-Detector-v2-Model"  # Latest 2025 model: 92%+ accuracy

print("Loading state-of-the-art deepfake detector from Hugging Face...")
processor = AutoImageProcessor.from_pretrained(MODEL_NAME)
model = ViTForImageClassification.from_pretrained(MODEL_NAME)

device = "cuda" if torch.cuda.is_available() else "cpu"
model.to(device)
model.eval()

print("Model loaded! Ready to detect deepfakes (SD3, Flux, DALL·E 3, Midjourney v6+)")


def predict_deepfake(image_path):
    # Load and preprocess image
    image = Image.open(image_path).convert("RGB")
    inputs = processor(images=image, return_tensors="pt").to(device)
    
    # Inference
    with torch.no_grad():
        outputs = model(**inputs)
        logits = outputs.logits
        probabilities = torch.nn.functional.softmax(logits, dim=-1)
    
    # Labels: 0 = Deepfake (AI), 1 = Real
    ai_prob = probabilities[0][0].item()  # Prob of fake/AI
    real_prob = probabilities[0][1].item()  # Prob of real
    
    return ai_prob, real_prob

# Run prediction
print(f"\nAnalyzing: {os.path.basename(IMAGE_PATH)}")
try:
    ai_score, real_score = predict_deepfake(IMAGE_PATH)
    
    print("\n" + "="*50)
    if ai_score > 0.90:
        print("🚨 HIGH CONFIDENCE: AI-GENERATED / DEEPFAKE")
    elif ai_score > 0.70:
        print("⚠️  LIKELY AI-GENERATED / DEEPFAKE")
    elif ai_score > 0.50:
        print("❓ Uncertain — possible AI or edited")
    elif real_score > 0.90:
        print("✅ HIGH CONFIDENCE: REAL IMAGE")
    else:
        print("ℹ️  Likely real, but check manually")
    
    print(f"AI/Deepfake probability: {ai_score:.4f}")
    print(f"Real probability:        {real_score:.4f}")
    print(f"Model Accuracy (on test): ~92% (trained on 56k+ images)")
    print("="*50)
    
except Exception as e:
    print("Error during analysis:", e)
    print("Make sure the image is valid (JPG/PNG) and not corrupted.")
