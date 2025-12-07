import os
import cv2
import numpy as np
import torch
import torchvision.transforms as T
from torchvision.models import efficientnet_b3, EfficientNet_B3_Weights
from PIL import Image
import clip
from sklearn.ensemble import GradientBoostingClassifier, VotingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.calibration import CalibratedClassifierCV
from sklearn.metrics import classification_report, confusion_matrix, ConfusionMatrixDisplay, roc_auc_score
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
from scipy.fft import fft2, fftshift
from skimage.feature import local_binary_pattern
from facenet_pytorch import MTCNN
import warnings
warnings.filterwarnings("ignore")


REAL_PATH = 'dataset/real/'
AI_PATH   = 'dataset/ai/'
IMG_SIZE = 380
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {DEVICE}")


print("Loading models...")
effnet = efficientnet_b3(weights=EfficientNet_B3_Weights.IMAGENET1K_V1).to(DEVICE)
effnet.eval()
effnet_features = torch.nn.Sequential(*list(effnet.children())[:-1])

clip_model, clip_preprocess = clip.load("ViT-L/14", device=DEVICE)
clip_model.eval()

mtcnn = MTCNN(select_largest=False, device=DEVICE, keep_all=True)

transform = T.Compose([
    T.Resize((IMG_SIZE, IMG_SIZE)),
    T.ToTensor(),
    T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
])


def safe_cnn_features(img_pil):
    try:
        x = transform(img_pil).unsqueeze(0).to(DEVICE)
        with torch.no_grad():
            feat = effnet_features(x)
            feat = torch.nn.functional.adaptive_avg_pool2d(feat, 1).flatten().cpu().numpy()
        return feat
    except:
        return np.zeros(1536, dtype=np.float32)

def safe_clip_features(img_pil):
    try:
        x = clip_preprocess(img_pil).unsqueeze(0).to(DEVICE)
        with torch.no_grad():
            feat = clip_model.encode_image(x).float().cpu().numpy().flatten()
        return feat
    except:
        return np.zeros(768, dtype=np.float32)

def safe_frequency_features(img_cv):
    try:
        gray = cv2.cvtColor(img_cv, cv2.COLOR_BGR2GRAY).astype(np.float32)
        f = fft2(gray)
        fshift = fftshift(f)
        magnitude = 20 * np.log(np.abs(fshift) + 1e-8)
        h, w = gray.shape
        cy, cx = h//2, w//2
        center_std = np.std(magnitude[cy-20:cy+20, cx-20:cx+20]) if h > 40 and w > 40 else 0
        return np.array([np.mean(magnitude), np.std(magnitude), center_std], dtype=np.float32)
    except:
        return np.zeros(3, dtype=np.float32)

def safe_noise_residual(img_cv):
    try:
        gray = cv2.cvtColor(img_cv, cv2.COLOR_BGR2GRAY).astype(np.float32)
        kernel = np.array([[0,-1,0],[-1,4,-1],[0,-1,0]], dtype=np.float32)
        residual = cv2.filter2D(gray, -1, kernel)
        return np.array([residual.mean(), residual.std(), np.median(np.abs(residual))], dtype=np.float32)
    except:
        return np.zeros(3, dtype=np.float32)

def safe_lbp(img_gray):
    try:
        lbp = local_binary_pattern(img_gray, P=24, R=3, method='uniform')
        hist, _ = np.histogram(lbp.ravel(), bins=26, range=(0, 26), density=True)
        return hist.astype(np.float32)
    except:
        return np.zeros(26, dtype=np.float32)

def safe_face_forensics(img_pil, img_cv):
    try:
        boxes, probs = mtcnn.detect(img_pil)
        if boxes is not None and len(boxes) > 0:
            # Take largest face
            areas = [(b[2]-b[0])*(b[3]-b[1]) for b in boxes]
            box = boxes[np.argmax(areas)]
            x1, y1, x2, y2 = map(int, box)
            x1, y1 = max(0, x1), max(0, y1)
            x2, y2 = min(img_cv.shape[1], x2), min(img_cv.shape[0], y2)
            if x2 <= x1 or y2 <= y1:
                raise ValueError("Invalid box")
            face = img_cv[y1:y2, x1:x2]
            if face.size == 0:
                raise ValueError("Empty face")
            
            # Eye symmetry (safe split)
            gray = cv2.cvtColor(face, cv2.COLOR_BGR2GRAY)
            h, w = gray.shape
            mid = w // 2
            left = gray[:, :mid]
            right = gray[:, mid:mid*2] if mid*2 <= w else gray[:, mid:]
            min_w = min(left.shape[1], right.shape[1])
            left = left[:, :min_w]
            right = cv2.flip(right[:, :min_w], 1)
            eye_sym = np.mean(np.abs(left.astype(float) - right.astype(float)))
            
            # Teeth whiteness
            hsv = cv2.cvtColor(face, cv2.COLOR_BGR2HSV)
            teeth_ratio = np.sum(hsv[:,:,2] > 240) / face.size
            
            return np.array([eye_sym / 255.0, teeth_ratio, len(boxes)], dtype=np.float32)
        else:
            return np.array([0.0, 0.0, 0.0], dtype=np.float32)
    except:
        return np.array([0.0, 0.0, 0.0], dtype=np.float32)

def extract_all_features_safe(img_path):
    try:
        img_cv = cv2.imread(img_path)
        if img_cv is None:
            return None
        img_cv = cv2.cvtColor(img_cv, cv2.COLOR_BGR2RGB)
        img_pil = Image.fromarray(img_cv)
        img_gray = cv2.cvtColor(img_cv, cv2.COLOR_RGB2GRAY)

        feats = np.concatenate([
            safe_cnn_features(img_pil),           # 1536
            safe_clip_features(img_pil),          # 768
            safe_frequency_features(img_cv),      # 3
            safe_noise_residual(img_cv),          # 3
            safe_lbp(img_gray),                   # 26
            safe_face_forensics(img_pil, cv2.cvtColor(img_cv, cv2.COLOR_RGB2BGR)),  # 3
        ])
        return feats
    except Exception as e:
        print(f"Critical error on {img_path}: {e}")
        return None

print("Loading dataset with bulletproof feature extraction...")
X_list, y_list = [], []

for path, label in [(REAL_PATH, 0), (AI_PATH, 1)]:
    folder = REAL_PATH if label == 0 else AI_PATH
    print(f"Processing {'REAL' if label==0 else 'AI'} folder...")
    for i, file in enumerate(os.listdir(folder)):
        if file.lower().endswith(('.png', '.jpg', '.jpeg', '.bmp', '.tiff', '.webp')):
            full_path = os.path.join(folder, file)
            feats = extract_all_features_safe(full_path)
            if feats is not None:
                X_list.append(feats)
                y_list.append(label)
        if (i+1) % 500 == 0:
            print(f"   Processed {i+1} images...")

X = np.array(X_list)
y = np.array(y_list)
print(f"Successfully loaded {len(X)} images with {X.shape[1]} features each")

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)

print("Training ultimate ensemble...")
gbt = CalibratedClassifierCV(
    GradientBoostingClassifier(n_estimators=600, learning_rate=0.05, max_depth=6, subsample=0.8, random_state=42),
    method='sigmoid', cv=3
)
lr = LogisticRegression(max_iter=1000, C=0.5, class_weight='balanced')

ensemble = VotingClassifier([('gbt', gbt), ('lr', lr)], voting='soft')
ensemble.fit(X_train, y_train)

y_pred = ensemble.predict(X_test)
y_prob = ensemble.predict_proba(X_test)[:, 1]

print("\n" + "="*70)
print("FINAL RESULTS (ROBUST VERSION)")
print("="*70)
print(classification_report(y_test, y_pred, target_names=['REAL', 'AI-GENERATED'], digits=4))
print(f"ROC-AUC: {roc_auc_score(y_test, y_prob):.5f}")

# Confusion matrix
cm = confusion_matrix(y_test, y_pred)
ConfusionMatrixDisplay(cm, display_labels=['Real', 'AI']).plot(cmap='Blues')
plt.title("Confusion Matrix - Bulletproof Detector")
plt.show()

from sklearn.metrics import roc_curve, auc, precision_recall_curve, average_precision_score
import matplotlib.pyplot as plt


# ROC Curve
fpr, tpr, _ = roc_curve(y_test, y_prob)
roc_auc = auc(fpr, tpr)

plt.figure(figsize=(8, 6))
plt.plot(fpr, tpr, color='darkorange', lw=2, label=f'ROC Curve (AUC = {roc_auc:.4f})')
plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
plt.xlim([0.0, 1.0])
plt.ylim([0.0, 1.05])
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('Receiver Operating Characteristic (ROC) Curve')
plt.legend(loc="lower right")
plt.grid(True, alpha=0.3)
plt.show()

# Precision-Recall Curve
precision, recall, _ = precision_recall_curve(y_test, y_prob)
avg_precision = average_precision_score(y_test, y_prob)

plt.figure(figsize=(8, 6))
plt.plot(recall, precision, color='blue', lw=2, label=f'PR Curve (AP = {avg_precision:.4f})')
plt.fill_between(recall, precision, alpha=0.2, color='blue')
plt.xlabel('Recall')
plt.ylabel('Precision')
plt.title('Precision-Recall Curve')
plt.legend(loc="lower left")
plt.grid(True, alpha=0.3)
plt.xlim([0.0, 1.0])
plt.ylim([0.0, 1.05])
plt.show()