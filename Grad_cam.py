import torch
from torchvision.models import resnet18
from torchvision import transforms
from torch.autograd import Variable
import cv2
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image

# Charger le modèle CNN pré-entraîné
model = resnet18(pretrained=True)
model.eval()

# Fonction Grad-CAM
def generate_grad_cam(model, img, target_layer):
    def hook_fn(module, input, output):
        features_blobs.append(output.data.cpu().numpy())
    
    features_blobs = []
    handle = model._modules.get(target_layer).register_forward_hook(hook_fn)

    output = model(img)
    handle.remove()
    
    output.backward()
    gradients = model._modules.get(target_layer).weight.grad.cpu().data.numpy()
    
    weights = np.mean(gradients, axis=(2, 3))[0, :]
    cam = np.zeros(features_blobs[0].shape[2:], dtype=np.float32)
    
    for i, w in enumerate(weights):
        cam += w * features_blobs[0][0, i, :, :]
    
    cam = np.maximum(cam, 0)
    cam = cv2.resize(cam, (224, 224))
    cam = cam - np.min(cam)
    cam = cam / np.max(cam)
    
    return cam

# Prétraitement de l'image
transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Resize((224, 224)),
    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
])
image = Image.open('path_to_image.jpg')
image = transform(image).unsqueeze(0)
image = Variable(image, requires_grad=True)

# Générer Grad-CAM
grad_cam = generate_grad_cam(model, image, 'layer4')

# Visualisation
plt.imshow(grad_cam, cmap='jet', alpha=0.5)
plt.show()
