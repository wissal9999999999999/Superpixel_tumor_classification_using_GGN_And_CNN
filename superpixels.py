import cv2
import numpy as np
import torch
import matplotlib.pyplot as plt
from skimage.segmentation import slic
from torchvision import models, transforms
from torch_geometric.data import Data

# Segmentation de l'image
def segment_image(image, n_segments=100):
    segments = slic(image, n_segments=n_segments, compactness=10, sigma=1)
    return segments

# Extraction des caractéristiques
def extract_features(cnn_model, image, segments):
    features = []
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Resize((224, 224)),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ])
    
    for seg_val in np.unique(segments):
        mask = np.zeros(image.shape[:2], dtype="uint8")
        mask[segments == seg_val] = 255
        masked_image = cv2.bitwise_and(image, image, mask=mask)
        pil_image = Image.fromarray(masked_image)
        input_tensor = transform(pil_image).unsqueeze(0)
        with torch.no_grad():
            feature = cnn_model(input_tensor)
        features.append(feature.squeeze(0).cpu().numpy())
    
    return np.array(features)

# Construction du graphe
def create_graph(segments, features):
    G = nx.Graph()
    num_segments = len(np.unique(segments))
    
    for i in range(num_segments):
        G.add_node(i, x=features[i])
    
    for i in range(num_segments):
        for j in range(i + 1, num_segments):
            if are_neighbors(segments, i, j):
                G.add_edge(i, j)
    
    edge_index = torch.tensor(list(G.edges)).t().contiguous()
    x = torch.tensor(features, dtype=torch.float)
    data = Data(x=x, edge_index=edge_index)
    
    return data

# Prédiction des superpixels
def predict_superpixels(model, data, device):
    model.eval()
    data = data.to(device)
    with torch.no_grad():
        out = model(data.x, data.edge_index)
    return out.argmax(dim=1).cpu().numpy()

# Visualisation des superpixels
def visualize_predictions(image, segments, predictions):
    unique_segments = np.unique(segments)
    result_image = np.zeros_like(image)
    
    for seg_val, prediction in zip(unique_segments, predictions):
        color = (0, 0, 255) if prediction == 1 else (0, 255, 0)
        mask = segments == seg_val
        result_image[mask] = color
    
    plt.figure(figsize=(10, 10))
    plt.imshow(result_image)
    plt.title("Superpixel Predictions")
    plt.show()

# Exemple d'exécution
image = cv2.imread('path_to_image')
image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
segments = segment_image(image, n_segments=100)

# Charger le modèle CNN pré-entraîné
cnn_model = models.resnet18(pretrained=True)
cnn_model = torch.nn.Sequential(*(list(cnn_model.children())[:-1]))
cnn_model.eval()

features = extract_features(cnn_model, image, segments)
data = create_graph(segments, features)

# Charger le modèle GNN pré-entraîné
gnn_model = torch.load('path_to_gnn_model.pt')
gnn_model = gnn_model.to(device)

predictions = predict_superpixels(gnn_model, data, device)
visualize_predictions(image, segments, predictions)
