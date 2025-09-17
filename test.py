import torch
from ultralytics import YOLO
import networkx as nx
from torch_geometric.data import Data
from torch_geometric.nn import GCNConv  # Example GNN layer
import torch.nn.functional as F
from sklearn.metrics.pairwise import euclidean_distances
import numpy as np
from PIL import Image

# Step 1: Load the 3 YOLO models
# Assume model paths: replace with actual paths
vehicle_model = YOLO('vehicles.pt')  # YOLO model for vehicles/cars
water_model = YOLO('water.pt')      # YOLO model for water bodies
campfire_model = YOLO('wildfire.pt')  # YOLO model for campfires

# Step 2: Load the test image
# Assume test image path: replace with actual path
test_image_path = 'testimage.jpg'
image = Image.open(test_image_path)
image_np = np.array(image)  # For later use if needed

# Step 3: Run inference on each YOLO model
def run_yolo(model, image_path):
    results = model(image_path)
    detections = []
    for result in results:
        for box in result.boxes:
            cls = int(box.cls[0])
            conf = float(box.conf[0])
            xyxy = box.xyxy[0].cpu().numpy()  # Bounding box [x1, y1, x2, y2]
            center = [(xyxy[0] + xyxy[2]) / 2, (xyxy[1] + xyxy[3]) / 2]  # Center point
            detections.append({
                'class': model.names[cls],
                'conf': conf,
                'center': center,
                'bbox': xyxy
            })
    return detections

vehicle_dets = run_yolo(vehicle_model, test_image_path)
water_dets = run_yolo(water_model, test_image_path)
campfire_dets = run_yolo(campfire_model, test_image_path)

# Combine all detections
all_detections = vehicle_dets + water_dets + campfire_dets
print("Detections:", all_detections)  # For debugging

# Step 4: Create a graph from detections
# Nodes: Each detection is a node with features (e.g., class one-hot, confidence, normalized position)
# Edges: Connect nodes if they are within a certain distance (e.g., proximity indicates relation)

# Define class mapping (assume classes: 'vehicle', 'water', 'campfire')
class_to_idx = {'vehicle': 0, 'water': 1, 'campfire': 2}
num_classes = len(class_to_idx)

# Node features: [one-hot class, confidence, norm_x, norm_y]
nodes = []
positions = []
for det in all_detections:
    one_hot = [0] * num_classes
    one_hot[class_to_idx.get(det['class'], 0)] = 1  # Default to 0 if unknown
    norm_x = det['center'][0] / image.width
    norm_y = det['center'][1] / image.height
    features = one_hot + [det['conf'], norm_x, norm_y]
    nodes.append(features)
    positions.append(det['center'])

if not nodes:
    print("No detections found. Poaching confidence: 0")
    exit()

node_features = torch.tensor(nodes, dtype=torch.float)

# Create edges based on distance (threshold, e.g., 0.1 * image diagonal)
diag = np.sqrt(image.width**2 + image.height**2)
threshold = 0.1 * diag
pos_array = np.array(positions)
dists = euclidean_distances(pos_array)

G = nx.Graph()
for i in range(len(positions)):
    G.add_node(i, features=node_features[i])

for i in range(len(positions)):
    for j in range(i + 1, len(positions)):
        if dists[i, j] < threshold:
            G.add_edge(i, j)

# Convert to torch_geometric Data
edge_index = torch.tensor(list(G.edges)).t().contiguous() if G.edges else torch.empty((2, 0), dtype=torch.long)
x = node_features  # Node features

graph_data = Data(x=x, edge_index=edge_index)

# Step 5: Define a simple pretrained GNN model (example: GCN for binary classification)
# Assume it's a binary classifier: poaching (1) or not (0)
# In reality, load your pretrained model weights

class SimpleGNN(torch.nn.Module):
    def __init__(self, input_dim, hidden_dim=16, output_dim=1):
        super(SimpleGNN, self).__init__()
        self.conv1 = GCNConv(input_dim, hidden_dim)
        self.conv2 = GCNConv(hidden_dim, hidden_dim)
        self.fc = torch.nn.Linear(hidden_dim, output_dim)

    def forward(self, data):
        x, edge_index = data.x, data.edge_index
        x = F.relu(self.conv1(x, edge_index))
        x = F.relu(self.conv2(x, edge_index))
        x = torch.mean(x, dim=0)  # Global mean pooling
        x = self.fc(x)
        return torch.sigmoid(x)  # Confidence score between 0 and 1

# Input dim: num_classes + 3 (conf + norm_x + norm_y)
input_dim = num_classes + 3
gnn_model = SimpleGNN(input_dim)
# Load pretrained weights (replace with actual loading)
# gnn_model.load_state_dict(torch.load('pretrained_gnn.pt'))
gnn_model.eval()  # Set to evaluation mode

# Run GNN on the graph
with torch.no_grad():
    confidence = gnn_model(graph_data).item()

print(f"Poaching Confidence: {confidence:.4f}")