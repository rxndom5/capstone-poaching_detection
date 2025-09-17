# gnn_aggregator_service.py
import json
from kafka import KafkaConsumer, KafkaProducer
import torch
import networkx as nx
from torch_geometric.data import Data
from torch_geometric.nn import GCNConv
import torch.nn.functional as F
from sklearn.metrics.pairwise import euclidean_distances
import numpy as np

# --- Kafka Configuration ---
KAFKA_BROKER = 'localhost:9092'
DETECTION_TOPIC = 'detection_results_topic'
SCORE_TOPIC = 'poaching_scores_topic'

# --- GNN and Graph Logic ---
DISTANCE_THRESHOLD = 150.0  # Pixels
CLASS_TO_ID = {'car': 0, 'water_body': 1, 'wildfire': 2, 'water': 1}
NUM_CLASSES = 3

class SimpleGNN(torch.nn.Module):
    def __init__(self, input_dim, hidden_dim=16, output_dim=1):
        super(SimpleGNN, self).__init__()
        self.conv1 = GCNConv(input_dim, hidden_dim)
        self.conv2 = GCNConv(hidden_dim, hidden_dim)
        self.fc = torch.nn.Linear(hidden_dim, output_dim)

    def forward(self, data):
        x, edge_index = data.x, data.edge_index
        if x is None or x.nelement() == 0:
            return torch.tensor([0.0])
        x = F.relu(self.conv1(x, edge_index))
        x = F.relu(self.conv2(x, edge_index))
        x = torch.mean(x, dim=0)
        x = self.fc(x)
        return torch.sigmoid(x)

gnn_model = SimpleGNN(input_dim=NUM_CLASSES + 1)

def process_detections(all_detections):
    if not all_detections:
        return 0.0

    nodes = []
    node_features_list = []

    for det in all_detections:
        center = np.array([(det['bbox'][0] + det['bbox'][2]) / 2, (det['bbox'][1] + det['bbox'][3]) / 2])
        nodes.append({'id': len(nodes), 'center': center, 'class': det['class'], 'conf': det['conf']})
        
        feature_vec = np.zeros(NUM_CLASSES + 1)
        class_name = det['class']
        if class_name in CLASS_TO_ID:
            feature_vec[CLASS_TO_ID[class_name]] = 1.0
        feature_vec[NUM_CLASSES] = det['conf']
        node_features_list.append(feature_vec)

    if not nodes:
        return 0.0

    node_features = torch.tensor(np.array(node_features_list), dtype=torch.float)
    
    G = nx.Graph()
    centers = np.array([node['center'] for node in nodes])
    if len(centers) > 1:
        dist_matrix = euclidean_distances(centers)
        for i in range(len(nodes)):
            for j in range(i + 1, len(nodes)):
                if dist_matrix[i, j] < DISTANCE_THRESHOLD:
                    G.add_edge(i, j)
    
    edge_index = torch.tensor(list(G.edges)).t().contiguous() if G.edges else torch.empty((2, 0), dtype=torch.long)
    graph_data = Data(x=node_features, edge_index=edge_index)
    
    with torch.no_grad():
        score = gnn_model(graph_data).item()
    return score

# --- Kafka Logic ---
consumer = KafkaConsumer(
    DETECTION_TOPIC,
    bootstrap_servers=KAFKA_BROKER,
    value_deserializer=lambda v: json.loads(v.decode('utf-8')),
    group_id='gnn-aggregator-group',
    auto_offset_reset='earliest'
)

producer = KafkaProducer(
    bootstrap_servers=KAFKA_BROKER,
    value_serializer=lambda v: json.dumps(v).encode('utf-8')
)

image_detections = {}
DETECTOR_SERVICES = {'vehicle_detector', 'water_detector', 'wildfire_detector'}

print("GNN Aggregator service is running...")
for message in consumer:
    data = message.value
    image_id = data['image_id']

    if image_id not in image_detections:
        image_detections[image_id] = {
            'detections': [],
            'services_responded': set(),
            'filename': data.get('filename') # UPDATED: Capture filename
        }

    image_detections[image_id]['detections'].extend(data['detections'])
    image_detections[image_id]['services_responded'].add(data['detector_id'])
    
    print(f"[GNN] Received detections for {image_id} from {data['detector_id']}. "
          f"Total services responded: {len(image_detections[image_id]['services_responded'])}/3")

    if image_detections[image_id]['services_responded'] == DETECTOR_SERVICES:
        print(f"[GNN] All detections received for {image_id}. Processing...")
        all_dets = image_detections[image_id]['detections']
        filename = image_detections[image_id].get('filename', 'unknown.jpg')
        
        score = process_detections(all_dets)
        
        # UPDATED: Send a richer message with all details for the dashboard
        score_message = {
            'image_id': image_id,
            'filename': filename,
            'poaching_score': score,
            'detections': all_dets
        }
        producer.send(SCORE_TOPIC, score_message)
        producer.flush()
        print(f"[GNN] Published score for {image_id}: {score:.4f}")
        del image_detections[image_id]
