# wildfire_detector_service.py
import json
import base64
from kafka import KafkaConsumer, KafkaProducer
from ultralytics import YOLO
from PIL import Image
import io

# --- Kafka Configuration ---
KAFKA_BROKER = 'localhost:9092'
IMAGE_TOPIC = 'image_ingestion_topic'
DETECTION_TOPIC = 'detection_results_topic'
SERVICE_ID = 'wildfire_detector'

# --- Model Loading ---
MODEL_PATH = 'models/wildfire.pt'
model = YOLO(MODEL_PATH)

# --- Kafka Consumers and Producers ---
consumer = KafkaConsumer(
    IMAGE_TOPIC,
    bootstrap_servers=KAFKA_BROKER,
    value_deserializer=lambda v: json.loads(v.decode('utf-8')),
    group_id='wildfire-detector-group' # Unique group ID
)

producer = KafkaProducer(
    bootstrap_servers=KAFKA_BROKER,
    value_serializer=lambda v: json.dumps(v).encode('utf-8')
)

print("Wildfire detector service is running...")
for message in consumer:
    image_info = message.value
    image_id = image_info['image_id']
    image_data = base64.b64decode(image_info['image_data'])
    
    print(f"[Wildfire] Processing image ID: {image_id}")
    
    image = Image.open(io.BytesIO(image_data))
    results = model(image)
    
    detections = []
    for result in results:
        for box in result.boxes:
            cls_name = model.names[int(box.cls[0])]
            conf = float(box.conf[0])
            xyxy = box.xyxy[0].cpu().numpy().tolist()
            detections.append({'class': cls_name, 'conf': conf, 'bbox': xyxy})

    # Send results
    result_message = {
        'image_id': image_id,
        'filename': image_info['filename'], # UPDATED: Pass filename
        'detector_id': SERVICE_ID,
        'detections': detections
    }
    producer.send(DETECTION_TOPIC, result_message)
    producer.flush()
    print(f"[Wildfire] Sent {len(detections)} detections for image ID: {image_id}")
