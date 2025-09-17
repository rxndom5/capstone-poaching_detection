# image_producer.py
import os
import time
import json
import base64
import uuid
from kafka import KafkaProducer

# --- Kafka Configuration ---
KAFKA_BROKER = 'localhost:9092'
IMAGE_TOPIC = 'image_ingestion_topic'
IMAGE_DIR = 'test_images'

producer = KafkaProducer(
    bootstrap_servers=KAFKA_BROKER,
    value_serializer=lambda v: json.dumps(v).encode('utf-8')
)

def send_image(image_path):
    try:
        with open(image_path, "rb") as image_file:
            image_data = base64.b64encode(image_file.read()).decode('utf-8')
        
        image_id = str(uuid.uuid4())
        message = {
            'image_id': image_id,
            'image_data': image_data,
            'filename': os.path.basename(image_path)
        }
        
        print(f"Sending image {message['filename']} with ID: {image_id}")
        producer.send(IMAGE_TOPIC, message)
        producer.flush()

    except Exception as e:
        print(f"Error sending image {image_path}: {e}")

if __name__ == "__main__":
    print("Starting image producer...")
    processed_files = set()
    while True:
        for filename in os.listdir(IMAGE_DIR):
            if filename.endswith(('.jpg', '.jpeg', '.png')) and filename not in processed_files:
                image_path = os.path.join(IMAGE_DIR, filename)
                send_image(image_path)
                processed_files.add(filename)
        
        print("Waiting for new images...")
        time.sleep(10)
