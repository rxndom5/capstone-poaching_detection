# json_writer_service.py
import json
import os
from kafka import KafkaConsumer
from kafka.errors import NoBrokersAvailable
import time
import datetime

# --- Configuration ---
KAFKA_BROKER = 'localhost:9092'
SCORE_TOPIC = 'poaching_scores_topic'
DATA_FILE = 'dashboard_data.json'

def main():
    """
    Connects to Kafka, consumes messages, and writes them to a JSON file.
    """
    print("Starting JSON Writer Service...")
    while True:
        try:
            consumer = KafkaConsumer(
                SCORE_TOPIC,
                bootstrap_servers=KAFKA_BROKER,
                value_deserializer=lambda v: json.loads(v.decode('utf-8')),
                group_id='json-writer-group',
                auto_offset_reset='earliest'
            )
            print("Successfully connected to Kafka. Waiting for messages...")

            for message in consumer:
                event_data = message.value
                event_data['timestamp'] = datetime.datetime.now().isoformat()
                
                print(f"Received event for image_id: {event_data['image_id']}")
                
                # --- Read, Update, and Write Logic ---
                existing_data = []
                if os.path.exists(DATA_FILE):
                    with open(DATA_FILE, 'r') as f:
                        try:
                            existing_data = json.load(f)
                        except json.JSONDecodeError:
                            pass # File is corrupt or empty, will overwrite
                
                # Avoid adding duplicate events
                if not any(e['image_id'] == event_data['image_id'] for e in existing_data):
                    updated_data = [event_data] + existing_data
                    with open(DATA_FILE, 'w') as f:
                        json.dump(updated_data, f, indent=4)
                    print(f"Updated {DATA_FILE} with new event.")

        except NoBrokersAvailable:
            print("Connection Error: Could not find Kafka brokers. Retrying in 5 seconds...")
            time.sleep(5)
        except Exception as e:
            print(f"An unexpected error occurred: {e}. Retrying in 5 seconds...")
            time.sleep(5)

if __name__ == "__main__":
    main()
