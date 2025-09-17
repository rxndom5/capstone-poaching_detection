# result_consumer.py
import json
from kafka import KafkaConsumer

KAFKA_BROKER = 'localhost:9092'
SCORE_TOPIC = 'poaching_scores_topic'

consumer = KafkaConsumer(
    SCORE_TOPIC,
    bootstrap_servers=KAFKA_BROKER,
    value_deserializer=lambda v: json.loads(v.decode('utf-8')),
    group_id='score-display-group',
    auto_offset_reset='earliest'
)

print("Result consumer is running...")
for message in consumer:
    score_data = message.value
    print("-------------------------------------------")
    print(f"Final Score Received for Image ID: {score_data['image_id']}")
    print(f"Poaching Confidence: {score_data['poaching_score']:.4f}")
    print("-------------------------------------------")
