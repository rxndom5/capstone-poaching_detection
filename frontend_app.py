# frontend_app.py
import streamlit as st
from kafka import KafkaProducer, KafkaConsumer
import json
import uuid
import base64
import time
from PIL import Image
import io

# --- Kafka Configuration ---
KAFKA_BROKER = 'localhost:9092'
IMAGE_TOPIC = 'image_ingestion_topic'
SCORE_TOPIC = 'poaching_scores_topic'

# --- Streamlit Page Configuration ---
st.set_page_config(page_title="Poaching Detection System", layout="centered")
st.title("Poaching Detection in the Amazon Rainforest 🌳")

# Initialize Kafka Producer
producer = KafkaProducer(
    bootstrap_servers=KAFKA_BROKER,
    value_serializer=lambda v: json.dumps(v).encode('utf-8')
)

# Initialize Session State
if 'image_id' not in st.session_state:
    st.session_state.image_id = None
if 'final_score' not in st.session_state:
    st.session_state.final_score = None
if 'image_uploaded' not in st.session_state:
    st.session_state.image_uploaded = None

# --- UI Components ---
uploaded_file = st.file_uploader("Choose an image to analyze...", type=["jpg", "jpeg", "png"])

if uploaded_file is not None:
    # Display the uploaded image
    st.image(uploaded_file, caption='Uploaded Image', use_column_width=True)
    st.session_state.image_uploaded = uploaded_file.getvalue()

if st.button('Detect Poaching Activity', disabled=(st.session_state.image_uploaded is None)):
    # Reset previous results
    st.session_state.final_score = None
    
    # 1. Produce the image to Kafka
    image_id = str(uuid.uuid4())
    st.session_state.image_id = image_id
    
    image_data = base64.b64encode(st.session_state.image_uploaded).decode('utf-8')
    
    message = {
        'image_id': image_id,
        'image_data': image_data,
        'filename': uploaded_file.name
    }
    
    producer.send(IMAGE_TOPIC, message)
    producer.flush()
    st.info(f"Image sent for processing with ID: {image_id}")

    # 2. Consume the result from Kafka
    with st.spinner('Backend services are analyzing the image... Please wait.'):
        consumer = KafkaConsumer(
            SCORE_TOPIC,
            bootstrap_servers=KAFKA_BROKER,
            value_deserializer=lambda v: json.loads(v.decode('utf-8')),
            group_id=f'streamlit-consumer-{image_id}', # Unique group_id to get all messages
            auto_offset_reset='latest' 
        )
        
        start_time = time.time()
        while time.time() - start_time < 60: # 60-second timeout
            # Poll for messages without blocking forever
            messages = consumer.poll(timeout_ms=1000)
            if not messages:
                continue

            for tp, records in messages.items():
                for record in records:
                    score_data = record.value
                    if score_data.get('image_id') == st.session_state.image_id:
                        st.session_state.final_score = score_data.get('poaching_score')
                        break # Found our result
            if st.session_state.final_score is not None:
                break
        
        consumer.close()


# Display the final score once it's available
if st.session_state.final_score is not None:
    score = st.session_state.final_score
    st.success("Analysis Complete!")
    
    # Determine color based on score
    if score < 0.5:
        color = "GREEN"
    elif score < 0.75:
        color = "ORANGE"
    else:
        color = "RED"
        
    st.metric(
        label="Poaching Confidence Score",
        value=f"{score:.2%}",
        delta=f"Risk Level: {'Low' if color == 'GREEN' else 'Moderate' if color == 'ORANGE' else 'High'}",
        delta_color="off" # Using custom styling for the label text instead
    )
    
    # Custom styling with markdown for color
    st.markdown(f"<h3 style='color:{color};'>{score:.2%}</h3>", unsafe_allow_html=True)
    
elif st.session_state.image_id and st.session_state.final_score is None:
    st.error("Processing timed out. The backend services might be slow or not running.")
