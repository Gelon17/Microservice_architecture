import json
import time
import numpy as np
import pika

from datetime import datetime
from sklearn.datasets import fetch_california_housing

# Load dataset
data = fetch_california_housing()
X = data.data
y = data.target

# Connect to RabbitMQ
connection = pika.BlockingConnection(pika.ConnectionParameters(host='rabbitmq'))
channel = connection.channel()

# Declare queues
channel.queue_declare(queue='X', durable=True)
channel.queue_declare(queue='y_true', durable=True)

print("Features service started. Sending data...")

while True:
    # Pick a random row
    random_row = np.random.randint(0, len(X))

    # Create unique message ID based on timestamp
    message_id = datetime.timestamp(datetime.now())

    # Message with features (for model)
    message_X = {
        "id": message_id,
        "body": X[random_row].tolist()
    }

    # Message with true label (for metric)
    message_y_true = {
        "id": message_id,
        "body": float(y[random_row])
    }

    # Send features to queue X
    channel.basic_publish(
        exchange='',
        routing_key='X',
        body=json.dumps(message_X),
        properties=pika.BasicProperties(delivery_mode=2)  # persistent
    )

    # Send true label to queue y_true
    channel.basic_publish(
        exchange='',
        routing_key='y_true',
        body=json.dumps(message_y_true),
        properties=pika.BasicProperties(delivery_mode=2)
    )

    print(f"[features] Sent message_id={message_id:.4f}, row={random_row}, y_true={y[random_row]:.4f}")

    time.sleep(10)
