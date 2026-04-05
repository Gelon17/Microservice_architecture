import json
import numpy as np
import pika

from sklearn.datasets import fetch_california_housing
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split

# Train a model at startup
print("[model] Loading dataset and training model...")
data = fetch_california_housing()
X, y = data.data, data.target

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

model = LinearRegression()
model.fit(X_train, y_train)
print(f"[model] Model trained. Test R2={model.score(X_test, y_test):.4f}")

# Connect to RabbitMQ
connection = pika.BlockingConnection(pika.ConnectionParameters(host='rabbitmq'))
channel = connection.channel()

# Declare queues
channel.queue_declare(queue='X', durable=True)
channel.queue_declare(queue='y_pred', durable=True)


def callback(ch, method, properties, body):
    message = json.loads(body)
    message_id = message["id"]
    features = np.array(message["body"]).reshape(1, -1)

    # Make prediction
    y_pred = float(model.predict(features)[0])

    # Send prediction to y_pred queue
    message_y_pred = {
        "id": message_id,
        "body": round(y_pred, 4)
    }
    channel.basic_publish(
        exchange='',
        routing_key='y_pred',
        body=json.dumps(message_y_pred),
        properties=pika.BasicProperties(delivery_mode=2)
    )

    print(f"[model] id={message_id:.4f} -> y_pred={y_pred:.4f}")
    ch.basic_ack(delivery_tag=method.delivery_tag)


channel.basic_qos(prefetch_count=1)
channel.basic_consume(queue='X', on_message_callback=callback)

print("[model] Waiting for messages in queue X...")
channel.start_consuming()
