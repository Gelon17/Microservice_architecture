import csv
import json
import os
import threading
import pika

# Path to log file (mounted from host)
LOG_FILE = "/logs/metric_log.csv"

# In-memory buffer: { message_id: {"y_true": ..., "y_pred": ...} }
buffer = {}
buffer_lock = threading.Lock()

# Initialize CSV file with header if not exists
if not os.path.exists(LOG_FILE):
    with open(LOG_FILE, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["id", "y_true", "y_pred", "absolute_error"])
    print(f"[metric] Created {LOG_FILE} with header.")


def try_write_metric(message_id):
    """If both y_true and y_pred are available for this id, compute error and write to CSV."""
    entry = buffer.get(message_id, {})
    if entry.get("y_true") is not None and entry.get("y_pred") is not None:
        y_true = entry["y_true"]
        y_pred = entry["y_pred"]
        absolute_error = round(abs(y_true - y_pred), 4)

        with open(LOG_FILE, "a", newline="") as f:
            writer = csv.writer(f)
            writer.writerow([message_id, y_true, y_pred, absolute_error])

        print(f"[metric] id={message_id:.4f} | y_true={y_true} | y_pred={y_pred} | abs_err={absolute_error}")

        # Remove from buffer
        del buffer[message_id]


def callback_y_true(ch, method, properties, body):
    message = json.loads(body)
    message_id = message["id"]
    y_true = float(message["body"])

    with buffer_lock:
        if message_id not in buffer:
            buffer[message_id] = {"y_true": None, "y_pred": None}
        buffer[message_id]["y_true"] = y_true
        try_write_metric(message_id)

    ch.basic_ack(delivery_tag=method.delivery_tag)


def callback_y_pred(ch, method, properties, body):
    message = json.loads(body)
    message_id = message["id"]
    y_pred = float(message["body"])

    with buffer_lock:
        if message_id not in buffer:
            buffer[message_id] = {"y_true": None, "y_pred": None}
        buffer[message_id]["y_pred"] = y_pred
        try_write_metric(message_id)

    ch.basic_ack(delivery_tag=method.delivery_tag)


# Connect to RabbitMQ
connection = pika.BlockingConnection(pika.ConnectionParameters(host='rabbitmq'))
channel = connection.channel()

# Declare queues
channel.queue_declare(queue='y_true', durable=True)
channel.queue_declare(queue='y_pred', durable=True)

channel.basic_qos(prefetch_count=1)
channel.basic_consume(queue='y_true', on_message_callback=callback_y_true)
channel.basic_consume(queue='y_pred', on_message_callback=callback_y_pred)

print("[metric] Waiting for messages in queues y_true and y_pred...")
channel.start_consuming()
