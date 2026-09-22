import socket
import requests

# DENOMINATION RECEIVER & WEB UI BRIDGE
HOST = "0.0.0.0"
PORT = 5001  # Ethernet TCP Port
API_URL = "http://localhost:5000/api/predict"

print("=" * 50)
print("DENOMINATION RECEIVER & VIBRACOIN WEB BRIDGE")
print("=" * 50)
print(f"Listening on port {PORT}...")
print()

server = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
server.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
server.bind((HOST, PORT))
server.listen(5)

print("SERVER READY")
print("Waiting for Ethernet connections...")
print()

while True:
    connection, address = server.accept()

    print("=" * 50)
    print("CONNECTED")
    print(f"Sender: {address}")
    print("=" * 50)

    with connection:
        while True:
            data = connection.recv(1024)

            if not data:
                print("Sender closed connection.")
                print()
                break

            # DEBUG: show exactly what arrived
            print(f"RAW DATA RECEIVED: {repr(data)}")

            try:
                prediction = data.decode("utf-8").strip()
                print(f"PREDICTION: {prediction}")

                if prediction:
                    # Forward prediction to Node.js Backend API
                    res = requests.post(API_URL, json={
                        "coin": prediction,
                        "source": f"Teammate ML (TCP Ethernet)"
                    })
                    if res.status_code == 200:
                        print(" -> Successfully forwarded to Web UI & Malayalam AI Generator!")
                    else:
                        print(f" -> Backend API Error: {res.status_code}")

            except Exception as e:
                print(f"Decode error: {e}")

            print()
