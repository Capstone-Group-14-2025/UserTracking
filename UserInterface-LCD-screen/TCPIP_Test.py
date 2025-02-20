import socket

# IP and port should match what DistanceAngleTracker is sending to
HOST = "0.0.0.0"  # Ensure this matches your RoboEyesApp setup
PORT = 12345            # Ensure this matches DistanceAngleTracker's LCD_pi_port

def start_tcp_test_server(host, port):
    """Starts a simple TCP server to receive and print commands from the tracker."""
    try:
        with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as server_socket:
            server_socket.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
            server_socket.bind((host, port))
            server_socket.listen(5)
            print(f"[TCP Test Server] Listening on {host}:{port}...")

            while True:
                client_conn, client_addr = server_socket.accept()
                with client_conn:
                    print(f"[TCP Test Server] Connection from {client_addr}")
                    data = client_conn.recv(1024)
                    if not data:
                        print("[TCP Test Server] Empty command received. Closing connection.")
                        continue
                    command = data.decode("utf-8").strip()
                    print(f"[TCP Test Server] Received command: '{command}'")

    except KeyboardInterrupt:
        print("\n[TCP Test Server] Shutting down...")
    except Exception as e:
        print(f"[TCP Test Server] Error: {e}")

if __name__ == "__main__":
    start_tcp_test_server(HOST, PORT)
