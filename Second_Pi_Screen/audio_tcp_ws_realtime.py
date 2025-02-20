"""
file:           audio_tcp_ws_realtime.py
version:        v.1

Description:    
    - Creates a TCP server to receive 'alerts' from another component (e.g., obstacle detection).
    - Connects to OpenAI’s Realtime API via WebSocket to send/receive audio chunks and text.
    - Handles live microphone input using PyAudio, and plays AI-generated audio to the speaker using OpenAI Realtime API.
    - Interrupts ongoing speech on receiving certain alerts, and speaks out custom or predefined messages.
    - NOTE: Works with python 3.12.8, later versions I tested had issues with pyaudio.
"""

import asyncio
import websockets
import json
import pyaudio
import base64
import socket
import sys
import time
from datetime import datetime


# ------------------------------------------------------------------------------
# CONFIG & CONSTANTS
# ------------------------------------------------------------------------------
API_URL = "wss://api.openai.com/v1/realtime?model=gpt-4o-realtime-preview-2024-12-17"  # Example endpoint, update as needed
API_KEY = ""

FORMAT = pyaudio.paInt16
CHANNELS = 1
SAMPLE_RATE = 24000
CHUNK_SIZE = 512

HOST = "0.0.0.0"
PORT = 5555

ALERT_MESSAGES = {
    "start": "We’re now getting started. Please stand in front of me at the distance you want.",
    "calibration complete": "Calibration done. I’m now following you.",
    "obstacle detected": "Obstacle detected, stopping now.",
    "user lost": "User lost, please come back.",
    "stop": "Stopping now, bye."
}


# ------------------------------------------------------------------------------
# TCP SERVER CLASSES & FUNCTIONS
# ------------------------------------------------------------------------------
async def handle_client(reader: asyncio.StreamReader, writer: asyncio.StreamWriter, alerts_queue: asyncio.Queue):
    """
    Handle communication with a single TCP client. 
    
    Continuously reads data (up to 512 bytes) from the client, decodes it into 
    a UTF-8 string, and if the message is non-empty, places it into the alerts_queue.

    :param reader:        StreamReader for reading from the client socket.
    :param writer:        StreamWriter for writing to the client socket.
    :param alerts_queue:  An asyncio.Queue shared with the main loop for passing messages.
    """
    try:
        while True:
            data = await reader.read(512)
            if not data:
                # The client has disconnected or no more data is available
                break

            msg = data.decode("utf-8", errors="ignore").strip().lower()
            if msg:
                await alerts_queue.put(msg)
    except (asyncio.CancelledError, ConnectionResetError):
        # Normal exceptions during shutdown or dropped connections
        pass
    finally:
        writer.close()
        await writer.wait_closed()


async def start_tcp_server(alerts_queue: asyncio.Queue):
    """
    Create and start a TCP server that listens for incoming connections on (HOST, PORT).
    Each new client connection is handled by 'handle_client' in a separate coroutine.

    :param alerts_queue:  The shared asyncio.Queue for pushing alerts to the main loop.
    """
    server = await asyncio.start_server(
        lambda r, w: handle_client(r, w, alerts_queue),
        HOST,
        PORT
    )
    print(f"[AlertServer] Listening on {HOST}:{PORT}")

    async with server:
        await server.serve_forever()


# ------------------------------------------------------------------------------
# SPEECH CONTROL & HELPER FUNCTIONS
# ------------------------------------------------------------------------------
async def cancel_and_drain(ws: websockets.WebSocketClientProtocol):
    """
    Sends a 'response.cancel' event to the Realtime API, which halts any in-progress 
    assistant speech. Then, it quickly drains queued partial events so that old content 
    doesn't continue to stream out.

    :param ws:  An active WebSocket client connection to the OpenAI Realtime API.
    """
    cancel_event = {"type": "response.cancel"}
    await ws.send(json.dumps(cancel_event))

    # Attempt to read/clear leftover events for ~200ms
    while True:
        try:
            message = await asyncio.wait_for(ws.recv(), timeout=0.02)
            if message:
                # We discard these partial events; they are no longer relevant
                pass
        except asyncio.TimeoutError:
            # No more leftover messages arrived in time
            break


async def interrupt_speech(ws: websockets.WebSocketClientProtocol, text: str):
    """
    Sends a 'response.create' event to the Realtime API, instructing the assistant 
    to speak (and optionally display) the specified text. By design, this interrupts 
    any prior speech if 'cancel_and_drain' was already called.

    :param ws:    An active WebSocket connection to the OpenAI Realtime API.
    :param text:  The text to be spoken by the assistant.
    """
    event = {
        "type": "response.create",
        "response": {
            "input": [],
            "instructions": f"Say exactly the following: {text}"
        }
    }
    await ws.send(json.dumps(event))


# ------------------------------------------------------------------------------
# MAIN ASYNC TASK & APPLICATION ENTRY
# ------------------------------------------------------------------------------
async def main():
    """
    Main asynchronous entry point.

    Orchestrates:
      1) Starting a TCP server to receive alerts in an asyncio.Queue.
      2) Setting up PyAudio streams for input (microphone) and output (speaker).
      3) Connecting to OpenAI Realtime via WebSocket.
      4) Continuously sending microphone audio and receiving partial AI responses (both text and audio).
      5) Handling 'alerts' which can interrupt AI speech with a different response.
    """
    # 1) Create a queue for alerts and start the TCP server
    alerts_queue = asyncio.Queue()
    tcp_server_task = asyncio.create_task(start_tcp_server(alerts_queue))

    # 2) Prepare PyAudio input and output streams
    import pyaudio
    audio_interface = pyaudio.PyAudio()

    mic_stream = audio_interface.open(
        format=FORMAT,
        channels=CHANNELS,
        rate=SAMPLE_RATE,
        input=True,
        frames_per_buffer=CHUNK_SIZE
    )
    speaker_stream = audio_interface.open(
        format=FORMAT,
        channels=CHANNELS,
        rate=SAMPLE_RATE,
        output=True,
        frames_per_buffer=CHUNK_SIZE
    )
    print("[Audio] Microphone and speaker streams opened.\n")

    # 3) Connect to the OpenAI Realtime API
    headers = {
        "Authorization": f"Bearer {API_KEY}",
        "OpenAI-Beta": "realtime=v1"
    }

    async with websockets.connect(API_URL, extra_headers=headers) as ws:
        print("[OpenAI] Connected to Realtime API.\n")

        # 3a) Configure the session with desired parameters (model, voice, instructions, etc.)
        session_update = {
            "type": "session.update",
            "session": {
                "model": "gpt-4o-realtime-preview-2024-12-17",
                "instructions": (
                    "You are a robot that helps the user carry their items. "
                    "You are polite, helpful, and friendly, but avoid stating explicitly "
                    "'I am your companion.' The user presses 'start' to begin. "
                    "Then you ask for their name and instruct them to calibrate. "
                    "Avoid repeating certain prompts like 'What can I do?' over and over. "
                    "Keep a casual tone and moderate enthusiasm. "
                    "Your main role is to follow the user and engage in small talk if desired. "
                    "Keep responses concise and avoid rambling."
                ),
                "modalities": ["audio", "text"],
                "voice": "sage",
                "turn_detection": {
                    "type": "server_vad",
                },
                "input_audio_transcription": {"model": "whisper-1"}
            }
        }
        await ws.send(json.dumps(session_update))

        # 3b) Speak an initial greeting
        await interrupt_speech(ws, "Hello, I am Jaguar! I'm here to help you. What's your name?")
        print("[Realtime] Session configured. Entering real-time loop. Press Ctrl+C to exit.\n")

        try:
            while True:
                # ------------------------------------------------------------------
                # (A) Check if any alerts arrived
                # ------------------------------------------------------------------
                if not alerts_queue.empty():
                    alert_msg = await alerts_queue.get()
                    print(f"[ALERT] Received: {alert_msg}")

                    # Cancel ongoing speech and drain pending audio
                    await cancel_and_drain(ws)

                    # If the alert is recognized in our predefined messages:
                    if alert_msg in ALERT_MESSAGES:
                        await interrupt_speech(ws, ALERT_MESSAGES[alert_msg])
                        
                        # If it's the 'stop' alert, break out of the loop
                        if alert_msg == "stop":
                            print("[ALERT] 'stop' alert detected. Exiting.")
                            break
                    else:
                        # Handle custom or unrecognized alerts
                        if alert_msg.startswith("obstacle detected"):
                            await interrupt_speech(ws, ALERT_MESSAGES["obstacle detected"])
                        elif alert_msg.startswith("user lost"):
                            await interrupt_speech(ws, ALERT_MESSAGES["user lost"])
                        else:
                            print(f"[Unknown Alert] {alert_msg}")

                # ------------------------------------------------------------------
                # (B) Capture audio from microphone
                # ------------------------------------------------------------------
                mic_data = mic_stream.read(CHUNK_SIZE, exception_on_overflow=False)
                audio_b64 = base64.b64encode(mic_data).decode("utf-8")

                # Send chunk to the Realtime API
                append_event = {
                    "type": "input_audio_buffer.append",
                    "audio": audio_b64
                }
                await ws.send(json.dumps(append_event))

                # ------------------------------------------------------------------
                # (C) Receive partial messages from the AI
                # ------------------------------------------------------------------
                try:
                    message = await asyncio.wait_for(ws.recv(), timeout=0.008)
                except asyncio.TimeoutError:
                    message = None  # No immediate data from the server

                if message:
                    data = json.loads(message)
                    event_type = data.get("type")

                    if event_type == "response.audio.delta":
                        # AI audio chunk
                        chunk_b64 = data.get("delta", "")
                        if chunk_b64:
                            decoded = base64.b64decode(chunk_b64)
                            speaker_stream.write(decoded)

                    elif event_type == "response.text.delta":
                        # Partial text
                        print(data.get("delta", ""), end="", flush=True)

                    elif event_type == "response.text.done":
                        # Completed text chunk
                        print("\n[Text response completed]")

                # ------------------------------------------------------------------
                # (D) Pause slightly to reduce CPU usage
                # ------------------------------------------------------------------
                await asyncio.sleep(0.0001)

        except KeyboardInterrupt:
            print("\n[KeyboardInterrupt] Exiting main loop...")

        # ----------------------------------------------------------------------
        # Clean up PyAudio streams
        # ----------------------------------------------------------------------
        mic_stream.stop_stream()
        mic_stream.close()
        speaker_stream.stop_stream()
        speaker_stream.close()
        audio_interface.terminate()
        print("[Audio] Streams closed.")

    # --------------------------------------------------------------------------
    # If we reached here, either due to 'stop' alert or outside reason
    # --------------------------------------------------------------------------
    tcp_server_task.cancel()
    print("[Main] Application shutdown complete.")


# ------------------------------------------------------------------------------
# SCRIPT ENTRY POINT
# ------------------------------------------------------------------------------
if __name__ == "__main__":
    """
    Launch the main asyncio run. 
    We capture a KeyboardInterrupt to allow graceful shutdown.
    """
    try:
        asyncio.run(main())
    except KeyboardInterrupt:
        sys.exit(0)
