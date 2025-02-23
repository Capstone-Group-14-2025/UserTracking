"""
file:           v2_AI_Conversation_TCPIP.py
version:        v.2

Description:
Description:    
    - Creates a TCP server to receive 'alerts' from another component (e.g., obstacle detection).
    - Connects to OpenAI’s Realtime API via WebSocket to send/receive audio chunks and text.
    - Handles live microphone input using PyAudio, and plays AI-generated audio to the speaker using OpenAI Realtime API.
    - Interrupts ongoing speech on receiving certain alerts, and speaks out custom or predefined messages.
    - NOTE: Works with python 3.12.8, later versions I tested had issues with pyaudio.
    - NOTE: worked with v.11 of pyaudio, later versions had issues with the code.    - Maintains two states: IDLE and RUNNING.
    NEW: - "start session": begin streaming with OpenAI (mic & speaker).
         - "end session": cleanly stop streaming and return to IDLE (no server shutdown).
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
API_URL = "wss://api.openai.com/v1/realtime?model=gpt-4o-realtime-preview-2024-12-17"  # Example endpoint
API_KEY = "" 

FORMAT = pyaudio.paInt16
CHANNELS = 1
SAMPLE_RATE = 24000
CHUNK_SIZE = 512

HOST = "0.0.0.0"
PORT = 54321

ALERT_MESSAGES = {
    "start session": "Hi I'm Jaguar. I'm here to help you carry your heavy items. We’re now getting started. Please stand in front of me at the distance you want and press calibrate when you are ready, and please let me know what your name is!.",
    "calibration complete": "Calibration done. I’m now following you.",
    "obstacle detected": "Obstacle detected, stopping now.",
    "user lost": "User lost, I'm stopping till you come back in my field of view.",
    "stop": "Stopping now.",
    "end session":"Ending session, see you later!",
    "tracking again": "I’m tracking you again."
}


# ------------------------------------------------------------------------------
# TCP SERVER
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
            leftover = await asyncio.wait_for(ws.recv(), timeout=0.02)
            if leftover:
                pass  # discard
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
            "instructions": f"Say the following in a way that flows with the conversation, you can say sorry to interrupt or blend it in: {text}"
        }
    }
    await ws.send(json.dumps(event))


# ------------------------------------------------------------------------------
# RUNNING STATE: OpenAI Streaming
# ------------------------------------------------------------------------------
async def run_openai_streaming(alerts_queue: asyncio.Queue):
    """
    Opens mic & speaker, connects to OpenAI Realtime,
    streams audio until 'end session' arrives, then closes resources 
    and returns to the caller (IDLE).
    """
    # 1) Setup PyAudio
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
    print("[Audio] Mic & speaker opened for conversation.\n")

    # 2) Connect to OpenAI Realtime
    headers = {
        "Authorization": f"Bearer {API_KEY}",
        "OpenAI-Beta": "realtime=v1"
    }
    async with websockets.connect(API_URL, extra_headers=headers) as ws:
        print("[OpenAI] Connected to Realtime API.\n")

        # Configure session
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
                "turn_detection": {"type": "server_vad"},
                "input_audio_transcription": {"model": "whisper-1"}
            }
        }
        await ws.send(json.dumps(session_update))

        print("[Realtime] Session configured. Entering streaming loop.\n")

        try:
            while True:
                # --------------------------------------------------------------
                # A) Check new alerts in RUNNING state
                # --------------------------------------------------------------
                if not alerts_queue.empty():
                    alert_msg = await alerts_queue.get()
                    print(f"[RUNNING ALERT] {alert_msg}")

                    # Cancel ongoing speech
                    await cancel_and_drain(ws)

                    # If the alert is recognized in our predefined messages:
                    if alert_msg in ALERT_MESSAGES:
                        await interrupt_speech(ws, ALERT_MESSAGES[alert_msg])

                        # 'end session' => break streaming, go back to IDLE
                        if alert_msg == "end session":
                            print("[Alert] 'end session' triggered: stopping streaming.")
                            break
                    else:
                        # Partial handling
                        if alert_msg.startswith("obstacle detected"):
                            await interrupt_speech(ws, ALERT_MESSAGES["obstacle detected"])
                        elif alert_msg.startswith("user lost"):
                            await interrupt_speech(ws, ALERT_MESSAGES["user lost"])
                        else:
                            print(f"[Unknown Alert] {alert_msg}")

                # --------------------------------------------------------------
                # B) Read mic data -> Realtime
                # --------------------------------------------------------------
                mic_data = mic_stream.read(CHUNK_SIZE, exception_on_overflow=False)
                audio_b64 = base64.b64encode(mic_data).decode("utf-8")
                append_event = {"type": "input_audio_buffer.append", "audio": audio_b64}
                await ws.send(json.dumps(append_event))

                # --------------------------------------------------------------
                # C) Receive partial AI audio
                # --------------------------------------------------------------
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

        except asyncio.CancelledError:
            print("[run_openai_streaming] CancelledError => stopping streaming.")
        except KeyboardInterrupt:
            print("[KeyboardInterrupt] inside streaming => stopping.")
        finally:
            # Cleanup resources
            mic_stream.stop_stream()
            mic_stream.close()
            speaker_stream.stop_stream()
            audio_interface.terminate()
            print("[Audio] Streams closed in run_openai_streaming().")

    print("[run_openai_streaming] Done, returning to IDLE.")


# ------------------------------------------------------------------------------
# MAIN (IDLE)
# ------------------------------------------------------------------------------
async def main():
    """
    Main loop remains IDLE, waiting for "start session" or "end session".
    - "start session": calls run_openai_streaming(), returns on "end session".
    - "end session": can also be triggered while IDLE, but that just logs a message 
      or you can choose to do something else. (Currently we just show message.)
    """
    alerts_queue = asyncio.Queue()
    tcp_server_task = asyncio.create_task(start_tcp_server(alerts_queue))
    print("[Main] TCP server started. IDLE. Send 'start session' or 'end session'.\n")

    try:
        while True:
            cmd = await alerts_queue.get()
            cmd = cmd.strip().lower()
            print(f"[IDLE] Received command: {cmd}")

            if cmd == "start session":
                print("[Main] Starting run_openai_streaming() ...")
                await run_openai_streaming(alerts_queue)
                print("[Main] Returned from streaming. Back to IDLE.")

            elif cmd == "end session":
                print("[Main] 'end session' in IDLE => ignoring.")

            else:
                # Possibly handle other alerts in IDLE (calibration, user lost, etc.)
                if cmd in ALERT_MESSAGES:
                    print(f"[Main] IDLE ignoring or storing '{cmd}' => No active streaming.")
                else:
                    print(f"[Main] IDLE unknown command '{cmd}', ignoring.")

    except KeyboardInterrupt:
        print("\n[KeyboardInterrupt] in main => shutting down.")
    finally:
        print("[Main] Cancelling TCP server task and exiting.")
        tcp_server_task.cancel()
        print("[Main] Clean exit.")


# ------------------------------------------------------------------------------
# SCRIPT ENTRY
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
