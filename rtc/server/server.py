from time import time
from rich import print
import webrtcvad
import argparse
import asyncio
import json
import logging
import os
import ssl
import sys
import uuid

import cv2
from aiohttp import web
from av.frame import Frame
from aiortc import MediaStreamTrack, RTCPeerConnection, RTCSessionDescription, AudioStreamTrack
from aiortc.contrib.media import MediaBlackhole, MediaPlayer, MediaRecorder, MediaRelay
from av import VideoFrame
import numpy as np
from faster_whisper import WhisperModel

import soundfile as sf

model_size = "distil-small.en"

# Run on GPU with FP16
device = 'cpu'
compute_type = 'int8' if device == 'cpu' else 'float16'

model = WhisperModel(model_size,
                     device=device,
                     compute_type=compute_type,
                     )

ROOT = os.path.dirname(__file__)

logger = logging.getLogger("pc")
pcs = set()
relay = MediaRelay()


def update_terminal(output):
    sys.stdout.write('\r' + output)
    sys.stdout.flush()


class VADAudioTrack(AudioStreamTrack):
    """
    An audio stream track that processes frames with VAD.
    """
    kind = 'audio'
    def __init__(self, source_track, queue):
        super().__init__()  # Initialize the base AudioStreamTrack
        self.source_track = source_track
        self.vad = webrtcvad.Vad()
        self.vad.set_mode(3)  # Mode can be 0 (normal), 1, or 2 (most aggressive)

        self.queue = queue

        #debugstuff
        self.framertons = []
        self.framecount = 0

    async def recv(self):
        _frame = await self.source_track.recv()
        frame_array = _frame.to_ndarray()[0]
        
        print(len( _frame.to_ndarray() ))
        self.framertons.append(frame_array)
        self.framecount += 1
        print(self.framecount)
        if not self.framecount < 100:
            print("writing raw sample")
            result = np.concatenate(self.framertons, axis=0)
            sf.write('recordings/raw_{}.wav'.format(time()), result, 48000)
            self.framertons = []
            self.framecount = 0
        
        # Assuming frame_array is already int16, check this assumption
        if frame_array.dtype != np.int16:
            raise ValueError("Frame array is not of type int16")

        # Reshape and slice the array to ensure correct frame sizes (480 samples per 10ms)
        frame_length = 480
        total_samples = frame_array.shape[0]
        frames = [frame_array[i:i + frame_length] for i in range(0, total_samples, frame_length)]
        
        start = time()

        for frame in frames:

            # Convert numpy array to bytes, ensure it is int16
            byte_frame = frame.astype(np.int16).tobytes()

            # VAD processing
            is_speech = self.vad.is_speech(byte_frame, 48000)  # Using 48000Hz as sample rate
            if is_speech:
                await self.queue.put(frame)


        return _frame

async def index(request):
    content = open(os.path.join(ROOT, "index.html"), "r").read()
    return web.Response(content_type="text/html", text=content)


async def javascript(request):
    content = open(os.path.join(ROOT, "client.js"), "r").read()
    return web.Response(content_type="application/javascript", text=content)


async def offer(request):
    params = await request.json()
    offer = RTCSessionDescription(sdp=params["sdp"], type=params["type"])

    pc = RTCPeerConnection()
    pc_id = "PeerConnection(%s)" % uuid.uuid4()
    pcs.add(pc)

    def log_info(msg, *args):
        logger.info(pc_id + " " + msg, *args)

    log_info("Created for %s", request.remote)

    # prepare local media
    player = MediaPlayer(os.path.join(ROOT, "demo_instruct.wav"))
    if args.record_to:
        recorder = MediaRecorder(args.record_to)
    else:
        recorder = MediaBlackhole()

    @pc.on("datachannel")
    def on_datachannel(channel):
        @channel.on("message")
        def on_message(message):
            if isinstance(message, str) and message.startswith("ping"):
                channel.send("pong" + message[4:])

    @pc.on("connectionstatechange")
    async def on_connectionstatechange():
        log_info("Connection state is %s", pc.connectionState)
        if pc.connectionState == "failed":
            await pc.close()
            pcs.discard(pc)

    async def generate_audio_segments(queue):
        frames = []
        while True:
            try:
                frame = await asyncio.wait_for(queue.get(), timeout=5)
                # Process the frame, e.g., send it to a transcription service
                frames.append(frame)
                queue.task_done()
            except asyncio.TimeoutError:
                if frames:
                    result = np.concatenate(frames, axis=0)
                    print('recording')

                    sf.write('recordings/{}.wav'.format(time()), result, 48000)

                    yield result
                    frames = []

    @pc.on("track")
    def on_track(track):
        log_info("Track %s received", track.kind)

        if track.kind == "audio":
            # pc.addTrack(player.audio)
            queue = asyncio.Queue()
            pc.addTrack(player.audio)
            recorder.addTrack(VADAudioTrack(track,
                                            queue,
                                            ))
            # Start the consumer task
            audio_segments = generate_audio_segments(queue)
            
            async def transcribe_segments(segments):
                async for segment in segments:
                    print(len( segment ))
                    # start = time()
                    # for x in model.transcribe(segment)[0]:
                    #     print(x)
                    # print(time() - start)
            
            processor_task = asyncio.create_task(transcribe_segments(audio_segments))

            @track.on("ended")
            async def on_ended():
                log_info("Track %s ended", track.kind)
                await recorder.stop()
                processor_task.cancel()  # Optionally cancel the consumer task when the track ends


        @track.on("ended")
        async def on_ended():
            log_info("Track %s ended", track.kind)
            await recorder.stop()

    # handle offer
    await pc.setRemoteDescription(offer)
    await recorder.start()

    # send answer
    answer = await pc.createAnswer()
    await pc.setLocalDescription(answer)

    return web.Response(
        content_type="application/json",
        text=json.dumps(
            {"sdp": pc.localDescription.sdp, "type": pc.localDescription.type}
        ),
    )


async def on_shutdown(app):
    # close peer connections
    coros = [pc.close() for pc in pcs]
    await asyncio.gather(*coros)
    pcs.clear()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="WebRTC audio / video / data-channels demo"
    )
    parser.add_argument("--cert-file", help="SSL certificate file (for HTTPS)")
    parser.add_argument("--key-file", help="SSL key file (for HTTPS)")
    parser.add_argument(
        "--host", default="0.0.0.0", help="Host for HTTP server (default: 0.0.0.0)"
    )
    parser.add_argument(
        "--port", type=int, default=8080, help="Port for HTTP server (default: 8080)"
    )
    parser.add_argument("--record-to", help="Write received media to a file.")
    parser.add_argument("--verbose", "-v", action="count")
    args = parser.parse_args()

    if args.verbose:
        logging.basicConfig(level=logging.DEBUG)
    else:
        logging.basicConfig(level=logging.INFO)

    if args.cert_file:
        ssl_context = ssl.SSLContext()
        ssl_context.load_cert_chain(args.cert_file, args.key_file)
    else:
        ssl_context = None

    app = web.Application()
    app.on_shutdown.append(on_shutdown)
    app.router.add_get("/", index)
    app.router.add_get("/client.js", javascript)
    app.router.add_post("/offer", offer)
    web.run_app(
        app, access_log=None, host=args.host, port=args.port, ssl_context=ssl_context
    )
