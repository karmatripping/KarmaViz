import numpy as np 

CHUNK = 151  
DATA_FORMAT = np.float32
CHANNELS = 1  # Mono audio for single centered waveform
RATE = 80000

WIDTH, HEIGHT = 1024, 576
CROSSFADE_SECS = 1.0

AUDIO_SETTINGS = {
    "samplerate": RATE,
    "channels": CHANNELS,
    "dtype": DATA_FORMAT,
    "blocksize": CHUNK,  # Use updated CHUNK
    "latency": "low",  # Request low latency
    "dither_off": True,  # Turn off dithering to reduce CPU load
    "exclusive": False,  # Don't use exclusive mode which can cause distortion
}
