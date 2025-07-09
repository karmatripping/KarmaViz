from queue import Empty, Full, Queue
from threading import Event, Lock, Thread
from time import sleep, time
import numpy as np


from dataclasses import dataclass

from config.constants import CHANNELS, CHUNK, DATA_FORMAT, RATE

from sounddevice import CallbackFlags, InputStream, PortAudioError
from modules.benchmark import benchmark
from modules.logging_config import get_logger

import numpy as np

# Get logger for this module
logger = get_logger('audio_handler')


def list_audio_devices():
    """List available audio input devices"""
    try:
        import sounddevice as sd
        devices = sd.query_devices()
        input_devices = []
        for i, device in enumerate(devices):
            if device['max_input_channels'] > 0:
                input_devices.append({
                    'id': i,
                    'name': device['name'],
                    'channels': device['max_input_channels'],
                    'sample_rate': device['default_samplerate']
                })
        return input_devices
    except Exception as e:
        logger.error(f"Error listing audio devices: {e}")
        return []


def list_audio_devices():
    """List available audio input devices"""
    try:
        import sounddevice as sd
        devices = sd.query_devices()
        input_devices = []
        for i, device in enumerate(devices):
            if device['max_input_channels'] > 0:
                input_devices.append({
                    'id': i,
                    'name': device['name'],
                    'channels': device['max_input_channels'],
                    'sample_rate': device['default_samplerate']
                })
        return input_devices
    except Exception as e:
        logger.error(f"Error listing audio devices: {e}")
        return []


def get_default_audio_device():
    """Get the default audio input device"""
    try:
        import sounddevice as sd
        return sd.default.device[0]
    except Exception as e:
        logger.error(f"Error getting default audio device: {e}")
        return None


@dataclass
class AudioData:
    """Data class for thread communication of processed audio data"""

    raw_data: np.ndarray
    fft_data: np.ndarray
    amplitude: float
    beat_detected: bool
    frequency_balance: float
    energy: float
    warmth: float


class DummyStream:
    ''' Create dummy audio stream for fallback '''
    def __init__(self, chunk_size=CHUNK):
        # Pre-allocate buffer to avoid garbage collection during audio processing
        self.chunk_size = chunk_size
        self.dummy_data = np.zeros((chunk_size,), dtype=DATA_FORMAT)
        # Add callback flags to match the expected return signature
        self.callback_flags = CallbackFlags()
        self.active = True  # Pretend to be active

    def read(self, frames):
        # Ignore the frames parameter and return the pre-allocated buffer
        # This ensures consistent behavior with the real stream
        return self.dummy_data, self.callback_flags

    def start(self):
        pass

    def stop(self):
        pass

    def close(self):
        pass


class AudioProcessor:
    """Handles audio processing in a separate thread"""

    def __init__(
        self,
        chunk_size=CHUNK,
        sample_rate=RATE,
        channels=CHANNELS,
        data_format=DATA_FORMAT,
        device=None,
    ):
        self.chunk_size = chunk_size
        self.sample_rate = sample_rate
        self.channels = channels
        self.data_format = data_format
        self.device = device

        # Thread control
        self.running = False
        self.thread = None
        self.stop_event = Event()

        # Queues for thread communication
        self.audio_queue = Queue(
            maxsize=10
        )  # Limit queue size to prevent memory growth

        # Initialize audio processing state with safe defaults
        self.amplitude = 0.0
        self.energy_buffer = 0.0
        self.threshold_buffer = 0.1
        self.last_beat_time = time()
        self.beat_detected = False
        self.min_beat_interval = 0.1
        self.last_manual_beat_time = time()
        self.excess_energy = 0.0
        self.rotation_direction = 0
        self.frequency_balance = 0.0
        self.beat_sensitivity = 1.0

        # Thread-safe locks
        self.state_lock = Lock()
        self.stream_lock = Lock()

        # Pre-allocate numpy arrays for audio processing
        self.audio_buffer = np.zeros(chunk_size, dtype=data_format)
        self.fft_buffer = np.zeros(chunk_size // 2 + 1, dtype=data_format)
        self.window = np.hanning(chunk_size)

        # Create audio stream
        self.stream = None
        try:
            self.stream = self._create_audio_stream()
        except Exception as e:
            logger.error(f"Error creating initial audio stream: {e}")
            self.stream = DummyStream(self.chunk_size)

    def _create_audio_stream(self):
        """Create the audio input stream with error handling"""
        try:
            # Use only supported parameters for InputStream
            stream = InputStream(
                samplerate=self.sample_rate,
                channels=self.channels,
                dtype=self.data_format,
                blocksize=self.chunk_size,
                device=self.device,
                latency="low",
                dither_off=True,  # Turn off dithering to reduce CPU load
            )
            return stream
        except Exception as e:
            logger.error(f"Error creating audio stream: {e}")
            return DummyStream(self.chunk_size)

    @benchmark("calculate_fft")
    def calculate_fft(self, audio_data, normalize=True, apply_window=True):
        """
        Centralized FFT calculation method with consistent preprocessing
        
        Args:
            audio_data: Input audio data as numpy array
            normalize: Whether to normalize the FFT output (default: True)
            apply_window: Whether to apply windowing function (default: True)
            
        Returns:
            tuple: (fft_data, frequencies) where fft_data is the magnitude spectrum
                   and frequencies are the corresponding frequency bins
        """
        try:
            # Ensure audio data is valid and properly shaped
            if audio_data is None or len(audio_data) == 0:
                return np.zeros(self.chunk_size // 2 + 1, dtype=self.data_format), \
                       np.fft.rfftfreq(self.chunk_size, d=1.0 / self.sample_rate)

            # Ensure audio data is the right shape and type
            if isinstance(audio_data, (bytes, bytearray)):
                audio_data = np.frombuffer(audio_data, dtype=self.data_format)

            # Flatten if multi-dimensional
            if len(audio_data.shape) > 1:
                audio_data = audio_data.flatten()

            # Pad or truncate to expected size
            if len(audio_data) != self.chunk_size:
                temp = np.zeros(self.chunk_size, dtype=self.data_format)
                temp[: min(len(audio_data), self.chunk_size)] = audio_data[
                    : min(len(audio_data), self.chunk_size)
                ]
                audio_data = temp

            # Replace any invalid values
            audio_data = np.nan_to_num(audio_data, nan=0.0, posinf=0.0, neginf=0.0)

            # Apply window function if requested
            if apply_window:
                windowed_data = audio_data * self.window
            else:
                windowed_data = audio_data

            # Perform FFT and get magnitude spectrum
            fft_data = np.abs(np.fft.rfft(windowed_data))

            # Ensure FFT data is valid
            fft_data = np.nan_to_num(fft_data, nan=0.0, posinf=0.0, neginf=0.0)
            np.clip(fft_data, 0, None, out=fft_data)  # Ensure non-negative

            # Normalize if requested
            if normalize and np.max(fft_data) > 0:
                fft_data = fft_data / np.max(fft_data)

            # Calculate frequency bins
            frequencies = np.fft.rfftfreq(len(audio_data), d=1.0 / self.sample_rate)

            return fft_data, frequencies

        except Exception as e:
            logger.error(f"Error in FFT calculation: {e}")
            # Return safe default values
            return np.zeros(self.chunk_size // 2 + 1, dtype=self.data_format), \
                   np.fft.rfftfreq(self.chunk_size, d=1.0 / self.sample_rate)

    def get_latest_fft_data(self):
        """
        Get the latest FFT data from the most recent audio processing
        This method doesn't consume from the main queue to avoid interfering with normal flow

        Returns:
            tuple: (fft_data, frequencies) or (None, None) if no data available
        """
        try:
            # Get the latest audio data using the existing get_audio_data method
            audio_data = self.get_audio_data()
            if audio_data and hasattr(audio_data, "fft_data"):
                frequencies = np.fft.rfftfreq(self.chunk_size, d=1.0 / self.sample_rate)
                return audio_data.fft_data, frequencies
            return None, None
        except Exception as e:
            logger.error(f"Error getting latest FFT data: {e}")
            return None, None

    def start(self):
        """Start the audio processing thread"""
        if not self.running:
            self.running = True
            self.stop_event.clear()
            self.thread = Thread(
                target=self._audio_thread, name="AudioProcessor"
            )
            self.thread.daemon = (
                True  # Thread will stop when main program exits
            )
            self.thread.start()

            with self.stream_lock:
                if self.stream:
                    try:
                        self.stream.start()
                    except Exception as e:
                        logger.error(f"Error starting stream: {e}")
                        self.stream = DummyStream(self.chunk_size)

    def stop(self):
        """Stop the audio processing thread and cleanup resources"""
        if self.running:
            self.running = False
            self.stop_event.set()

            # Stop and close the stream safely
            with self.stream_lock:
                if self.stream:
                    try:
                        if isinstance(self.stream, InputStream):
                            if self.stream.active:
                                self.stream.stop()
                            self.stream.close()
                    except Exception as e:
                        logger.error(f"Error closing stream: {e}")
                    finally:
                        self.stream = None

            # Wait for thread to finish
            if self.thread:
                try:
                    self.thread.join(timeout=1.0)
                except Exception as e:
                    logger.error(f"Error joining thread: {e}")
                finally:
                    self.thread = None

            # Clear the queue
            try:
                while True:
                    self.audio_queue.get_nowait()
            except Empty:
                pass

    def __del__(self):
        """Ensure cleanup on deletion"""
        self.stop()

    @benchmark("audio_process_chunk")
    def _process_audio_chunk(self, audio_data):
        """Process a chunk of audio data with robust error handling and bounds checking"""
        try:
            # Ensure audio data is valid, finite, and properly shaped
            if audio_data is None or len(audio_data) == 0:
                empty_fft = np.zeros(self.chunk_size // 2 + 1, dtype=self.data_format)
                return AudioData(
                    raw_data=np.zeros(self.chunk_size, dtype=self.data_format),
                    fft_data=empty_fft,
                    amplitude=0.0,
                    beat_detected=False,
                    frequency_balance=0.0,
                    energy=0.0,
                    warmth=0.0,
                )

            # Use centralized FFT calculation - this handles all preprocessing
            fft_data, freqs = self.calculate_fft(audio_data, normalize=False, apply_window=True)

            # Get the preprocessed audio data from the FFT calculation
            # We need to redo the preprocessing here to get the cleaned audio_data
            if isinstance(audio_data, (bytes, bytearray)):
                audio_data = np.frombuffer(audio_data, dtype=self.data_format)

            if len(audio_data.shape) > 1:
                audio_data = audio_data.flatten()

            if len(audio_data) != self.chunk_size:
                temp = np.zeros(self.chunk_size, dtype=self.data_format)
                temp[: min(len(audio_data), self.chunk_size)] = audio_data[
                    : min(len(audio_data), self.chunk_size)
                ]
                audio_data = temp

            audio_data = np.nan_to_num(audio_data, nan=0.0, posinf=0.0, neginf=0.0)

            # Calculate amplitude with bounds
            amplitude = np.clip(np.mean(np.abs(audio_data)) * 8.0, 0.0, 1.0)

            # Calculate frequency balance with bounds checking
            frequency_balance = 0.0
            try:
                vocal_range = (50, 250)  # Hz range for vocal frequencies
                vocal_indices = np.logical_and(
                    freqs >= vocal_range[0], freqs <= vocal_range[1]
                )

                # Initialize these variables to avoid reference-before-assignment errors
                vocal_freqs = np.array([])
                vocal_fft = np.array([])

                if np.any(vocal_indices):
                    vocal_freqs = freqs[vocal_indices]
                    vocal_fft = fft_data[vocal_indices]

                    # Only calculate weighted frequency if we have valid data
                    if len(vocal_freqs) > 0 and len(vocal_fft) > 0 and np.sum(vocal_fft) > 0:
                        weighted_freq = np.average(vocal_freqs, weights=vocal_fft)
                        vocal_mid = (vocal_range[1] + vocal_range[0]) / 2
                        frequency_balance = np.clip(
                            (weighted_freq - vocal_mid) / (vocal_range[1] - vocal_range[0]),
                            -1.0,
                            1.0,
                        )

                self.frequency_balance = frequency_balance
                self.rotation_direction = 1 if frequency_balance > 0 else -1
            except Exception as e:
                logger.error(f"Error in frequency balance calculation: {e}")
                frequency_balance = 0.0

            # Beat detection with thread safety
            with self.state_lock:
                beat_detected = self._detect_beat(audio_data, fft_data)

            # Calculate mood metrics with safety bounds
            energy = np.clip(amplitude * 4.0, 0.0, 1.0)
            zero_crossings = np.sum(np.diff(np.signbit(audio_data)))
            warmth = 1.0 - np.clip(zero_crossings / (self.chunk_size / 4), 0.0, 1.0)

            return AudioData(
                raw_data=audio_data,
                fft_data=fft_data,
                amplitude=amplitude,
                beat_detected=beat_detected,
                frequency_balance=frequency_balance,
                energy=energy,
                warmth=warmth,
            )

        except Exception as e:
            logger.error(f"Error in audio processing: {e}")
            # Return safe default values
            empty_fft = np.zeros(self.chunk_size // 2 + 1, dtype=self.data_format)
            return AudioData(
                raw_data=np.zeros(self.chunk_size, dtype=self.data_format),
                fft_data=empty_fft,
                amplitude=0.0,
                beat_detected=False,
                frequency_balance=0.0,
                energy=0.0,
                warmth=0.0,
            )

    def _detect_beat(self, audio_data, fft_data):
        """Detect beats in audio using frequency analysis with robust error handling"""
        try:
            # Focus on bass frequencies (40-200 Hz)
            bass_bins = slice(2, 10)  # ~43-215 Hz with 22050 Hz sample rate
            bass_energy = np.sum(np.clip(fft_data[bass_bins] ** 2, 0.0, np.inf))

            # Sub-bass frequencies (20-60 Hz)
            sub_bass_bins = slice(1, 3)  # ~21-64 Hz
            sub_bass_energy = np.sum(np.clip(fft_data[sub_bass_bins] ** 2, 0.0, np.inf))

            # Calculate RMS energy in dB with safety checks
            bass_rms = np.sqrt(np.maximum(np.mean(bass_energy), 1e-10))
            sub_bass_rms = np.sqrt(np.maximum(np.mean(sub_bass_energy), 1e-10))

            # Use log10 with offset to prevent log(0)
            bass_db = 20 * np.log10(bass_rms + 1e-10)
            sub_bass_db = 20 * np.log10(sub_bass_rms + 1e-10)

            # Combine with weighting
            total_energy = np.clip(bass_db * 0.6 + sub_bass_db * 0.4, -100, 100)

            # Normalize energy with safety bounds
            normalized_energy = np.clip(
                total_energy / (len(audio_data) * 1.5), 0.0, 10.0
            )

            # Initialize buffers if needed
            if not hasattr(self, "energy_buffer") or np.isnan(self.energy_buffer):
                self.energy_buffer = normalized_energy
            if not hasattr(self, "threshold_buffer") or np.isnan(self.threshold_buffer):
                self.threshold_buffer = normalized_energy

            # Smooth energy values with faster attack, slower decay
            if normalized_energy > self.energy_buffer:
                self.energy_buffer = np.clip(
                    0.7 * self.energy_buffer + 0.3 * normalized_energy,
                    0.0,
                    10.0,
                )
            else:
                self.energy_buffer = np.clip(
                    0.95 * self.energy_buffer + 0.05 * normalized_energy,
                    0.0,
                    10.0,
                )

            # Update threshold with beat sensitivity adjustment
            self.threshold_buffer = np.clip(
                0.98 * self.threshold_buffer + 0.02 * self.energy_buffer,
                0.0,
                10.0,
            )

            # Check for beat with sensitivity adjustment
            current_time = time()
            min_time_between_beats = (
                self.min_beat_interval
            )  # Minimum 500ms between beats

            # Calculate adjusted threshold with sensitivity
            beat_sensitivity = np.clip(self.beat_sensitivity, 0.25, 4.0)
            adjusted_threshold = self.threshold_buffer / beat_sensitivity

            if (
                normalized_energy > adjusted_threshold
                and current_time - self.last_beat_time > min_time_between_beats
            ):
                self.last_beat_time = current_time
                # Calculate excess energy with bounds
                self.excess_energy = np.clip(
                    (normalized_energy - self.threshold_buffer) / self.threshold_buffer,
                    -1.0,
                    1.0,
                )
                return True
            return False

        except Exception as e:
            logger.error(f"Error in beat detection: {e}")
            return False

    def _audio_thread(self):
        """Main audio processing thread"""
        while not self.stop_event.is_set():

            try:
                with self.stream_lock:
                    if not self.stream or (
                        isinstance(self.stream, InputStream)
                        and not self.stream.active
                    ):
                        logger.debug("Stream inactive, attempting to restart...")
                        try:
                            if self.stream:
                                try:
                                    if isinstance(self.stream, InputStream):
                                        self.stream.close()
                                except:
                                    pass
                            self.stream = self._create_audio_stream()
                            if isinstance(self.stream, InputStream):
                                self.stream.start()
                        except Exception as e:
                            logger.error(f"Error restarting stream: {e}")
                            self.stream = DummyStream(self.chunk_size)
                            sleep(0.5)
                            continue

                # Read audio data
                try:
                    with self.stream_lock:  # Re-acquire lock briefly for read
                        if self.stream:
                            data, overflowed = self.stream.read(self.chunk_size)
                            if overflowed:
                                logger.debug("Audio buffer overflow")
                        else:
                            data = np.zeros(self.chunk_size, dtype=self.data_format)
                            overflowed = False
                except Exception as e:
                    logger.error(f"Error reading audio stream: {e}")
                    sleep(0.1)
                    continue

                # Process audio data
                try:
                    if isinstance(data, (bytes, bytearray)):
                        audio_data = np.frombuffer(data, dtype=self.data_format)
                    else:
                        audio_data = data

                    if len(audio_data.shape) > 1:
                        audio_data = audio_data.flatten()

                    processed_data = self._process_audio_chunk(audio_data)

                    try:
                        self.audio_queue.put_nowait(processed_data)
                    except Full:
                        try:
                            # Queue is full, remove oldest item and add new one
                            self.audio_queue.get_nowait()
                            self.audio_queue.put_nowait(processed_data)
                        except (Empty, Full):
                            # Queue state changed between operations, skip this frame
                            pass

                except Exception as e:
                    logger.error(f"Error processing audio data: {e}")
                    sleep(0.1)
                    continue
            except PortAudioError as e:
                logger.error(f"PortAudio error in audio thread: {e}")
                with self.stream_lock:
                    try:
                        if self.stream and isinstance(self.stream, InputStream):
                            self.stream.stop()
                            self.stream.close()
                    except:
                        pass
                    finally:
                        self.stream = None

                    sleep(0.5)
                    try:
                        self.stream = self._create_audio_stream()
                        if isinstance(self.stream, InputStream):
                            self.stream.start()
                    except Exception as stream_error:
                        logger.error(f"Error recreating stream: {stream_error}")
                        self.stream = DummyStream(self.chunk_size)
                        sleep(1.0)
            except Exception as e:
                logger.error(f"Error in audio thread: {e}")
                sleep(0.1)

    @benchmark("audio_get_data")
    def get_audio_data(self, timeout=0.1):
        """Get processed audio data from the queue"""
        try:
            data = self.audio_queue.get(timeout=timeout)
            return data
        except Empty:
            return None

    def set_beat_sensitivity(self, sensitivity):
        with self.state_lock:
            self.beat_sensitivity = max(0.1, min(2.0, sensitivity))

    def get_beat_sensitivity(self):
        with self.state_lock:
            return self.beat_sensitivity

    def increase_beat_sensitivity(self):
        current = self.get_beat_sensitivity()
        self.set_beat_sensitivity(current + 0.02)
        logger.debug(f"beat_sensitivity: {self.get_beat_sensitivity()}")

    def decrease_beat_sensitivity(self):
        current = self.get_beat_sensitivity()
        self.set_beat_sensitivity(current - 0.02)
        logger.debug(f"pulse_intensity = {self.get_beat_sensitivity()}")

    def set_chunk_size(self, new_chunk_size):
        """Change the chunk size and restart the audio stream"""
        if new_chunk_size == self.chunk_size:
            return  # No change needed

        logger.debug(f"Changing audio buffer size from {self.chunk_size} to {new_chunk_size}")

        # Stop current processing
        was_running = self.running
        if was_running:
            self.stop()

        # Update chunk size and related parameters
        self.chunk_size = new_chunk_size

        # Re-allocate buffers with new chunk size
        self.audio_buffer = np.zeros(new_chunk_size, dtype=self.data_format)
        self.fft_buffer = np.zeros(new_chunk_size // 2 + 1, dtype=self.data_format)
        self.window = np.hanning(new_chunk_size)

        # Recreate audio stream with new chunk size
        try:
            self.stream = self._create_audio_stream()
        except Exception as e:
            logger.error(f"Error creating audio stream with new chunk size: {e}")
            self.stream = DummyStream(self.chunk_size)

        # Restart if it was running before
        if was_running:
            self.start()

        logger.debug(f"Audio buffer size changed to {new_chunk_size}")

    def get_chunk_size(self):
        """Get the current chunk size"""
        return self.chunk_size

    def set_sample_rate(self, new_sample_rate):
        """Change the sample rate and restart the audio stream"""
        if new_sample_rate == self.sample_rate:
            return  # No change needed

        logger.debug(f"Changing sample rate from {self.sample_rate} to {new_sample_rate}")

        # Stop current processing
        was_running = self.running
        if was_running:
            self.stop()

        # Update sample rate
        self.sample_rate = new_sample_rate

        # Recreate audio stream with new sample rate
        try:
            self.stream = self._create_audio_stream()
        except Exception as e:
            logger.error(f"Error creating audio stream with new sample rate: {e}")
            self.stream = DummyStream(self.chunk_size)

        # Restart if it was running before
        if was_running:
            self.start()

        logger.debug(f"Sample rate changed to {new_sample_rate}")

    def set_device(self, new_device):
        """Change the audio input device and restart the audio stream"""
        if new_device == self.device:
            return  # No change needed

        logger.debug(f"Changing audio device from {self.device} to {new_device}")

        # Stop current processing
        was_running = self.running
        if was_running:
            self.stop()

        # Update device
        self.device = new_device

        # Recreate audio stream with new device
        try:
            self.stream = self._create_audio_stream()
        except Exception as e:
            logger.error(f"Error creating audio stream with new device: {e}")
            self.stream = DummyStream(self.chunk_size)

        # Restart if it was running before
        if was_running:
            self.start()

        logger.debug(f"Audio device changed to {new_device}")

    def set_device(self, new_device):
        """Change the audio input device and restart the audio stream"""
        if new_device == self.device:
            return  # No change needed

        logger.debug(f"Changing audio device from {self.device} to {new_device}")

        # Stop current processing
        was_running = self.running
        if was_running:
            self.stop()

        # Update device
        self.device = new_device

        # Recreate audio stream with new device
        try:
            self.stream = self._create_audio_stream()
        except Exception as e:
            logger.error(f"Error creating audio stream with new device: {e}")
            self.stream = DummyStream(self.chunk_size)

        # Restart if it was running before
        if was_running:
            self.start()

        logger.debug(f"Audio device changed to {new_device}")

    def get_sample_rate(self):
        """Get the current sample rate"""
        return self.sample_rate


class DummyAudioProcessor:
    """A dummy audio processor that provides minimal audio data for testing or when audio is disabled"""

    def __init__(self):
        self.running = False

    def start(self):
        self.running = True

    def stop(self):
        self.running = False

    def get_audio_data(self):
        # Return minimal audio data structure
        import numpy as np

        return AudioData(
            raw_data=np.zeros(1024),
            fft_data=np.zeros(513),  # 1024//2 + 1 = 513 for FFT
            amplitude=0.1,  # Small amplitude for minimal visual activity
            beat_detected=False,
            frequency_balance=0.0,
            energy=0.1,
            warmth=0.5,
        )

    def decrease_beat_sensitivity(self):
        pass

    def increase_beat_sensitivity(self):
        pass

    def set_beat_sensitivity(self, value):
        pass

    def get_chunk_size(self):
        return 256  # Default chunk size

    def set_chunk_size(self, value):
        pass

    def get_sample_rate(self):
        return 44100  # Default sample rate

    def set_sample_rate(self, value):
        pass

    def calculate_fft(self, audio_data, normalize=True, apply_window=True):
        """Dummy FFT calculation that returns empty data"""
        import numpy as np

        chunk_size = 1024 if audio_data is None else len(audio_data)
        fft_size = chunk_size // 2 + 1
        return np.zeros(fft_size), np.fft.rfftfreq(chunk_size, d=1.0 / 22050)
