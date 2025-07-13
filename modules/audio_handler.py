from queue import Empty, Full, Queue
from threading import Event, Lock, Thread
from time import sleep, time
from typing import Any, Dict, List, Optional, Tuple, Union
import numpy as np
from dataclasses import dataclass

from config.constants import CHANNELS, CHUNK, DATA_FORMAT, RATE

from sounddevice import CallbackFlags, InputStream, PortAudioError
from modules.benchmark import benchmark
from modules.logging_config import get_logger

# Get logger for this module
logger = get_logger('audio_handler')


def list_audio_devices() -> List[Dict[str, Any]]:
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


def get_default_audio_device() -> Optional[int]:
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
    beat_count: int
    frequency_balance: float
    energy: float
    warmth: float
    last_beat_time: float


class DummyStream:
    ''' Create dummy audio stream for fallback '''
    def __init__(self, chunk_size: int = CHUNK):
        # Pre-allocate buffer to avoid garbage collection during audio processing
        self.chunk_size = chunk_size
        self.dummy_data = np.zeros((chunk_size,), dtype=DATA_FORMAT)
        # Add callback flags to match the expected return signature
        self.callback_flags = CallbackFlags()
        self.active = True  # Pretend to be active

    def read(self, frames: int) -> Tuple[np.ndarray, CallbackFlags]:
        # Ignore the frames parameter and return the pre-allocated buffer
        # This ensures consistent behavior with the real stream
        return self.dummy_data, self.callback_flags

    def start(self) -> None:
        pass

    def stop(self) -> None:
        pass

    def close(self) -> None:
        pass


class AudioProcessor:
    """Handles audio processing in a separate thread"""

    def __init__(
        self,
        chunk_size: int = CHUNK,
        sample_rate: int = RATE,
        channels: int = CHANNELS,
        data_format: np.dtype = DATA_FORMAT,
        device: Optional[int] = None,
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
        self.threshold_buffer = 0.01  # Much lower initial threshold
        self.last_beat_time = time()
        self.beat_detected = False
        self.beat_count = 0.0
        self.min_beat_interval = 0.05  # Reduced from 0.1 to allow faster beats
        self.excess_energy = 0.0
        self.rotation_direction = 0
        self.frequency_balance = 0.0
        self.beat_sensitivity = 1.5  # Increased default sensitivity from 1.0 to 1.5

        # Mel transform configuration
        self.mel_transform_enabled = True  # Enable mel transform by default
        self.n_mels = 128  # Number of mel bands

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

    def _create_audio_stream(self) -> Union[InputStream, 'DummyStream']:
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

    def apply_logarithmic_scaling(self, fft_data: np.ndarray, min_db: float = -60.0) -> np.ndarray:
        """
        Apply logarithmic scaling to FFT data to match human hearing perception
        
        Args:
            fft_data: Linear magnitude spectrum from FFT
            min_db: Minimum dB level to clamp to (default: -60.0)
            
        Returns:
            np.ndarray: Logarithmically scaled FFT data normalized to [0, 1]
        """
        try:
            # Avoid log(0) by adding small epsilon
            epsilon = 1e-10
            fft_data_safe = np.maximum(fft_data, epsilon)
            
            # Convert to dB scale (20 * log10 for magnitude)
            fft_db = 20.0 * np.log10(fft_data_safe)
            
            # Clamp to minimum dB level
            fft_db = np.maximum(fft_db, min_db)
            
            # Normalize to [0, 1] range
            # Assuming maximum possible dB is 0 (when magnitude = 1.0)
            max_db = 0.0
            fft_normalized = (fft_db - min_db) / (max_db - min_db)
            
            # Ensure values are in [0, 1] range
            fft_normalized = np.clip(fft_normalized, 0.0, 1.0)
            
            return fft_normalized
            
        except Exception as e:
            logger.error(f"Error in logarithmic scaling: {e}")
            # Return original data as fallback
            return fft_data

    def hz_to_mel(self, hz: float) -> float:
        """Convert frequency in Hz to mel scale"""
        return 2595.0 * np.log10(1.0 + hz / 700.0)

    def mel_to_hz(self, mel: float) -> float:
        """Convert mel scale to frequency in Hz"""
        return 700.0 * (10.0**(mel / 2595.0) - 1.0)

    def create_mel_filterbank(self, n_mels: int = 128, fmin: float = 0.0, fmax: Optional[float] = None) -> np.ndarray:
        """
        Create a mel-scale filterbank matrix
        
        Args:
            n_mels: Number of mel bands (default: 128)
            fmin: Minimum frequency in Hz (default: 0.0)
            fmax: Maximum frequency in Hz (default: sample_rate/2)
            
        Returns:
            np.ndarray: Filterbank matrix of shape (n_mels, n_fft_bins)
        """
        try:
            if fmax is None:
                fmax = self.sample_rate / 2.0
            
            # Number of FFT bins
            n_fft_bins = self.chunk_size // 2 + 1
            
            # Create mel-spaced frequency points
            mel_min = self.hz_to_mel(fmin)
            mel_max = self.hz_to_mel(fmax)
            mel_points = np.linspace(mel_min, mel_max, n_mels + 2)
            hz_points = np.array([self.mel_to_hz(mel) for mel in mel_points])
            
            # Convert Hz points to FFT bin indices
            bin_points = np.floor((n_fft_bins - 1) * hz_points / fmax).astype(int)
            bin_points = np.clip(bin_points, 0, n_fft_bins - 1)
            
            # Create filterbank matrix
            filterbank = np.zeros((n_mels, n_fft_bins))
            
            for i in range(n_mels):
                left = bin_points[i]
                center = bin_points[i + 1]
                right = bin_points[i + 2]
                
                # Create triangular filter
                for j in range(left, center):
                    if center != left:
                        filterbank[i, j] = (j - left) / (center - left)
                
                for j in range(center, right):
                    if right != center:
                        filterbank[i, j] = (right - j) / (right - center)
            
            return filterbank
            
        except Exception as e:
            logger.error(f"Error creating mel filterbank: {e}")
            # Return identity-like matrix as fallback
            return np.eye(min(n_mels, self.chunk_size // 2 + 1), self.chunk_size // 2 + 1)

    def create_mel_filterbank_for_size(self, n_mels: int, n_fft_bins: int, fmin: float = 0.0, fmax: Optional[float] = None) -> np.ndarray:
        """
        Create a mel-scale filterbank matrix for a specific FFT size
        
        Args:
            n_mels: Number of mel bands
            n_fft_bins: Number of FFT bins in the input data
            fmin: Minimum frequency in Hz (default: 0.0)
            fmax: Maximum frequency in Hz (default: sample_rate/2)
            
        Returns:
            np.ndarray: Filterbank matrix of shape (n_mels, n_fft_bins)
        """
        try:
            if fmax is None:
                fmax = self.sample_rate / 2.0
            
            # Create mel-spaced frequency points
            mel_min = self.hz_to_mel(fmin)
            mel_max = self.hz_to_mel(fmax)
            mel_points = np.linspace(mel_min, mel_max, n_mels + 2)
            hz_points = np.array([self.mel_to_hz(mel) for mel in mel_points])
            
            # Convert Hz points to FFT bin indices
            bin_points = np.floor((n_fft_bins - 1) * hz_points / fmax).astype(int)
            bin_points = np.clip(bin_points, 0, n_fft_bins - 1)
            
            # Create filterbank matrix
            filterbank = np.zeros((n_mels, n_fft_bins))
            
            for i in range(n_mels):
                left = bin_points[i]
                center = bin_points[i + 1]
                right = bin_points[i + 2]
                
                # Create triangular filter
                for j in range(left, center):
                    if center != left:
                        filterbank[i, j] = (j - left) / (center - left)
                
                for j in range(center, right):
                    if right != center:
                        filterbank[i, j] = (right - j) / (right - center)
            
            return filterbank
            
        except Exception as e:
            logger.error(f"Error creating mel filterbank for size {n_fft_bins}: {e}")
            # Return identity-like matrix as fallback
            return np.eye(min(n_mels, n_fft_bins), n_fft_bins)

    def apply_mel_transform(self, fft_data: np.ndarray, n_mels: int = 128) -> np.ndarray:
        """
        Apply mel-scale transform to logarithmically scaled FFT data
        
        Args:
            fft_data: Logarithmically scaled FFT magnitude spectrum
            n_mels: Number of mel bands to output (default: 128)
            
        Returns:
            np.ndarray: Mel-scaled spectrum of length n_mels
        """
        try:
            # Get actual FFT data size
            n_fft_bins = len(fft_data)
            
            # Create mel filterbank if not cached or if parameters changed
            cache_key = (n_mels, n_fft_bins)
            if not hasattr(self, '_mel_filterbank_cache_key') or self._mel_filterbank_cache_key != cache_key:
                self._mel_filterbank = self.create_mel_filterbank_for_size(n_mels, n_fft_bins)
                self._mel_filterbank_cache_key = cache_key
                logger.debug(f"Created mel filterbank: {n_mels} bands x {n_fft_bins} FFT bins")
            
            # Apply mel filterbank to FFT data
            mel_spectrum = np.dot(self._mel_filterbank, fft_data)
            
            # Ensure non-negative values
            mel_spectrum = np.maximum(mel_spectrum, 0.0)
            
            # Normalize to maintain [0, 1] range
            if np.max(mel_spectrum) > 0:
                mel_spectrum = mel_spectrum / np.max(mel_spectrum)
            
            return mel_spectrum
            
        except Exception as e:
            logger.error(f"Error in mel transform: {e}")
            # Return truncated or padded original data as fallback
            if len(fft_data) >= n_mels:
                return fft_data[:n_mels]
            else:
                padded = np.zeros(n_mels)
                padded[:len(fft_data)] = fft_data
                return padded

    def toggle_mel_transform(self) -> None:
        """Toggle mel transform on/off"""
        self.mel_transform_enabled = not self.mel_transform_enabled
        logger.info(f"Mel transform {'enabled' if self.mel_transform_enabled else 'disabled'}")
        
        # Clear cached filterbank to force recreation with new settings
        if hasattr(self, '_mel_filterbank_cache_key'):
            delattr(self, '_mel_filterbank_cache_key')

    def reset_beat_detection_buffers(self) -> None:
        """Reset beat detection buffers to initial state"""
        with self.state_lock:
            self.energy_buffer = 0.0
            self.beat_count = 0.0
            self.threshold_buffer = 0.01
            self.last_beat_time = time()
            self.beat_detected = False
            self.excess_energy = 0.0
            logger.info("Beat detection buffers reset")
        if hasattr(self, '_mel_filterbank'):
            delattr(self, '_mel_filterbank')

    def set_mel_bands(self, n_mels: int) -> None:
        """Set the number of mel bands"""
        if n_mels > 0 and n_mels <= 512:  # Reasonable limits
            self.n_mels = n_mels
            logger.info(f"Mel bands set to: {n_mels}")
            
            # Clear cached filterbank to force recreation with new settings
            if hasattr(self, '_mel_filterbank_cache_key'):
                delattr(self, '_mel_filterbank_cache_key')
            if hasattr(self, '_mel_filterbank'):
                delattr(self, '_mel_filterbank')
        else:
            logger.warning(f"Invalid number of mel bands: {n_mels}. Must be between 1 and 512.")

    @benchmark("calculate_fft")
    def calculate_fft(
        self, 
        audio_data: np.ndarray, 
        normalize: bool = True, 
        apply_window: bool = True, 
        logarithmic: bool = True, 
        mel_transform: bool = True, 
        n_mels: int = 128
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        Centralized FFT calculation method with consistent preprocessing
        
        Args:
            audio_data: Input audio data as numpy array
            normalize: Whether to normalize the FFT output (default: True)
            apply_window: Whether to apply windowing function (default: True)
            logarithmic: Whether to apply logarithmic scaling for human hearing (default: True)
            mel_transform: Whether to apply mel-scale transform after logarithmic scaling (default: True)
            n_mels: Number of mel bands if mel_transform is True (default: 128)
            
        Returns:
            tuple: (fft_data, frequencies) where fft_data is the magnitude spectrum
                   and frequencies are the corresponding frequency bins
        """
        try:
            # Ensure audio data is valid and properly shaped
            if audio_data is None or len(audio_data) == 0:
                if mel_transform and logarithmic:
                    return np.zeros(n_mels, dtype=self.data_format), \
                           np.linspace(0, self.sample_rate / 2, n_mels)
                else:
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

            # Apply logarithmic scaling for human hearing perception
            if logarithmic:
                fft_data = self.apply_logarithmic_scaling(fft_data)
            elif normalize and np.max(fft_data) > 0:
                # Only apply linear normalization if not using logarithmic scaling
                fft_data = fft_data / np.max(fft_data)

            # Apply mel-scale transform if requested
            if mel_transform and logarithmic:
                original_size = len(fft_data)
                fft_data = self.apply_mel_transform(fft_data, n_mels)
                # Create mel-spaced frequency bins for output
                mel_freqs = np.linspace(0, self.sample_rate / 2, n_mels)
                frequencies = mel_freqs
                # logger.debug(f"Applied mel transform: {original_size} FFT bins -> {n_mels} mel bands")
            else:
                # Calculate standard frequency bins
                frequencies = np.fft.rfftfreq(len(audio_data), d=1.0 / self.sample_rate)

            return fft_data, frequencies

        except Exception as e:
            logger.error(f"Error in FFT calculation: {e}")
            # Return safe default values
            if mel_transform and logarithmic:
                return np.zeros(n_mels, dtype=self.data_format), \
                       np.linspace(0, self.sample_rate / 2, n_mels)
            else:
                return np.zeros(self.chunk_size // 2 + 1, dtype=self.data_format), \
                       np.fft.rfftfreq(self.chunk_size, d=1.0 / self.sample_rate)

    def get_latest_fft_data(self) -> Tuple[Optional[np.ndarray], Optional[np.ndarray]]:
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
                # Since we're using mel transform with 128 bands, create mel-spaced frequencies
                n_mels = len(audio_data.fft_data)
                frequencies = np.linspace(0, self.sample_rate / 2, n_mels)
                return audio_data.fft_data, frequencies
            return None, None
        except Exception as e:
            logger.error(f"Error getting latest FFT data: {e}")
            return None, None

    def start(self) -> None:
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

    def stop(self) -> None:
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

    def __del__(self) -> None:
        """Ensure cleanup on deletion"""
        self.stop()

    @benchmark("audio_process_chunk")
    def _process_audio_chunk(self, audio_data: np.ndarray) -> AudioData:
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
                    beat_count = 0.0,
                    frequency_balance=0.0,
                    energy=0.0,
                    warmth=0.0,
                    last_beat_time=self.last_beat_time,
                )

            # Use centralized FFT calculation - this handles all preprocessing with logarithmic scaling and mel transform
            fft_data, freqs = self.calculate_fft(
                audio_data, 
                normalize=False, 
                apply_window=True, 
                logarithmic=True, 
                mel_transform=self.mel_transform_enabled, 
                n_mels=self.n_mels
            )

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

                # Ifnitialize these variables to avoid reference-before-assignment errors
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
                self.beat_detected = beat_detected
                if beat_detected:
                    self.beat_count += 1
                    

            # Calculate mood metrics with safety bounds
            energy = np.clip(amplitude * 4.0, 0.0, 1.0)
            zero_crossings = np.sum(np.diff(np.signbit(audio_data)))
            warmth = 1.0 - np.clip(zero_crossings / (self.chunk_size / 4), 0.0, 1.0)

            return AudioData(
                raw_data=audio_data,
                fft_data=fft_data,
                amplitude=amplitude,
                beat_detected=beat_detected,
                beat_count=self.beat_count,
                frequency_balance=frequency_balance,
                energy=energy,
                warmth=warmth,
                last_beat_time=self.last_beat_time,
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
                beat_count=0,
                frequency_balance=0.0,
                energy=0.0,
                warmth=0.0,
                last_beat_time=self.last_beat_time,
            )

    def _detect_beat(self, audio_data: np.ndarray, fft_data: np.ndarray) -> bool:
        """Detect beats in audio using frequency analysis optimized for mel-transformed logarithmic data"""
        try:
            n_bands = len(fft_data)
            
            # With mel transform, bass frequencies (20-200 Hz) are concentrated in the first few bands
            # Mel scale gives more resolution to lower frequencies, so we need fewer bands for bass
            if self.mel_transform_enabled:
                # For 128 mel bands, roughly first 15-20 bands cover 20-300 Hz bass range
                bass_band_count = max(3, min(20, int(n_bands * 0.15)))  # Increased from 10% to 15%
                sub_bass_band_count = max(2, min(10, int(n_bands * 0.08)))  # Increased from 5% to 8%
            else:
                # For linear FFT bins, use frequency-based calculation
                nyquist = self.sample_rate / 2
                bass_freq_max = 300  # Hz - increased range
                sub_bass_freq_max = 120  # Hz - increased range
                bass_band_count = max(3, int((bass_freq_max / nyquist) * n_bands))
                sub_bass_band_count = max(2, int((sub_bass_freq_max / nyquist) * n_bands))
            
            # Extract bass energy - since data is already log-scaled [0,1], work directly with it
            bass_bands = fft_data[:bass_band_count]
            sub_bass_bands = fft_data[:sub_bass_band_count]
            
            # Work directly with the log-scaled mel data but emphasize differences
            # The key insight: even in log scale, beats should show relative increases
            
            # Use a power function to expand the differences in the log-scaled data
            bass_expanded = np.power(bass_bands, 0.5)  # Square root to expand small differences
            sub_bass_expanded = np.power(sub_bass_bands, 0.5)
            
            # Calculate weighted sum with emphasis on bass
            bass_energy = np.sum(bass_expanded)
            sub_bass_energy = np.sum(sub_bass_expanded)
            
            # Also include mid-range frequencies for fuller beat detection
            mid_band_count = max(5, min(30, int(n_bands * 0.2)))
            if bass_band_count + mid_band_count <= n_bands:
                mid_bands = fft_data[bass_band_count:bass_band_count + mid_band_count]
                mid_expanded = np.power(mid_bands, 0.5)
                mid_energy = np.sum(mid_expanded) * 0.4
            else:
                mid_energy = 0.0
            
            # Combine energies with strong weighting toward bass
            total_energy = bass_energy * 2.0 + sub_bass_energy * 1.5 + mid_energy
            
            # Normalize and scale to get reasonable values
            normalized_energy = total_energy / max(bass_band_count + sub_bass_band_count, 1)
            
            # Initialize buffers if needed
            if not hasattr(self, "energy_buffer") or np.isnan(self.energy_buffer):
                self.energy_buffer = normalized_energy
            if not hasattr(self, "threshold_buffer") or np.isnan(self.threshold_buffer):
                self.threshold_buffer = max(normalized_energy * 0.3, 0.01)  # Start with 30% of current energy

            # Responsive smoothing for beat detection
            attack_rate = 0.4  # Fast attack to catch beats
            decay_rate = 0.1   # Faster decay to allow for quick changes
            
            if normalized_energy > self.energy_buffer:
                self.energy_buffer = (1 - attack_rate) * self.energy_buffer + attack_rate * normalized_energy
            else:
                self.energy_buffer = (1 - decay_rate) * self.energy_buffer + decay_rate * normalized_energy

            # Adaptive threshold - track a percentage of the smoothed energy
            # Use faster adaptation if threshold is way off
            threshold_diff = abs(self.threshold_buffer - self.energy_buffer * 0.5)
            if threshold_diff > 0.02:  # If threshold is way off, adapt faster
                threshold_rate = 0.05
            else:
                threshold_rate = 0.001  # Slower normal adaptation rate
            
            target_threshold = self.energy_buffer * 0.4  # Target 40% of smoothed energy (more sensitive)
            self.threshold_buffer = (1 - threshold_rate) * self.threshold_buffer + threshold_rate * target_threshold
            
            # Lower minimum threshold for better sensitivity
            self.threshold_buffer = max(self.threshold_buffer, 0.01)

            # Beat detection logic
            current_time = time()
            time_since_last_beat = current_time - self.last_beat_time
            
            # Apply beat sensitivity
            beat_sensitivity = np.clip(self.beat_sensitivity, 0.25, 4.0)
            adjusted_threshold = self.threshold_buffer / beat_sensitivity
            
            # Minimum threshold to prevent false positives
            adjusted_threshold = max(adjusted_threshold, 0.005)
            
            # Energy spike detection to ensure we're detecting actual beats, not noise
            energy_ratio = normalized_energy / max(self.energy_buffer, 0.0001)
            spike_threshold = 1.05  # 5% increase over smoothed energy (more sensitive)
            
            # Beat conditions: energy must exceed threshold AND show a spike pattern
            beat_detected = (
                normalized_energy > adjusted_threshold and
                time_since_last_beat > self.min_beat_interval and
                energy_ratio > spike_threshold
            )

            # Debug logging every 50th frame to see more activity
            if hasattr(self, '_debug_counter'):
                self._debug_counter += 1
            else:
                self._debug_counter = 0
                
            if beat_detected:
                self.last_beat_time = current_time
                # Calculate excess energy for visual effects
                self.excess_energy = np.clip(
                    (normalized_energy - adjusted_threshold) / max(adjusted_threshold, 0.0001),
                    0.0,
                    5.0,  # Increased max excess energy
                )
                # logger.info(f"🎵 BEAT DETECTED! Energy: {normalized_energy:.4f}, Threshold: {adjusted_threshold:.4f}, "
                #          f"Excess: {self.excess_energy:.3f}, Ratio: {energy_ratio:.2f}")
                return True
                
            return False

        except Exception as e:
            logger.error(f"Error in beat detection: {e}")
            return False

    def _audio_thread(self) -> None:
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
    def get_audio_data(self, timeout: float = 0.1) -> Optional[AudioData]:
        """Get processed audio data from the queue"""
        try:
            data = self.audio_queue.get(timeout=timeout)
            return data
        except Empty:
            return None

    def set_beat_sensitivity(self, sensitivity: float) -> None:
        with self.state_lock:
            self.beat_sensitivity = max(0.1, min(2.0, sensitivity))

    def get_beat_sensitivity(self) -> float:
        with self.state_lock:
            return self.beat_sensitivity

    def increase_beat_sensitivity(self) -> None:
        current = self.get_beat_sensitivity()
        self.set_beat_sensitivity(current + 0.02)
        logger.debug(f"beat_sensitivity: {self.get_beat_sensitivity()}")

    def decrease_beat_sensitivity(self) -> None:
        current = self.get_beat_sensitivity()
        self.set_beat_sensitivity(current - 0.02)
        logger.debug(f"pulse_intensity = {self.get_beat_sensitivity()}")

    def reset_beat_detection(self) -> None:
        """Reset beat detection buffers - useful for debugging"""
        with self.state_lock:
            self.energy_buffer = 0.0
            self.threshold_buffer = 0.02
            self.last_beat_time = time()
            logger.info("Beat detection buffers reset")

    def get_beat_detection_stats(self) -> Dict[str, Any]:
        """Get current beat detection statistics for debugging"""
        with self.state_lock:
            return {
                'energy_buffer': getattr(self, 'energy_buffer', 0.0),
                'threshold_buffer': getattr(self, 'threshold_buffer', 0.0),
                'beat_sensitivity': self.beat_sensitivity,
                'min_beat_interval': self.min_beat_interval,
                'mel_transform_enabled': self.mel_transform_enabled,
                'time_since_last_beat': time() - self.last_beat_time,
                'beat_count': self.beat_count
            }

    def set_chunk_size(self, new_chunk_size: int) -> None:
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

    def get_chunk_size(self) -> int:
        """Get the current chunk size"""
        return self.chunk_size

    def set_sample_rate(self, new_sample_rate: int) -> None:
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

    def set_device(self, new_device: int) -> None:
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

    def get_sample_rate(self) -> int:
        """Get the current sample rate"""
        return self.sample_rate


class DummyAudioProcessor:
    """A dummy audio processor that provides minimal audio data for testing or when audio is disabled"""

    def __init__(self) -> None:
        self.running = False

    def start(self) -> None:
        self.running = True

    def stop(self) -> None:
        self.running = False

    def get_audio_data(self) -> AudioData:
        # Return minimal audio data structure
        import numpy as np

        return AudioData(
            raw_data=np.zeros(1024),
            fft_data=np.zeros(513),  # 1024//2 + 1 = 513 for FFT
            amplitude=0.1,  # Small amplitude for minimal visual activity
            beat_detected=False,
            beat_count=0,
            frequency_balance=0.0,
            energy=0.1,
            warmth=0.5,
            last_beat_time=time(),
        )

    def decrease_beat_sensitivity(self) -> None:
        pass

    def increase_beat_sensitivity(self) -> None:
        pass

    def set_beat_sensitivity(self, value: float) -> None:
        pass

    def get_chunk_size(self) -> int:
        return 256  # Default chunk size

    def set_chunk_size(self, value: int) -> None:
        pass

    def get_sample_rate(self) -> int:
        return 44100  # Default sample rate

    def set_sample_rate(self, value: int) -> None:
        pass

    def calculate_fft(
        self, 
        audio_data: np.ndarray, 
        normalize: bool = True, 
        apply_window: bool = True, 
        logarithmic: bool = True
    ) -> Tuple[np.ndarray, np.ndarray]:
        """Dummy FFT calculation that returns empty data"""
        import numpy as np

        chunk_size = 1024 if audio_data is None else len(audio_data)
        fft_size = chunk_size // 2 + 1
        return np.zeros(fft_size), np.fft.rfftfreq(chunk_size, d=1.0 / 22050)
