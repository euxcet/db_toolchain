import wave
import numpy as np
import matplotlib.pyplot as plt

# Function to read and plot a WAV file
def plot_wav(filename):
    # Open the WAV file
    with wave.open(filename, 'r') as wav_file:
        # Extract basic parameters
        n_channels = wav_file.getnchannels()
        sampwidth = wav_file.getsampwidth()
        framerate = wav_file.getframerate()
        n_frames = wav_file.getnframes()

        # Read frames and convert to numpy array
        frames = wav_file.readframes(n_frames)
        waveform = np.frombuffer(frames, dtype=np.int16)

        # If stereo, take only one channel
        if n_channels == 2:
            waveform = waveform[::2]

        # Generate time axis
        time = np.linspace(0, n_frames / framerate, num=n_frames)

        # Plot the waveform
        plt.figure(figsize=(10, 4))
        plt.plot(time, waveform, label='Waveform')
        plt.xlabel('Time [s]')
        plt.ylabel('Amplitude')
        plt.title('Waveform of {}'.format(filename))
        plt.legend()
        plt.grid()
        plt.show()

# Example usage
filename = '1.wav'
plot_wav(filename)
