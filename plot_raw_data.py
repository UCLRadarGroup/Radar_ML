import os
import numpy as np
import matplotlib.pyplot as plt

# Define the sample rate
fs = 3.84e9 / 32  

# Channel names corresponding to the 3 channels
channel_names = ["ADC0", "ADC2", "ADC4"]

# Get the directory where the script is located
script_dir = os.path.dirname(os.path.abspath(__file__))

# Loop through all .npy files in the script directory
for fname in os.listdir(script_dir):
    if fname.lower().endswith(".npy"):
        file_path = os.path.join(script_dir, fname)
        
        # Load the combined data (3 x 1000 x signal_length)
        raw_data = np.load(file_path)
        
        print(f"Processing {fname}:")
        print("Shape:", raw_data.shape)
        print("Length:", len(raw_data))
        print("Type of first element:", type(raw_data[0][0]))

        # Iterate over the three channels
        for channel_idx in range(raw_data.shape[0]):
            
            # Get the channel name
            channel_name = channel_names[channel_idx]
            print(f"\nChannel: {channel_name}")

            # Iterate over the rows
            for x in range(len(raw_data[channel_idx])):
                
                # Only plot the last acquisition
                if x == 999:
                    row_data = raw_data[channel_idx][x]
                    cmplx_data = row_data[0::2] + 1.0j * row_data[1::2]
                    time = np.arange(len(cmplx_data)) / fs
                    power = np.abs(cmplx_data)**2
                    power_db = 10 * np.log10(power)
                    
                    # Plot power (in dB) in time domain
                    plt.figure(figsize=(10, 6))
                    plt.plot(time, power_db, label="Power")
                    plt.legend()
                    plt.title(f"{fname}_{channel_name}_{x} - Power")
                    plt.xlabel("Time (s)")
                    plt.ylabel("Power (dB)")
                    plt.show()
                    #filename = f"{fname}_{channel_name}_{x}_power.png"
                    #plt.savefig(filename)
                    
                    # Plot I, Q and Magnitude components in time domain
                    plt.figure(figsize=(10, 6))
                    real_part = np.real(cmplx_data)
                    imag_part = np.imag(cmplx_data)
                    mag_part = np.abs(cmplx_data)
                    phase_part = np.angle(cmplx_data)
                    plt.plot(time, real_part, label="I")
                    plt.plot(time, imag_part, label="Q")
                    plt.plot(time, mag_part, label="Mag")
                    plt.plot(time, phase_part, label="Phase")
                    plt.legend()
                    plt.title(f"{fname}_{channel_name}_{x} - I/Q/Mag/Phase")
                    plt.xlabel("Time (s)")
                    plt.ylabel("Power (dB)")
                    plt.show()
                    #filename = f"{fname}_{channel_name}_{x}_components.png"
                    #plt.savefig(filename)

                    # Plot spectrogram
                    plt.figure(figsize=(10, 6))
                    plt.specgram(cmplx_data, NFFT=256, Fs=fs, noverlap=128, cmap="viridis")
                    plt.colorbar(label="Power (dB)")
                    plt.title(f"{fname}_{channel_name}_{x} - Spectrogram")
                    plt.xlabel("Time (s)")
                    plt.ylabel("Frequency (Hz)")
                    plt.show()          
                    #filename = f"{fname}_{channel_name}_{x}_spectrogram.png"
                    #plt.savefig(filename)

