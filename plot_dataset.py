import os
import numpy as np
import matplotlib.pyplot as plt

# Define the sample rate
fs = 3.84e9 / 32  

# Channel names corresponding to the 3 channels
channel_names = ["ADC0", "ADC2", "ADC4"]

# SNR index to value mapping (Should be consistent between both python scripts)
target_SNR_dBs = [30, 27, 24, 21, 18, 15, 12, 9, 6, 3, 0, -3, -6, -9, -12, -15, -18, -21, -24, -27, -30]

# Get the directory where the script is located
script_dir = os.path.dirname(os.path.abspath(__file__))

# Get the directory where the processed numpy files are located
numpy_dir = os.path.join(script_dir, "degraded") 

# Loop through all .npy files in the script directory
for fname in os.listdir(numpy_dir):
    if fname.lower().endswith(".npy"):
        file_path = os.path.join(numpy_dir, fname)
        
        name, ext = os.path.splitext(fname)
        
        # Load the combined data (3 x 1000 x signal_length)
        raw_data = np.load(file_path)
        
        """
        print(f"Processing {fname}:")
        print("Shape:", raw_data.shape)
        print("Number of Channels:", len(raw_data))
        print("Number of SNRs:", len(raw_data[0]))
        print("Number of Pulses:", len(raw_data[0][0]))
        print("Number of interleaved samples:", len(raw_data[0][0][0]))
        print("Type of interleaved samples:", type(raw_data[0][0][0]))
        print("Numpy Type of interleaved samples:", raw_data[0][0][0].dtype)
        """

        # Iterate over the three channels
        for channel_idx in range(len(raw_data)):
        
            # Get the channel name
            channel_name = channel_names[channel_idx]
            print(f"Channel: {channel_name}")
        
            for snr_idx in range(len(raw_data[0])):
           
                # Get the SNR Value
                SNR_value = target_SNR_dBs[snr_idx]
                print(f"SNR: {SNR_value}")
           
                for pulse_idx in range(len(raw_data[0][0])):
            
                    # Only plot the first pulse
                    if pulse_idx == 0:
                        row_data = raw_data[channel_idx][snr_idx][pulse_idx]
                        cmplx_data = row_data[0::2] + 1.0j * row_data[1::2]
                        time = np.arange(len(cmplx_data)) / fs
                        power = np.abs(cmplx_data)**2
                        power_db = 10 * np.log10(power + 1e-12)
                    
                        # Plot power (in dB) in time domain
                        plt.figure(figsize=(10, 6))
                        plt.plot(time, power_db, label="Power")
                        plt.legend()
                        plt.title(f"{name}_{channel_name}_SNR{SNR_value}dB_Pulse{pulse_idx} - Power")
                        plt.xlabel("Time (s)")
                        plt.ylabel("Power (dB)")
                        plt.show()
                        plt.close() 
                        #filename = f"{name}_{channel_name}_SNR{SNR_value}dB_Pulse{pulse_idx}_power.png"
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
                        plt.title(f"{name}_{channel_name}_SNR{SNR_value}dB_Pulse{pulse_idx} - I/Q/Mag/Phase")
                        plt.xlabel("Time (s)")
                        plt.ylabel("Power (dB)")
                        plt.show()
                        plt.close() 
                        #filename = f"{name}_{channel_name}_SNR{SNR_value}dB_Pulse{pulse_idx}_components.png"
                        #plt.savefig(filename)

                        # Plot spectrogram
                        plt.figure(figsize=(10, 6))
                        plt.specgram(cmplx_data, NFFT=256, Fs=fs, noverlap=128, cmap="viridis")
                        plt.colorbar(label="Power (dB)")
                        plt.title(f"{name}_{channel_name}_SNR{SNR_value}dB_Pulse{pulse_idx} - Spectrogram")
                        plt.xlabel("Time (s)")
                        plt.ylabel("Frequency (Hz)")
                        plt.show()
                        plt.close()       
                        #filename = f"{name}_{channel_name}_SNR{SNR_value}dB_Pulse{pulse_idx}_spectrogram.png"
                        #plt.savefig(filename)
                        	
