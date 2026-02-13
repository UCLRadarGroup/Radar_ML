import os
import numpy as np
import matplotlib.pyplot as plt
import hashlib
import multiprocessing
from tqdm import tqdm
from concurrent.futures import ProcessPoolExecutor


# USER TO SET: Number of pulses per SNR value:
repeats_per_snr = 300

# Sampling frequency in Hz
fs = 3.84e9 / 32  

# Channel names corresponding to the 3 channels
channel_names = ["ADC0", "ADC2", "ADC4"]

# Selected as the optimum seed after 7.5 million years of computation
global_seed = 42
np.random.seed(global_seed)

# USER TO SET: List of desired SNRs
target_SNR_dBs = [30, 27, 24, 21, 18, 15, 12, 9, 6, 3, 0, -3, -6, -9, -12, -15, -18, -21, -24, -27, -30]

# Get the directory where the script is located
script_dir = os.path.dirname(os.path.abspath(__file__))

data_dir = os.path.join(script_dir, "unprocessed")
out_dir = os.path.join(script_dir, "processed")
os.makedirs(out_dir, exist_ok=True)

def process_file(fname):
    file_path = os.path.join(data_dir, fname)

    # 3 Sensors, N SNRs, N repeats, 33600 samples
    degraded_array = np.zeros((3, len(target_SNR_dBs), repeats_per_snr, 33600), dtype=np.int16)

    # Load raw data: shape [3, 1000, N]
    raw_data = np.load(file_path)

    for channel_idx in range(raw_data.shape[0]):
        for x in range(repeats_per_snr):
            for snr_idx, target_SNR_dB in enumerate(target_SNR_dBs):
            
                # Generate repeatable hash
                combined_string = f"{global_seed}_{fname}_{target_SNR_dB}_{x}"
                hash_object = hashlib.md5(combined_string.encode())
                unique_seed = int(hash_object.hexdigest()[:8], 16)
                noise_rng = np.random.default_rng(unique_seed)

                row_data = raw_data[channel_idx][x]
                cmplx_data = row_data[0::2] + 1.0j * row_data[1::2]

                # Calculate power estimates
                noise = cmplx_data[0:int((20e-6)*fs)]
                noise_power = np.mean(np.abs(noise) ** 2)
                signal = cmplx_data[int((21e-6)*fs):int((119e-6)*fs)]
                signal_power = np.mean(np.abs(signal) ** 2)

                target_SNR = 10 ** (target_SNR_dB / 10)
                additive_noise_power = signal_power / target_SNR

                # Generate Gaussian noise
                length = len(cmplx_data)
                noise_I = noise_rng.normal(0, np.sqrt(additive_noise_power / 2), length)
                noise_Q = noise_rng.normal(0, np.sqrt(additive_noise_power / 2), length)
                noise_cmplx = noise_I + 1j * noise_Q
                
                # Add noise to our signal
                degraded = cmplx_data + noise_cmplx

                # Interleave
                row_data_degraded = np.empty(2 * len(degraded), dtype=np.int16)
                row_data_degraded[0::2] = degraded.real.astype(np.int16)
                row_data_degraded[1::2] = degraded.imag.astype(np.int16)

                degraded_array[channel_idx, snr_idx, x, :] = row_data_degraded

    # Save output
    name, ext = os.path.splitext(fname)
    new_name = name + "_degraded" + ext
    new_loc = os.path.join(out_dir, new_name)
    np.save(new_loc, degraded_array)
        
if __name__ == "__main__":
    # Get all .npy files in data_dir
    npy_files = [f for f in os.listdir(data_dir) if f.lower().endswith(".npy")]

    print(f"Found {len(npy_files)} .npy files to process in parallel.\n")

    num_cpus = multiprocessing.cpu_count()
    max_workers = max(1, int(num_cpus * 0.8))
    with ProcessPoolExecutor(max_workers=max_workers) as executor:
        list(tqdm(executor.map(process_file, npy_files), total=len(npy_files)))

    print("All files processed.")

