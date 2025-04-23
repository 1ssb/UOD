import os
import subprocess
from tqdm import tqdm
import psutil
import pynvml
import time

# Directory containing subdirectories
base_dir = "dataset/test/" # Change this as needed

# Command to run
script_name = "python re10k_triangulate.py"

# Log file path
log_file_path = "./triangulation.log"

# Create log directory
log_dir = "./logs"
os.makedirs(log_dir, exist_ok=True)

# Get the number of available GPUs
pynvml.nvmlInit()
num_gpus = pynvml.nvmlDeviceGetCount()

# Set the batch size to roughly 5 times the number of GPUs
batch_size = num_gpus * 5

def log_gpu_usage(gpu_id):
    handle = pynvml.nvmlDeviceGetHandleByIndex(gpu_id)
    mem_info = pynvml.nvmlDeviceGetMemoryInfo(handle)
    util_info = pynvml.nvmlDeviceGetUtilizationRates(handle)
    temp = pynvml.nvmlDeviceGetTemperature(handle, pynvml.NVML_TEMPERATURE_GPU)
    return mem_info.used / (1024 ** 2), util_info.gpu, temp

def monitor_processes(processes):
    while any(process.poll() is None for _, process in processes):
        for directory, process in processes:
            if process.poll() is None:
                with open(log_file_path, 'a') as log_file:
                    try:
                        ram_usage = psutil.Process(process.pid).memory_info().rss / (1024 ** 2)
                        for gpu_id in range(num_gpus):
                            gpu_mem, gpu_util, gpu_temp = log_gpu_usage(gpu_id)
                            if ram_usage > 120000:  # Safe threshold for high RAM usage (in MB)
                                log_file.write(f"High RAM Usage! Directory: {directory}, RAM Usage: {ram_usage:.2f} MB\n")
                            if gpu_temp > 85:  # Safe threshold for high GPU temperature (in °C)
                                log_file.write(f"High GPU Temperature! Directory: {directory}, GPU {gpu_id}: Temperature: {gpu_temp}°C\n")
                    except psutil.NoSuchProcess:
                        pass
        time.sleep(10)

def run_script(directory):
    """Function to run the script on a directory and capture its output."""
    print(f"Processing directory: {directory}")  # Verbose output
    with open(log_file_path, 'a') as log_file:
        try:
            # Execute the script and redirect output to log file
            process = subprocess.Popen(f"{script_name} -p {directory}", shell=True, stdout=log_file, stderr=subprocess.STDOUT)
            return process
        except Exception as e:
            log_file.write(f"Exception occurred while processing directory: {directory}. Error: {str(e)}\n")
            print(f"Exception occurred while processing directory: {directory}. Error: {str(e)}")
            return None

def main():
    # List all subdirectories in base_dir
    directories = [os.path.join(base_dir, d) for d in os.listdir(base_dir) if os.path.isdir(os.path.join(base_dir, d))]
    
    # Process directories in batches
    for i in tqdm(range(0, len(directories), batch_size), desc="Processing Batches"):
        batch = directories[i:i + batch_size]
        processes = [(directory, run_script(directory)) for directory in batch]
        
        # Monitor the processes
        monitor_processes(processes)
        
        # Check for errors and log completion
        for directory, process in processes:
            if process:
                process.wait()
                if process.returncode != 0:
                    print(f"Error processing directory: {directory}. Check log file for details.")
                else:
                    print(f"Completed processing: {directory}")  # Verbose output

if __name__ == "__main__":
    main()
