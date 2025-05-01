import cv2
import shutil
import logging
import os
import glob
import sys
import yt_dlp as youtube_dl
from multiprocessing import Pool, cpu_count
from tqdm import tqdm
from typing import List

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

class Data:
    """
    Represents the metadata for a video sequence.

    Attributes:
        url (str): The URL of the video.
        seqname (str): The name of the sequence.
        list_timestamps (List[int]): A list of timestamps for frame extraction.
    """
    def __init__(self, url: str, seqname: str, list_timestamps: List[int]):
        self.url = url
        self.seqname = seqname
        self.list_timestamps = list_timestamps

def download_video(url: str, output_dir: str, title: str, retries: int = 3) -> bool:
    """
    Downloads a video from a URL.

    Args:
        url (str): The URL of the video.
        output_dir (str): The directory to save the downloaded video.
        title (str): The title for the saved video file.
        retries (int): The number of retry attempts for downloading.

    Returns:
        bool: True if the download was successful, False otherwise.
    """
    ydl_opts = {
        'format': 'bestvideo[ext=mp4]+bestaudio[ext=mp4]/best[ext=mp4]/best',
        'outtmpl': os.path.join(output_dir, f'{title}.%(ext)s'),
        'noplaylist': True,
        'quiet': True,
    }
    for attempt in range(retries):
        with youtube_dl.YoutubeDL(ydl_opts) as ydl:
            try:
                ydl.download([url])
                logging.info(f"Downloaded video for: {title}")
                return True
            except Exception as e:
                logging.error(f"Download failed for {url} (attempt {attempt + 1}/{retries}): {e}")
    return False

def validate_video(videoname: str) -> bool:
    """
    Validates if the video file can be opened.

    Args:
        videoname (str): The path to the video file.

    Returns:
        bool: True if the video is valid, False otherwise.
    """
    cap = cv2.VideoCapture(videoname)
    valid = cap.isOpened()
    cap.release()
    if not valid:
        logging.error(f"Video validation failed, unable to open: {videoname}")
    return valid

def extract_frames(data: Data, videoname: str, output_root: str) -> bool:
    """
    Extracts frames from a video at specified timestamps.

    Args:
        data (Data): The metadata for the video sequence.
        videoname (str): The path to the video file.
        output_root (str): The root directory to save the extracted frames.

    Returns:
        bool: True if frames were extracted successfully, False otherwise.
    """
    output_dir = os.path.join(output_root, data.seqname)
    os.makedirs(output_dir, exist_ok=True)

    cap = cv2.VideoCapture(videoname)
    if not cap.isOpened():
        logging.error(f"Failed to open video file: {videoname}")
        return False

    for timestamp in data.list_timestamps:
        timestamp_sec = timestamp / 1000.0
        cap.set(cv2.CAP_PROP_POS_MSEC, timestamp_sec)
        ret, frame = cap.read()
        if ret:
            frame_output_path = os.path.join(output_dir, f"{timestamp}.jpg")
            cv2.imwrite(frame_output_path, frame)
            if not validate_and_resize_frame(frame_output_path):
                logging.error(f"Extracted frame at {timestamp_sec} seconds is corrupted or has incorrect size for {videoname}")
                os.remove(frame_output_path)
        else:
            logging.error(f"Failed to extract frame at {timestamp_sec} seconds for {videoname}")

    cap.release()
    return True

def validate_and_resize_frame(image_path: str) -> bool:
    """
    Validates and resizes the extracted frame to 640x360.

    Args:
        image_path (str): The path to the extracted frame image.

    Returns:
        bool: True if the frame was validated and resized successfully, False otherwise.
    """
    try:
        img = cv2.imread(image_path)
        resized_img = cv2.resize(img, (640, 360))
        cv2.imwrite(image_path, resized_img)
        return True
    except Exception as e:
        logging.error(f"Failed to validate or resize frame {image_path}: {e}")
        return False

def cleanup_videos(video_dir: str):
    """
    Cleans up the downloaded video files.

    Args:
        video_dir (str): The directory containing the downloaded video files.
    """
    parent_dir = os.path.dirname(video_dir.rstrip('/'))
    if os.path.exists(parent_dir):
        shutil.rmtree(parent_dir)
        logging.info(f"Deleted video directory and its parent: {parent_dir}")
    else:
        logging.warning(f"Parent directory not found: {parent_dir}")

def process_data(data: Data, video_dir: str, output_root: str) -> bool:
    """
    Processes a single data sequence: downloads the video, validates it, extracts frames.

    Args:
        data (Data): The metadata for the video sequence.
        video_dir (str): The directory to save the downloaded video.
        output_root (str): The root directory to save the extracted frames.

    Returns:
        bool: True if the data was processed successfully, False otherwise.
    """
    output_dir = os.path.join(output_root, data.seqname)
    
    # Skip if frames already exist
    if all(os.path.exists(os.path.join(output_dir, f"{timestamp}.jpg")) for timestamp in data.list_timestamps):
        logging.info(f"Frames for {data.seqname} already exist. Skipping download and extraction.")
        return True
    
    if download_video(data.url, video_dir, data.seqname):
        videoname = max(glob.glob(os.path.join(video_dir, '*.mp4')), key=os.path.getctime)
        if validate_video(videoname):
            if extract_frames(data, videoname, output_root):
                logging.info(f"Extracted frames for {data.seqname}")
                return True
            else:
                logging.error(f"Failed to extract frames for {data.seqname}")
        else:
            logging.error(f"Downloaded video is corrupted or incomplete: {data.seqname}")
        os.remove(videoname)
    return False

class DataDownloader:
    """
    Manages the downloading and processing of video data sequences.

    Attributes:
        dataroot (str): The root directory of the data.
        mode (str): The mode of operation, 'test' or 'train'.
        max_scenes (int): The maximum number of scenes to process.
        video_dir (str): The directory to save the downloaded videos.
        output_root (str): The root directory to save the extracted frames.
        list_data (List[Data]): The list of data sequences to process.
    """
    def __init__(self, dataroot: str, mode: str = 'test', max_scenes: int = 10):
        self.dataroot = dataroot
        self.mode = mode
        self.max_scenes = max_scenes
        self.video_dir = f'./temp_videos/{mode}/'
        self.output_root = f'./dataset/{mode}/'
        os.makedirs(self.video_dir, exist_ok=True)
        os.makedirs(self.output_root, exist_ok=True)
        self.list_data = self.prepare_data()

    def prepare_data(self) -> List[Data]:
        """
        Prepares the list of data sequences to be processed.

        Returns:
            List[Data]: The list of data sequences.
        """
        data_list = []
        list_seqnames = sorted(glob.glob(os.path.join(self.dataroot, '*.txt')))
        for txt_file in list_seqnames:
            seq_name = os.path.basename(txt_file).split('.')[0]
            with open(txt_file, "r") as seq_file:
                lines = seq_file.readlines()
                youtube_url = lines[0].strip()
                list_timestamps = [int(line.split()[0]) for line in lines[1:]]
                data_list.append(Data(youtube_url, seq_name, list_timestamps))
        return data_list

    def run(self):
        """
        Runs the downloading and processing of video data sequences.
        """
        processed_count = 0
        current_index = 0

        while processed_count < self.max_scenes and current_index < len(self.list_data):
            data_batch = self.list_data[current_index:current_index + self.max_scenes - processed_count]
            with Pool(processes=cpu_count()) as pool:
                results = list(tqdm(pool.imap_unordered(self.process_single_data, data_batch), total=len(data_batch)))
            processed_count += sum(results)
            current_index += len(data_batch)
            if processed_count < self.max_scenes:
                logging.info(f"Processed {processed_count} scenes, looking for more data to process...")

        cleanup_videos(self.video_dir)
        logging.info("Download and frame extraction task completed.")

    def process_single_data(self, data: Data) -> bool:
        """
        Processes a single data sequence.

        Args:
            data (Data): The metadata for the video sequence.

        Returns:
            bool: True if the data was processed successfully, False otherwise.
        """
        output_dir = os.path.join(self.output_root, data.seqname)
        if all(os.path.exists(os.path.join(output_dir, f"{timestamp}.jpg")) for timestamp in data.list_timestamps):
            logging.info(f"Frames for {data.seqname} already exist. Skipping download and extraction.")
            return False
        return process_data(data, self.video_dir, self.output_root)

if __name__ == "__main__":
    if len(sys.argv) != 3:
        print("Usage: script.py [test | train] [max_scenes | ".")]")
        sys.exit(1)

    mode = sys.argv[1]
    max_scenes = sys.argv[2]

    if mode not in ["test", "train"]:
        print("Invalid mode")
        sys.exit(1)

    dataroot = f"./RealEstate10K/{mode}"
    max_scenes = len(glob.glob(os.path.join(dataroot, '*.txt'))) if max_scenes == '.' else int(max_scenes)
    
    downloader = DataDownloader(dataroot, mode, max_scenes)
    downloader.run()
