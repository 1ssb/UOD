# Code by @1ssb
import numpy as np, open3d as o3d, matplotlib.pyplot as plt, json, os, glob, shutil, sqlite3, subprocess, logging, cv2, sys, argparse, torch, psutil
from scipy.spatial.transform import Rotation as R

"""Usage: python re10k_triangulate.py -k (if you want to keep the intermediate workspace files) -p /path/to/dir/with/images/and/text/files/"""

# Setup Logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

def parse_arguments():
    parser = argparse.ArgumentParser(description='Process COLMAP workspace.')
    parser.add_argument('--path', '-p', type=str, default=".", help='The workspace directory full path')
    # Fix the flag type by using 'store_true' for a boolean flag
    parser.add_argument('--keep_intermediate_files', '-k', action='store_true', help='Remove intermediate COLMAP files')
    args = parser.parse_args()
    
    if args.path.strip() == ".":
        logging.info("No path provided. Exiting....")
        exit(1)
    
    return args

# Call parse_arguments and retrieve the parsed args
parsed_args = parse_arguments()
workspace = parsed_args.path.strip()

# Set DELETE_INTERMEDIATES based on the flag
DELETE_INTERMEDIATES = not parsed_args.keep_intermediate_files

def check_workspace_directory(workspace):
    # Ensure the workspace path is absolute and normalized
    workspace = os.path.normpath(os.path.abspath(workspace))

    has_images_directory = os.path.isdir(os.path.join(workspace, 'images'))
    txt_files = glob.glob(os.path.join(workspace, '*.txt'))

    logging.info(f"Images directory {'exists' if has_images_directory else 'does not exist'}.")
    logging.info(f"Found {len(txt_files)} .txt file(s).")

    if len(txt_files) != 1:
        logging.info("There should be exactly one .txt file in the directory.")
        raise Exception("Manual cleanup required: Multiple or no .txt files found.")

    # Define required items explicitly
    required_items = {os.path.join(workspace, 'images')} | set(txt_files)
    all_items = set(glob.glob(os.path.join(workspace, '*')))

    # Identify extra items to remove
    extra_items = all_items - required_items

    if extra_items:
        logging.info("Extra items found which will be deleted:")
        for item in extra_items:
            logging.info(f" - {item}")
            for item in extra_items:
                if os.path.isdir(item):
                    shutil.rmtree(item)
                    logging.info(f"Deleted directory: {item}")
                elif os.path.isfile(item):
                    os.remove(item)
                    logging.info(f"Deleted file: {item}")
        else:
            logging.info("No items were deleted.")
            
def check_required_hardware():
    # Check if CUDA is available
    if not torch.cuda.is_available():
        logging.info("CUDA is not available on this system. Exiting...")
        exit(1)

    # Count the number of CUDA-enabled GPUs
    num_gpus = torch.cuda.device_count()
    if num_gpus == 0:
        logging.info("No CUDA GPUs available.")
        return False, "", "", ""

    # Check total physical memory
    total_memory_gb = psutil.virtual_memory().total / (1024 ** 3)  # Convert bytes to gigabytes
    if total_memory_gb < 16:
        logging.info(f"Not enough RAM: {total_memory_gb:.2f} GB available, 16 GB required.")
        return False, "", "", ""

    # Generate the simple list of GPU indices
    gpus = ','.join(str(i) for i in range(num_gpus))

    # Generate GPU indices for thread assignment, doubling each for assumed two threads per GPU
    gpu_indices = []
    for i in range(num_gpus):
        gpu_indices.append(str(i))
        gpu_indices.append(str(i))  # Add twice for assumed dual-threading

    # Create a string from the list of extended indices
    indices_str = ','.join(gpu_indices)  # No spaces as requested

    # logging.info out details about each GPU
    logging.info(f"Hardware checks passed. Starting process with {num_gpus} GPUs and {total_memory_gb:.2f} GB of RAM.")

    return True, gpus, indices_str, str(total_memory_gb)

def delete_directory(directory):
    """Delete a directory and its contents."""
    if os.path.exists(directory):
        shutil.rmtree(directory)
        logging.info(f"Deleted directory: {directory}")
    else:
        logging.info(f"Directory not found, skipping delete: {directory}")

def rename_output_file(workspace, output_filename, dense):
    """Rename the output file to match the workspace last directory name."""
    workspace_name = os.path.basename(workspace.rstrip('/'))
    new_filename = os.path.join(os.path.dirname(output_filename), f"{workspace_name}_dense.ply") if dense else os.path.join(os.path.dirname(output_filename), f"{workspace_name}_sparse.ply")
    os.rename(output_filename, new_filename)
    logging.info(f"Renamed output file to: {new_filename}")

def compute_fundamental_matrix(P1, P2, K):
    """ Compute the fundamental matrix from camera poses and intrinsic matrices. """
    R1, t1 = P1[:, :3], P1[:, 3]
    R2, t2 = P2[:, :3], P2[:, 3]
    R = R2.T @ R1
    t = R1.T @ (t2 - t1)
    tx = np.array([[0, -t[2], t[1]], [t[2], 0, -t[0]], [-t[1], t[0], 0]])
    Y = np.linalg.solve(K.T, R) @ tx
    F = np.linalg.solve(K.T, Y.T).T
    return F

def calculate_epipolar_error(F, pts1, pts2):
    """ Calculate the epipolar errors given the fundamental matrix and matched points. """
    ones = np.ones((pts1.shape[0], 1))
    pts1_h = np.hstack([pts1, ones])
    pts2_h = np.hstack([pts2, ones])
    epilines1 = F.T @ pts2_h.T
    epilines2 = F @ pts1_h.T
    errors1 = np.abs(np.sum(pts1_h * epilines2.T, axis=1) / np.linalg.norm(epilines2[:2], axis=0))
    errors2 = np.abs(np.sum(pts2_h * epilines1.T, axis=1) / np.linalg.norm(epilines1[:2], axis=0))
    return np.mean(errors1), np.max(errors1), np.mean(errors2), np.max(errors2)

def visualize_errors(errors):
    """ Plot the average and maximum epipolar errors for each image pair and save to file. """
    fig, ax = plt.subplots(2, 1, figsize=(10, 8))
    averages, maxima = zip(*errors)
    ax[0].plot(averages, marker='o', linestyle='-')
    ax[0].set_title('Average Epipolar Errors')
    ax[0].set_xlabel('Image Pair Index')
    ax[0].set_ylabel('Average Error')
    ax[1].plot(maxima, marker='o', linestyle='-')
    ax[1].set_title('Maximum Epipolar Errors')
    ax[1].set_xlabel('Image Pair Index')
    ax[1].set_ylabel('Maximum Error')
    plt.tight_layout()
    plt.savefig(f"{workspace}/epipolar_errors.png")
    plt.close()

def epipolar_computation():
    json_files = glob.glob(f"{workspace}/*.json")
    if not json_files:
        logging.error("No JSON file found in the directory.")
        return False, []
    
    json_file = json_files[0]  # Assumes there's only one JSON file in the directory
    with open(json_file, 'r') as file:
        data = json.load(file)

    image_directory = f"{workspace}/images"
    image_paths = sorted([os.path.join(image_directory, f) for f in os.listdir(image_directory) if f.endswith('.jpg')])

    sift = cv2.SIFT_create()  # SIFT feature detector and descriptor
    errors = []
    max_of_maxes = 0
    error_pairs = []  # List to store pairs with high errors
    triangulation_possible = True  # Assume triangulation is possible initially

    pose_keys = sorted(data['poses'].keys())  # Sort the keys to ensure correct sequential processing
    
    for i in range(len(pose_keys) - 1):
        current_key = pose_keys[i]
        next_key = pose_keys[i + 1]

        img1_path = os.path.join(image_directory, f"{current_key}.jpg")
        img2_path = os.path.join(image_directory, f"{next_key}.jpg")
        img1 = cv2.imread(img1_path, cv2.IMREAD_GRAYSCALE)
        img2 = cv2.imread(img2_path, cv2.IMREAD_GRAYSCALE)

        if img1 is None or img2 is None:
            logging.warning(f"Image missing at {img1_path} or {img2_path}")
            continue

        kp1, des1 = sift.detectAndCompute(img1, None)
        kp2, des2 = sift.detectAndCompute(img2, None)

        matcher = cv2.BFMatcher()
        matches = matcher.knnMatch(des1, des2, k=2)
        good_matches = [m for m, n in matches if m.distance < 0.75 * n.distance]

        if not good_matches:
            continue

        pts1 = np.float32([kp1[m.queryIdx].pt for m in good_matches])
        pts2 = np.float32([kp2[m.trainIdx].pt for m in good_matches])

        P1 = np.array(data['poses'][current_key]['w2c_matrix'])
        P2 = np.array(data['poses'][next_key]['w2c_matrix'])
        K = np.array(data['intrinsic_matrix'])

        F = compute_fundamental_matrix(P1, P2, K)
        avg_error1, max_error1, avg_error2, max_error2 = calculate_epipolar_error(F, pts1, pts2)
        max_error = max(max_error1, max_error2)
        errors.append((0.5 * (avg_error1 + avg_error2), max_error))
        max_of_maxes = max(max_of_maxes, max_error)

        if max_error > 10: # Max Epipolar Error Tolerance is set to 10px
            triangulation_possible = False
            error_pairs.append((current_key, next_key))

    visualize_errors(errors)
    return triangulation_possible, error_pairs

def read_pose_data(file_path):
    """Read pose data from a text file and generate a structured dictionary."""
    if not os.path.exists(file_path):
        raise FileNotFoundError(f"The specified file {file_path} does not exist.")

    with open(file_path, 'r') as file:
        lines = file.readlines()

    data = {'intrinsic_matrix': None, 'poses': {}}
    for i, line in enumerate(lines[1:], 1):
        parts = line.split()
        if len(parts) < 19:
            logging.info(f"Skipping line {i} due to insufficient data.")
            continue
        
        timestamp, params = parts[0], list(map(float, parts[1:]))
        fx, fy, cx, cy = params[:4]
        matrix_values = params[6:18]
        H, W = 360, 640
        if data['intrinsic_matrix'] is None:
            data['intrinsic_matrix'] = [[fx * W, 0, cx * W], [0, fy * H, cy * H], [0, 0, 1]]

        matrix_3x4 = np.array(matrix_values).reshape((3, 4))
        w2c_matrix_4x4 = np.vstack([matrix_3x4, np.array([0, 0, 0, 1])])
        try:
            c2w_matrix = np.linalg.inv(w2c_matrix_4x4)
        except np.linalg.LinAlgError as e:
            logging.info(f"Skipping line {i} due to non-invertible matrix: {e}")
            continue

        rotation_matrix = w2c_matrix_4x4[:3, :3]
        translation_vector = w2c_matrix_4x4[:3, 3]
        try:
            rot = R.from_matrix(rotation_matrix)
            quaternion = rot.as_quat()
            combined_transform = [quaternion[3], quaternion[0], quaternion[1], quaternion[2]] + translation_vector.tolist()
        except ValueError as e:
            logging.info(f"Skipping line {i} due to quaternion conversion error: {e}")
            continue

        data['poses'][timestamp] = {
            "w2c_matrix": w2c_matrix_4x4.tolist(),
            "c2w_matrix": c2w_matrix.tolist(),
            "quaternion": combined_transform
        }   
    return data

def save_files(data, base_path):
    """Save data to JSON and create model/ directory with necessary COLMAP files."""
    output_json_path = os.path.splitext(base_path)[0] + '.json'
    with open(output_json_path, 'w') as file:
        json.dump(data, file, indent=4)

    model_dir = os.path.join(workspace, "model")
    os.makedirs(model_dir, exist_ok=True)
    
    with open(os.path.join(model_dir, 'cameras.txt'), 'w') as file:
        fx, fy, cx, cy = data['intrinsic_matrix'][0][0], data['intrinsic_matrix'][1][1], data['intrinsic_matrix'][0][2], data['intrinsic_matrix'][1][2]
        file.write(f"1 PINHOLE 640 360 {fx} {fy} {cx} {cy}\n")
    
    with open(os.path.join(model_dir, 'images.txt'), 'w') as file:
        for i, (timestamp, pose_data) in enumerate(data['poses'].items(), 1):
            file.write(f"{i} {' '.join(map(str, pose_data['quaternion']))} 1 {timestamp}.jpg\n")

    with open(os.path.join(model_dir, 'points3D.txt'), 'w') as file:
        pass  # This file is left blank

def clean_images_txt():
    """Clean up images.txt based on actual image files in the {workspace}/images/ directory."""
    images_dir = os.path.join(workspace, 'images')
    images_txt_path = os.path.join(workspace, 'model', 'images.txt')
    
    if not os.path.exists(images_dir):
        raise FileNotFoundError("The images directory does not exist.")
    
    if not os.path.exists(images_txt_path):
        raise FileNotFoundError("The images.txt file does not exist in the model directory.")

    missing_images = []
    valid_lines = []
    
    with open(images_txt_path, 'r') as file:
        lines = file.readlines()
    
    for line in lines:
        if line.strip():
            parts = line.strip().split()
            if len(parts) > 9:
                image_name = parts[9]
                image_path = os.path.join(images_dir, image_name)
                if not os.path.exists(image_path):
                    missing_images.append(image_name)
                else:
                    valid_lines.append(line)
    
    if missing_images:
        logging.info("Missing image files:")
        for img in missing_images:
            logging.info(img)
    else:
        logging.info("No missing image files found.")

    with open(images_txt_path, 'w') as file:
        file.writelines(valid_lines)
    logging.info(f"Cleaned images.txt and updated it in the same location: {images_txt_path}")

def clean_point_cloud(input_path, output_path):
    # Load the point cloud from the given path
    pcd = o3d.io.read_point_cloud(input_path)
    logging.info("Point cloud loaded. Point count: %d", len(pcd.points))

    # Perform outlier removal using statistical outlier removal
    clean_pcd, ind = pcd.remove_statistical_outlier(nb_neighbors=10, std_ratio=0.5)
    logging.info("Outliers removed. Remaining points: %d", len(clean_pcd.points))

    # Save the cleaned point cloud to a PLY file
    o3d.io.write_point_cloud(output_path, clean_pcd)
    logging.info(f"Cleaned point cloud saved to {output_path}")
    
def clear_existing_database(database_path):
    """Remove existing database if it exists."""
    if os.path.exists(database_path):
        os.remove(database_path)
        logging.info(f"Removed existing database at {database_path}")

def setup_colmap_database(database_path):
    """Initializes the COLMAP database file."""
    conn = sqlite3.connect(database_path)
    conn.close()
    logging.info("Initialized new database at %s", database_path)

def parse_camera_params(camera_file):
    """Parse the camera parameters from cameras.txt file."""
    if not os.path.exists(camera_file):
        logging.error("The specified camera parameter file does not exist.")
        raise FileNotFoundError("The specified camera parameter file does not exist.")
    with open(camera_file, 'r') as file:
        for line in file:
            parts = line.strip().split()
            if parts[1] == "PINHOLE":
                width, height, fx, fy, cx, cy = map(float, parts[2:8])
                return f"{fx},{fy},{cx},{cy}"
    logging.error("No valid PINHOLE camera parameters found in file.")
    raise ValueError("No valid PINHOLE camera parameters found in file.")

def run_colmap_command(cmd, stage):
    """Run a COLMAP command with error checking."""
    try:
        subprocess.run(cmd, check=True)
        logging.info("%s completed successfully.", stage)
    except subprocess.CalledProcessError as e:
        logging.error("Failed to complete %s: %s", stage, e)
        raise

def check_and_create_dir(directory):
    """Check if a directory exists, if not, create it."""
    if not os.path.exists(directory):
        os.makedirs(directory)
        logging.info("Created directory: %s", directory)

def insert_images_into_database(database_path, images_file):
    """Insert images information into the COLMAP database from images.txt."""
    conn = sqlite3.connect(database_path)
    cursor = conn.cursor()
    total_inserted = 0

    with open(images_file, 'r') as file:
        for line in file:
            if line.strip() and not line.startswith('#'):
                parts = line.strip().split()
                image_id = int(parts[0])
                qw, qx, qy, qz = map(float, parts[1:5])
                tx, ty, tz = map(float, parts[5:8])
                camera_id = int(parts[8])
                name = parts[9]

                cursor.execute("""
                    INSERT OR REPLACE INTO images (image_id, name, camera_id, prior_qw, prior_qx, prior_qy, prior_qz, prior_tx, prior_ty, prior_tz)
                    VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                """, (image_id, name, camera_id, qw, qx, qy, qz, tx, ty, tz))
                total_inserted += 1

    conn.commit()
    conn.close()
    logging.info("Inserted %d images information into the database from images.txt", total_inserted)

def copy_all_images(image_dir, selected_dir):
    """Copy all images to a new directory."""
    check_and_create_dir(selected_dir)
    for image in os.listdir(image_dir):
        if image.endswith('.jpg'):
            src = os.path.join(image_dir, image)
            dst = os.path.join(selected_dir, image)
            shutil.copyfile(src, dst)
    logging.info("Copied all images to %s", selected_dir)

def main():
    is_gpu_available, gpus, gpu_indices, memory = check_required_hardware() # Hardware requirements
    if is_gpu_available:
        check_workspace_directory(workspace)
        database_path = os.path.join(workspace, "database.db")
        image_dir = os.path.join(workspace, "images")
        selected_image_dir = os.path.join(workspace, "selected_images")
        camera_file = os.path.join(workspace, "model", "cameras.txt")
        images_file = os.path.join(workspace, "model", "images.txt")
        input_path = os.path.join(workspace, "model")
        sparse_output_path = os.path.join(workspace, "sparse", "model")
        ply_output_path = os.path.join(workspace, "sparse_model.ply")
        dense_output_path = os.path.join(workspace, "dense")
        dense_sparse_path = os.path.join(dense_output_path, 'sparse/0')
        dense_stereo_path = os.path.join(dense_output_path, 'stereo')

        try:
            if os.path.exists(dense_output_path):
                shutil.rmtree(dense_output_path)
                logging.info(f"'{dense_output_path}' has been removed.")
        except Exception as e:
            logging.error(f"Error while removing dense directory: {e}")
            return

        try:
            txt_files = glob.glob(f"{workspace}/*.txt")
            if len(txt_files) == 1:
                input_file = txt_files[0]
                data = read_pose_data(input_file)
                save_files(data, input_file)
                clean_images_txt()
                logging.info("Files have been successfully created and cleaned.")
            elif len(txt_files) > 1:
                logging.error("Error: More than one .txt file found. Please ensure only one .txt file is present.")
                return
            else:
                logging.error("Error: No .txt files found in the current directory.")
                return
        except Exception as e:
            logging.error(f"Error processing pose data: {e}")
            return
        
        to_triangulate, pairs = epipolar_computation()
        if not to_triangulate:
            logging.info("Problem pairs: ", pairs)
            logging.error("Triangulation Abandoned for this scene. Epipolar Errors are very high, attempt filtering the problem pairs!")

        try:
            clear_existing_database(database_path)
        except Exception as e:
            logging.error(f"Error clearing existing database: {e}")
            return

        try:
            check_and_create_dir(sparse_output_path)
        except Exception as e:
            logging.error(f"Error creating sparse output directory: {e}")
            return

        try:
            setup_colmap_database(database_path)
        except Exception as e:
            logging.error(f"Error setting up COLMAP database: {e}")
            return

        try:
            camera_params = parse_camera_params(camera_file)
        except Exception as e:
            logging.error(f"Error parsing camera parameters: {e}")
            return

        try:
            copy_all_images(image_dir, selected_image_dir)
        except Exception as e:
            logging.error(f"Error copying images: {e}")
            return

        logging.info("=========================== Initial Checks passed --- Starting Triangulation ========================")

        try:
            run_colmap_command([
                'colmap', 'feature_extractor',
                '--database_path', database_path,
                '--image_path', selected_image_dir,
                '--ImageReader.camera_model', 'PINHOLE',
                '--ImageReader.single_camera', '1',
                '--ImageReader.camera_params', camera_params,
                '--SiftExtraction.max_num_features', '50000',
                '--SiftExtraction.use_gpu', '1',
                '--SiftExtraction.gpu_index', gpus,
                '--SiftExtraction.num_threads', '-1'
            ], "Feature extraction")
        except Exception as e:
            logging.error(f"Error during feature extraction: {e}")
            return

        try:
            run_colmap_command([
                'colmap', 'sequential_matcher',
                '--database_path', database_path,
                '--SiftMatching.cross_check', '1',
                '--SiftMatching.guided_matching', '1',
                '--SequentialMatching.overlap', '20',
                '--SiftMatching.max_num_matches', '50000',
                '--SiftMatching.gpu_index', gpus
            ], "Sequential matching")
        except Exception as e:
            logging.error(f"Error during sequential matching: {e}")
            return

        try:
            insert_images_into_database(database_path, images_file)
        except Exception as e:
            logging.error(f"Error inserting images into database: {e}")
            return

        try:
            run_colmap_command([
                'colmap', 'point_triangulator',
                '--database_path', database_path,
                '--image_path', image_dir,
                '--input_path', input_path,
                '--output_path', sparse_output_path,
                '--clear_points', '1',
                '--refine_intrinsics', '0',
                '--Mapper.fix_existing_images', '1',
                '--Mapper.ba_global_function_tolerance', '0',
                '--Mapper.ba_global_max_num_iterations', '200',
                '--Mapper.ba_global_max_refinements', '50',
                '--Mapper.ba_global_max_refinement_change', '0.01'
            ], "Point triangulation")
        except Exception as e:
            logging.error(f"Error during point triangulation: {e}")
            return

        try:
            run_colmap_command([
                'colmap', 'model_converter',
                '--input_path', sparse_output_path,
                '--output_path', ply_output_path,
                '--output_type', 'PLY'
            ], "Model conversion to PLY")
        except Exception as e:
            logging.error(f"Error converting model to PLY: {e}")
            return

        try:
            check_and_create_dir(dense_output_path)
            check_and_create_dir(dense_sparse_path)
            check_and_create_dir(dense_stereo_path)
        except Exception as e:
            logging.error(f"Error creating dense reconstruction directories: {e}")
            return

        try:
            for file_name in ['cameras.bin', 'images.bin', 'points3D.bin']:
                src = os.path.join(sparse_output_path, file_name)
                dst = os.path.join(dense_sparse_path, file_name)
                shutil.copyfile(src, dst)
        except Exception as e:
            logging.error(f"Error copying sparse model files to dense directory: {e}")
            return

        try:
            run_colmap_command([
                'colmap', 'image_undistorter',
                '--image_path', selected_image_dir,
                '--input_path', dense_sparse_path,
                '--output_path', dense_output_path,
                '--output_type', 'COLMAP',
                '--max_image_size', '2000'
            ], "Image undistortion")
        except Exception as e:
            logging.error(f"Error during image undistortion: {e}")
            return

        try:
            run_colmap_command([
                'colmap', 'patch_match_stereo',
                '--workspace_path', dense_output_path,
                '--workspace_format', 'COLMAP',
                '--PatchMatchStereo.window_radius', '5',
                '--PatchMatchStereo.window_step', '1',
                '--PatchMatchStereo.sigma_color', '0.2',
                '--PatchMatchStereo.num_samples', '15',
                '--PatchMatchStereo.ncc_sigma', '0.6',
                '--PatchMatchStereo.cache_size', memory,
                '--PatchMatchStereo.gpu_index', gpu_indices,
                '--PatchMatchStereo.min_triangulation_angle', '1',
                '--PatchMatchStereo.incident_angle_sigma', '0.9',
                '--PatchMatchStereo.geom_consistency', '1'
            ], "Patch match stereo")
        except Exception as e:
            logging.error(f"Error during patch match stereo: {e}")
            return

        try:
            dense_reconstruction_cloud_path = os.path.join(dense_output_path, 'fused.ply')
            run_colmap_command([
                'colmap', 'stereo_fusion',
                '--workspace_path', dense_output_path,
                '--workspace_format', 'COLMAP',
                '--input_type', 'geometric',
                '--output_path', dense_reconstruction_cloud_path
            ], "Stereo fusion")
        except Exception as e:
            logging.error(f"Error during stereo fusion: {e}")
            return

        try:
            final_output_cloud = os.path.join(workspace, "dense_filtered_output.ply")
            clean_point_cloud(dense_reconstruction_cloud_path, final_output_cloud)
        except Exception as e:
            logging.error(f"Error cleaning point cloud: {e}")
            return
        
        # Rename the sparse output file
        final_output_cloud = os.path.join(workspace, "sparse_model.ply")
        rename_output_file(workspace, final_output_cloud, dense = False)
        
        if DELETE_INTERMEDIATES:
            # Delete the selected_images and model directories and all that COLMAP has created
            delete_directory(selected_image_dir)
            delete_directory(os.path.join(workspace, "model"))
            delete_directory(os.path.join(workspace, "sparse"))
            delete_directory(dense_output_path)
            os.remove(database_path)
            
        # Rename the dense output file
        final_output_cloud = os.path.join(workspace, "dense_filtered_output.ply")
        rename_output_file(workspace, final_output_cloud, dense = True)

    else:
        logging.error("Device requirements not met! Exiting....")
        exit(1)
        
if __name__ == "__main__":
    main()