# Dependencies (run once in your conda env):
#   pip install opencv-python-headless mediapipe
#
# Extracts hand and lip features for the sub-dataset (YX, YZ) and saves them to the specified directories.
#
# Usage example:
#   python scripts/vit_extract_subdataset_features.py \
#       --device cuda:0

import argparse
import os
import os.path as osp
import subprocess
import datetime


def get_parser():
    p = argparse.ArgumentParser(
        description='Extract hand and lip features for the sub-dataset (YX, YZ).'
    )
    p.add_argument('--device', default='cuda:0',
                   help='Device to run the model on (e.g., cuda:0 or cpu)')
    p.add_argument('--batch_size', type=int, default=32,
                   help='Batch size for feature extraction')
    p.add_argument('--log_file', default='extract_features.log',
                   help='Log file path (default: extract_features.log)')
    return p


def log_message(message, log_file=None):
    """Print message to console and write to log file if provided."""
    print(message)
    if log_file:
        with open(log_file, 'a', encoding='utf-8') as f:
            f.write(f"{message}\n")


def run_command(cmd, log_file=None):
    """Run a command and print its output to console and log file."""
    log_message(f"Running: {' '.join(cmd)}", log_file)
    result = subprocess.run(cmd, capture_output=True, text=True)
    log_message(result.stdout, log_file)
    if result.stderr:
        log_message(f"stderr: {result.stderr}", log_file)
    return result.returncode


def main():
    parser = get_parser()
    args = parser.parse_args()
    
    # Paths
    subdataset_root = '/home/uic2/mccsd_sub'
    hand_save_dir = '/home/uic2/mccsd_datasets/Hand_Features'
    lip_save_dir = '/home/uic2/mccsd_datasets/Lip_Features'
    signers = ['YX', 'YZ']
    
    # Initialize log file with timestamp
    log_file = args.log_file
    timestamp = datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')
    log_message(f"\n=== Feature extraction started at {timestamp} ===", log_file)
    log_message(f"Subdataset root: {subdataset_root}", log_file)
    log_message(f"Hand features save dir: {hand_save_dir}", log_file)
    log_message(f"Lip features save dir: {lip_save_dir}", log_file)
    log_message(f"Signers: {', '.join(signers)}", log_file)
    log_message(f"Device: {args.device}", log_file)
    log_message(f"Batch size: {args.batch_size}", log_file)
    
    # Extract hand features
    log_message("\n=== Extracting hand features ===", log_file)
    hand_cmd = [
        'python', 'scripts/vit_extract_hand_feature.py',
        '--video_root', subdataset_root,
        '--save_dir', hand_save_dir,
        '--device', args.device,
        '--batch_size', str(args.batch_size),
        '--signers'] + signers
    
    hand_returncode = run_command(hand_cmd, log_file)
    if hand_returncode != 0:
        log_message(f"Hand feature extraction failed with return code {hand_returncode}", log_file)
        return
    
    # Extract lip features
    log_message("\n=== Extracting lip features ===", log_file)
    lip_cmd = [
        'python', 'scripts/vit_extract_lip_feature.py',
        '--video_root', subdataset_root,
        '--save_dir', lip_save_dir,
        '--device', args.device,
        '--batch_size', str(args.batch_size),
        '--signers'] + signers
    
    lip_returncode = run_command(lip_cmd, log_file)
    if lip_returncode != 0:
        log_message(f"Lip feature extraction failed with return code {lip_returncode}", log_file)
        return
    
    # Final message
    timestamp = datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')
    log_message(f"\n=== Feature extraction completed successfully at {timestamp}! ===", log_file)
    log_message(f"Hand features saved to: {hand_save_dir}", log_file)
    log_message(f"Lip features saved to: {lip_save_dir}", log_file)


if __name__ == '__main__':
    main()
