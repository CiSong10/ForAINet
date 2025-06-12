import os
import subprocess
import json
import argparse

def generate_eval_command(ply_paths, eval_script="eval.py"):

    # Create the data.fold argument string with escaped double quotes
    fold_argument = json.dumps(ply_paths, ensure_ascii=False)

    # Escape the double quotes for the shell command
    escaped_fold_argument = fold_argument.replace('"', '\\"')

    # Generate the full main.py command
    eval_command = f'python {eval_script} data.fold="{escaped_fold_argument}"'
    
    return eval_command

def main():
    # Path to the ply_output_files.txt
    # ply_output_file = '/home/ubuntu/binbin/OutdoorPanopticSeg_V2/data_test/tiles_50_ULS_Åsmåsan_VUX1_2017_80_120mAGL/ply_output_file_paths.txt'
    
    parser = argparse.ArgumentParser(description="Generate evaluation commands.")
    parser.add_argument('--ply-output-file', type=str, required=True, help='Path to the ply output file paths.')

    args = parser.parse_args()

    ply_output_file = args.ply_output_file

    # Read all .ply file paths from ply_output_files.txt
    with open(ply_output_file, 'r') as file:
        ply_paths = [line.strip() for line in file.readlines()]

    # Batch size
    batch_size = 5

    # Process in batches of batch_size
    for i in range(0, len(ply_paths), batch_size):
        batch_paths = ply_paths[i:i + batch_size]
        
        # Generate the eval command for the current batch
        eval_command = generate_eval_command(batch_paths)
        
        # Print the eval command (optional)
        print(eval_command)
        
        # Run the eval command
        result = subprocess.run(eval_command, shell=True, capture_output=True, text=True)
        
        # Print the command output
        print(result.stdout)
        print(result.stderr)

if __name__ == "__main__":
    main()
