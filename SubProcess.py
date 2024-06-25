"""Process to sequentially run the World Models"""

import subprocess

def run_script(script_cmd):
    """Runs a Python script with arguments and waits for it to finish."""
    try:
        result = subprocess.run(script_cmd, check=True)
        print(f"Script {script_cmd[0]} finished with return code {result.returncode}")
    except subprocess.CalledProcessError as e:
        print(f"Script {script_cmd[0]} failed with return code {e.returncode}")

def main():
    # List of Python scripts and their arguments to run sequentially
    scripts = [
        ['python3', '/mnt/c/Users/Vidyavarshini/WorldModels_2.0/data/generation_script.py', '--rollouts', '5000', '--rootdir', 'datasets/carracing', '--threads', '8'],
        ['python3', '/mnt/c/Users/Vidyavarshini/WorldModels_2.0/trainvae.py'],
        ['python3', '/mnt/c/Users/Vidyavarshini/WorldModels_2.0/trainmdrnn.py'],
        ['python3', '/mnt/c/Users/Vidyavarshini/WorldModels_2.0/traincontroller.py', '--logdir', './logs', '--n-samples', '4', '--pop-size', '4', '--target-return', '950', '--display']
        # Add the specific arguments for traincontroller.py script
    ]

    # Run each script in order
    for script_cmd in scripts:
        run_script(script_cmd)

if __name__ == "__main__":
    main()
