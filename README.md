# Adalat Evaluation - EC2 Deployment

This repository contains the standalone, production-ready version of the Adalat evaluation pipeline, specifically optimized to run on an Ubuntu 22.04 EC2 instance with an NVIDIA GPU (e.g., g6.2xlarge with L4 VRAM).

## Files Required
Before running the script, ensure you have uploaded your dataset into the same directory as the script:
- `audio.zip` (Contains `.wav` or `.mp3` files)
- `sarvam.csv` (The reference transcripts)

## EC2 Setup Instructions

1. **Connect to your EC2 instance via SSH**:
   ```bash
   ssh -i /path/to/your-key.pem ubuntu@<your-ec2-ip-address>
   ```

2. **Clone this repository (or copy the files over)**:
   ```bash
   git clone <your-github-repo-url>
   cd <your-repo-directory>
   ```

3. **Install NVIDIA Drivers & CUDA (if using a base Ubuntu AMI)**:
   *Note: If you are using the "Deep Learning AMI GPU PyTorch", CUDA is already installed.*
   ```bash
   sudo apt update
   sudo apt install -y python3-pip python3-venv
   ```

4. **Set up a Python Virtual Environment**:
   ```bash
   python3 -m venv adalat_env
   source adalat_env/bin/activate
   ```

5. **Install Dependencies**:
   ```bash
   pip install -r requirements.txt
   ```

6. **Run the Evaluation**:
   ```bash
   python evaluate_adalat.py
   ```

## Checkpointing and Resuming
If your SSH session drops or the script gets interrupted, simply run `python evaluate_adalat.py` again. 
The script automatically reads from `results/transcriptions.csv` and will seamlessly resume from the exact file it left off at without re-transcribing completed audio clips.

## Outputs
Once finished, all data will be available in the following structure:
```
results/
  ├── bucket_assignments.csv     (Your 30 sampled files)
  ├── transcriptions.csv         (Raw Adalat transcriptions)
  ├── evaluation_results.csv     (WER, CER, Similarity per file)
  ├── critical_term_analysis.csv (Keyword spotting hits and misses)
  └── summary_report.txt         (The final aggregate report)
logs/
  └── evaluation.log             (Detailed console output)
```
