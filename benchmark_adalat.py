import os
import sys
import glob
import time
import zipfile
import logging
import argparse
import pandas as pd
import numpy as np
import librosa
import torch
import jiwer
from rapidfuzz.distance import Levenshtein
from sklearn.preprocessing import MinMaxScaler
from tqdm import tqdm
from faster_whisper import WhisperModel

# ==========================================
# 1. SETUP & LOGGING
# ==========================================
os.makedirs("results", exist_ok=True)
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    handlers=[
        logging.FileHandler("benchmark.log"),
        logging.StreamHandler(sys.stdout)
    ]
)
logger = logging.getLogger(__name__)

# ==========================================
# 2. AUDIO PROFILING
# ==========================================
def profile_audio(path):
    try:
        y, sr = librosa.load(path, sr=None)
        duration = librosa.get_duration(y=y, sr=sr)
        rms = np.sqrt(np.mean(y**2))
        intervals = librosa.effects.split(y, top_db=30)
        non_silent = sum([(e - s) for s, e in intervals]) / sr
        silence_ratio = 1.0 - (non_silent / duration) if duration > 0 else 0
        dynamic_range = 20 * np.log10(np.max(np.abs(y)) / (np.min(np.abs(y[y!=0])) + 1e-9) + 1e-9)
        signal_variance = np.var(y)
        clipping_ratio = np.sum(np.abs(y) >= 0.99) / len(y)
        return pd.Series([duration, sr, rms, silence_ratio, dynamic_range, signal_variance, clipping_ratio])
    except Exception as e:
        logger.error(f"Error profiling {path}: {e}")
        return pd.Series([np.nan]*7)

# ==========================================
# 3. METRICS
# ==========================================
wer_transform = jiwer.Compose([jiwer.ToLowerCase(), jiwer.RemoveMultipleSpaces(), jiwer.RemovePunctuation(), jiwer.Strip()])

def compute_wer_cer_sim(ref, hyp):
    ref_str = str(ref)
    hyp_str = str(hyp)
    if not ref_str.strip(): return np.nan, np.nan, 0.0
    try:
        ref_clean = wer_transform(ref_str)
        hyp_clean = wer_transform(hyp_str)
        w = jiwer.wer(ref_clean, hyp_clean)
        c = jiwer.cer(ref_clean, hyp_clean)
    except:
        w = jiwer.wer(ref_str, hyp_str)
        c = jiwer.cer(ref_str, hyp_str)
    from rapidfuzz import fuzz
    s = fuzz.ratio(ref_str, hyp_str)
    return w, c, s

def compute_keyword_spotting(ref_text, hyp_text):
    ref_words = str(ref_text).split()
    hyp_words = str(hyp_text).split()
    if not ref_words: return 1.0, "", ""
    matched, missed = [], []
    for rw in ref_words:
        if len(rw) == 0: continue
        is_detected = False
        for hw in hyp_words:
            if len(hw) == 0: continue
            cer = Levenshtein.distance(rw, hw) / len(rw)
            if cer <= 0.30:
                is_detected = True
                break
        if is_detected: matched.append(rw)
        else: missed.append(rw)
    return len(matched)/len(ref_words), ", ".join(matched), ", ".join(missed)

# ==========================================
# 4. MAIN PIPELINE
# ==========================================
def main():
    parser = argparse.ArgumentParser(description="Adalat-AI Production Benchmark")
    parser.add_argument("--audio_zip", required=True, help="Path to Hindi Audio Zip")
    parser.add_argument("--sarvam_csv", required=True, help="Path to Sarvam Transcripts CSV")
    args = parser.parse_args()

    logger.info("Extracting Audio Zip...")
    audio_dir = "audio_extracted"
    os.makedirs(audio_dir, exist_ok=True)
    with zipfile.ZipFile(args.audio_zip, 'r') as zip_ref:
        zip_ref.extractall(audio_dir)

    logger.info("Matching Audio Files...")
    audio_paths = glob.glob(f'{audio_dir}/**/*.wav', recursive=True) + glob.glob(f'{audio_dir}/**/*.mp3', recursive=True)
    audio_df = pd.DataFrame({'full_path': audio_paths})
    audio_df['clip_id'] = audio_df['full_path'].apply(lambda x: os.path.splitext(os.path.basename(x))[0])

    sarvam_df = pd.read_csv(args.sarvam_csv)
    if 'clip_id' not in sarvam_df.columns and 'audio' in sarvam_df.columns:
        sarvam_df['clip_id'] = sarvam_df['audio'].astype(str).apply(lambda x: os.path.splitext(os.path.basename(x))[0])

    master_df = pd.merge(audio_df, sarvam_df[['clip_id', 'sarvam_transcript']], on='clip_id', how='inner')
    logger.info(f"Matched {len(master_df)} total files.")

    if master_df.empty:
        logger.error("No matching files found. Exiting.")
        return

    logger.info("Profiling Audio...")
    tqdm.pandas(desc='Profiling')
    cols = ['duration', 'sample_rate', 'rms_energy', 'silence_ratio', 'dynamic_range', 'signal_variance', 'clipping_ratio']
    master_df[cols] = master_df['full_path'].progress_apply(profile_audio)
    master_df = master_df[master_df['duration'] > 0]

    logger.info("Running Hybrid Quality Bucketing...")
    scaler = MinMaxScaler()
    norm_rms = scaler.fit_transform(master_df[['rms_energy']])
    norm_var = scaler.fit_transform(master_df[['signal_variance']])
    norm_dr = scaler.fit_transform(master_df[['dynamic_range']])
    norm_silence = 1 - scaler.fit_transform(master_df[['silence_ratio']])
    norm_clipping = 1 - scaler.fit_transform(master_df[['clipping_ratio']])
    
    master_df['quality_score'] = (0.2 * norm_rms + 0.2 * norm_var + 0.1 * norm_dr + 0.2 * norm_silence + 0.2 * norm_clipping)
    q33, q66 = master_df['quality_score'].quantile([0.33, 0.66])
    def assign_bucket(s): return 'GOOD' if s >= q66 else ('MODERATE' if s >= q33 else 'BAD')
    master_df['audio_bucket'] = master_df['quality_score'].apply(assign_bucket)

    logger.info("Random Sampling (2 per bucket)...")
    eval_df = master_df.groupby('audio_bucket', group_keys=False).apply(lambda x: x.sample(n=min(2, len(x)), random_state=42)).copy()
    logger.info(f"Selected {len(eval_df)} files.")

    logger.info("Initializing Silero VAD exactly once...")
    vad_model, utils = torch.hub.load(repo_or_dir='snakers4/silero-vad', model='silero_vad', force_reload=False, onnx=False)
    (get_speech_timestamps, save_audio, read_audio, VADIterator, collect_chunks) = utils

    logger.info("Loading Faster-Whisper exactly once (adalat-ai/whisper-medium-hi-high-lr)...")
    # Note: faster-whisper requires CTranslate2 models. If the HF repo is not CTranslate2 converted, 
    # you must use `ct2-transformers-converter` prior to running, or standard HuggingFace pipeline.
    # We attempt to load it directly as requested.
    try:
        model = WhisperModel("adalat_ct2", device="cuda", compute_type="float16")
    except Exception as e:
        logger.error(f"Failed to load faster-whisper model: {e}")
        logger.error("Please ensure the HF repo contains CTranslate2 weights, or convert it locally first.")
        return

    results_list = []
    logger.info("Starting Inference Pipeline...")
    
    for idx, row in tqdm(eval_df.iterrows(), total=len(eval_df), desc="Inference"):
        start_t = time.time()
        wav = read_audio(row['full_path'], sampling_rate=16000)
        speech_timestamps = get_speech_timestamps(wav, vad_model, sampling_rate=16000)
        
        seg_count = len(speech_timestamps) if speech_timestamps else 1
        total_dur = sum([(s['end'] - s['start'])/16000 for s in speech_timestamps]) if speech_timestamps else len(wav)/16000
        avg_dur = total_dur / seg_count if seg_count > 0 else 0
        
        try:
            # Use faster-whisper's native VAD on the entire audio array to preserve sentence context.
            # We use beam_size=1 (greedy) to perfectly match HuggingFace Pipeline's default Kaggle behavior.
            segments, _ = model.transcribe(
                wav.numpy(), 
                beam_size=1, 
                language="hi",
                vad_filter=True,
                vad_parameters=dict(min_silence_duration_ms=500)
            )
            transcript = ' '.join([seg.text for seg in segments]).strip()
        except Exception as e:
            logger.error(f"Transcription error on {row['clip_id']}: {e}")
            transcript = ""
        
        # Metrics
        w, c, s = compute_wer_cer_sim(row['sarvam_transcript'], transcript)
        k_rec, k_match, k_miss = compute_keyword_spotting(row['sarvam_transcript'], transcript)
        
        row_dict = row.to_dict()
        row_dict.update({
            'adalat_transcript': transcript,
            'segment_count': seg_count,
            'total_speech_duration': total_dur,
            'avg_segment_duration': avg_dur,
            'inference_time': time.time() - start_t,
            'wer': w,
            'cer': c,
            'similarity': s,
            'keyword_recall': k_rec,
            'matched_keywords': k_match,
            'missed_keywords': k_miss
        })
        results_list.append(row_dict)
        
        # Checkpointing
        pd.DataFrame(results_list).to_csv("results/evaluation_results.csv", index=False)
        
    final_df = pd.DataFrame(results_list)
    
    if final_df.empty:
        logger.error("No successful evaluations were completed. Results dataframe is empty!")
        return

    # Safely select columns for the keyword results
    kw_cols = [c for c in ['clip_id', 'audio_bucket', 'keyword_recall', 'matched_keywords', 'missed_keywords'] if c in final_df.columns]
    final_df.to_csv("results/keyword_results.csv", columns=kw_cols, index=False)
    
    if 'audio_bucket' in final_df.columns:
        bucket_summary = final_df.groupby('audio_bucket')[['wer', 'cer', 'similarity', 'keyword_recall']].mean()
        bucket_summary.to_csv("results/bucket_summary.csv")
    
    # Console Print
    print("\n==============================================")
    print("                FINAL RESULTS                 ")
    print("==============================================")
    print(f"Dataset Total Files : {len(audio_df)}")
    print(f"Bucketed Total      : {len(master_df)}")
    print(f"Files Evaluated     : {len(final_df)}")
    
    if 'audio_bucket' in master_df.columns:
        print("\n=== BUCKET COUNTS ===")
        print(master_df['audio_bucket'].value_counts().to_string())
        
    print("\n=== SELECTED FILES ===")
    print_cols = [c for c in ['clip_id', 'audio_bucket'] if c in final_df.columns]
    print(final_df[print_cols].to_string(index=False))
    
    print("\n=== OVERALL AVERAGES ===")
    for metric in ['wer', 'cer', 'similarity', 'keyword_recall', 'inference_time']:
        if metric in final_df.columns:
            if metric == 'inference_time':
                print(f"Average {metric.capitalize():<14} : {final_df[metric].mean():.4f} seconds")
            else:
                print(f"Average {metric.capitalize():<14} : {final_df[metric].mean():.4f}")
            
    print("\nCheck the 'results/' folder for detailed CSV outputs.")

if __name__ == "__main__":
    main()
