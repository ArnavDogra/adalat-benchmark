import os
import sys
import time
import json
import glob
import logging
import zipfile
import re
import pandas as pd
import numpy as np
import librosa
import torch
import jiwer
from rapidfuzz import fuzz
from rapidfuzz.distance import Levenshtein
from transformers import pipeline
from tqdm import tqdm

# --- CONFIGURATION ---
AUDIO_ZIP = "audio.zip"
UNZIP_DIR = "audio_unzipped"
TRANSCRIPT_CSV = "sarvam.csv"
RESULTS_DIR = "results"
LOGS_DIR = "logs"

os.makedirs(RESULTS_DIR, exist_ok=True)
os.makedirs(LOGS_DIR, exist_ok=True)

# --- LOGGING SETUP ---
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    handlers=[
        logging.FileHandler(os.path.join(LOGS_DIR, "evaluation.log")),
        logging.StreamHandler(sys.stdout)
    ]
)
logger = logging.getLogger(__name__)

# --- DEVICE CONFIG ---
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
logger.info(f"Starting execution on device: {DEVICE}")

# --- HELPER FUNCTIONS ---
def unzip_audio():
    if not os.path.exists(UNZIP_DIR):
        zip_file_to_extract = None
        if os.path.exists("hindi_audio.zip"):
            zip_file_to_extract = "hindi_audio.zip"
        elif os.path.exists(AUDIO_ZIP):
            zip_file_to_extract = AUDIO_ZIP
            
        if zip_file_to_extract:
            logger.info(f"Unzipping {zip_file_to_extract} to {UNZIP_DIR}...")
            with zipfile.ZipFile(zip_file_to_extract, 'r') as zip_ref:
                zip_ref.extractall(UNZIP_DIR)
            logger.info("Unzip complete.")
        else:
            logger.warning(f"Could not find audio.zip or hindi_audio.zip, and {UNZIP_DIR} does not exist. Scanning current directory instead.")
    else:
        logger.info(f"Audio directory {UNZIP_DIR} already exists. Skipping unzip.")

def extract_keywords(text):
    # CRITICAL TERM RECALL LOGIC (Length > 3 heuristic)
    clean_text = re.sub(r'[^\w\s]', '', str(text))
    return set([w for w in clean_text.split() if len(w) > 3])

def compute_keyword_spotting(ref_text, hyp_text):
    # LEVENSHTEIN KEYWORD SPOTTING LOGIC (30% CER logic)
    ref_words = str(ref_text).split()
    hyp_words = str(hyp_text).split()
    if not ref_words:
        return 1.0, "", ""
    matched_keywords = []
    missed_keywords = []
    for rw in ref_words:
        if len(rw) == 0: continue
        is_detected = False
        for hw in hyp_words:
            if len(hw) == 0: continue
            distance = Levenshtein.distance(rw, hw)
            cer = distance / len(rw)
            if cer <= 0.30:
                is_detected = True
                break
        if is_detected: matched_keywords.append(rw)
        else: missed_keywords.append(rw)
    keyword_recall = len(matched_keywords) / len(ref_words)
    return keyword_recall, ", ".join(matched_keywords), ", ".join(missed_keywords)

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

# --- MAIN WORKFLOW ---
def main():
    import time
    pipeline_start_time = time.time()
    logger.info("=== STARTING ADALAT EVALUATION PIPELINE ===")
    
    unzip_audio()
    
    # 1. Load Data
    logger.info("Scanning for audio files...")
    audio_paths = glob.glob(f"{UNZIP_DIR}/**/*.wav", recursive=True) + glob.glob(f"{UNZIP_DIR}/**/*.mp3", recursive=True)
    if not audio_paths:
        # Fallback to current directory
        audio_paths = glob.glob("**/*.wav", recursive=True) + glob.glob("**/*.mp3", recursive=True)
    
    audio_df = pd.DataFrame({'full_path': audio_paths})
    if audio_df.empty:
        logger.error("No audio files found. Exiting.")
        return
        
    audio_df['clip_id'] = audio_df['full_path'].apply(lambda x: os.path.splitext(os.path.basename(x))[0])
    
    if not os.path.exists(TRANSCRIPT_CSV):
        logger.error(f"Transcript file {TRANSCRIPT_CSV} not found. Exiting.")
        return
        
    sarvam_df = pd.read_csv(TRANSCRIPT_CSV)
    if 'clip_id' not in sarvam_df.columns and 'audio' in sarvam_df.columns:
        sarvam_df['clip_id'] = sarvam_df['audio'].astype(str).apply(lambda x: os.path.splitext(os.path.basename(x))[0])
        
    master_df = pd.merge(audio_df, sarvam_df[['clip_id', 'sarvam_transcript']], on='clip_id', how='inner')
    logger.info(f"Matched {len(master_df)} files between audio directory and CSV.")
    if master_df.empty: return

    # 2. Profiling & Bucketing
    logger.info("Profiling audio files...")
    tqdm.pandas(desc="Profiling Audio")
    cols = ['duration', 'sample_rate', 'rms_energy', 'silence_ratio', 'dynamic_range', 'signal_variance', 'clipping_ratio']
    master_df[cols] = master_df['full_path'].progress_apply(profile_audio)
    
    master_df = master_df[(master_df['duration'] >= 15) & (master_df['duration'] <= 300)]
    master_df['word_count'] = master_df['sarvam_transcript'].apply(lambda x: len(str(x).split()))
    master_df['speaking_rate'] = master_df['word_count'] / (master_df['duration'] + 1e-9)
    
    # Handle NaN values for scaler
    master_df = master_df.dropna(subset=['rms_energy', 'signal_variance', 'dynamic_range', 'silence_ratio', 'clipping_ratio', 'speaking_rate'])
    
    from sklearn.preprocessing import MinMaxScaler
    scaler = MinMaxScaler()
    norm_rms = scaler.fit_transform(master_df[['rms_energy']])
    norm_var = scaler.fit_transform(master_df[['signal_variance']])
    norm_dr = scaler.fit_transform(master_df[['dynamic_range']])
    norm_silence = 1 - scaler.fit_transform(master_df[['silence_ratio']])
    norm_clipping = 1 - scaler.fit_transform(master_df[['clipping_ratio']])
    norm_clarity = scaler.fit_transform(master_df[['speaking_rate']])
    
    score = (0.15 * norm_rms + 0.10 * norm_var + 0.10 * norm_dr + 0.15 * norm_silence + 0.20 * norm_clipping + 0.30 * norm_clarity)
    master_df['quality_score'] = score
    
    q33, q66 = master_df['quality_score'].quantile([0.33, 0.66])
    def assign_bucket(s): return 'GOOD' if s >= q66 else ('MODERATE' if s >= q33 else 'BAD')
    master_df['audio_bucket'] = master_df['quality_score'].apply(assign_bucket)
    
    # 3. Sampling
    logger.info("Sampling 10 files per bucket (duration <= 90s)...")
    filtered_df = master_df[master_df['duration'] <= 90]
    samples = []
    for bucket, group in filtered_df.groupby('audio_bucket'):
        samples.append(group.sample(n=min(10, len(group)), random_state=42))
    evaluation_sample = pd.concat(samples, ignore_index=True) if samples else pd.DataFrame(columns=master_df.columns)
    bucket_csv_path = os.path.join(RESULTS_DIR, "bucket_assignments.csv")
    evaluation_sample.to_csv(bucket_csv_path, index=False)
    logger.info(f"Selected {len(evaluation_sample)} files. Saved to {bucket_csv_path}")
    
    # 4. Checkpointing setup
    transcriptions_csv_path = os.path.join(RESULTS_DIR, "transcriptions.csv")
    transcribed_data = {}
    if os.path.exists(transcriptions_csv_path):
        logger.info(f"Found existing {transcriptions_csv_path}. Loading checkpoints...")
        df_existing = pd.read_csv(transcriptions_csv_path)
        for _, r in df_existing.iterrows():
            transcribed_data[r['clip_id']] = r.to_dict()
    
    # 5. Model Loading (GPU Optimized)
    logger.info("Loading Adalat Whisper Model & Silero VAD (FP16)...")
    try:
        torch_dtype = torch.float16 if DEVICE == "cuda" else torch.float32
        adalat_pipe = pipeline('automatic-speech-recognition', model='adalat-ai/whisper-medium-hi-rmft', 
                               device=0 if DEVICE=='cuda' else -1, torch_dtype=torch_dtype, chunk_length_s=30)
        
        vad_model, utils = torch.hub.load(repo_or_dir='snakers4/silero-vad', model='silero_vad', force_reload=False, onnx=False)
        (get_speech_timestamps, save_audio, read_audio, VADIterator, collect_chunks) = utils
        if DEVICE == "cuda":
            vad_model = vad_model.to(DEVICE)
    except Exception as e:
        logger.error(f"Failed to load models: {e}")
        return

    # 6. Transcription & VAD Loop
    logger.info("Starting transcription loop...")
    results_list = []
                     
    for _, row in tqdm(evaluation_sample.iterrows(), total=len(evaluation_sample), desc="Transcribing"):
        clip_id = row['clip_id']
        if clip_id in transcribed_data:
            results_list.append(transcribed_data[clip_id])
            continue
            
        try:
            inf_start_time = time.time()
            path = row['full_path']
            wav = read_audio(path, sampling_rate=16000)
            
            # Send to VAD
            wav_vad = wav.to(DEVICE) if DEVICE == "cuda" else wav
            speech_timestamps = get_speech_timestamps(wav_vad, vad_model, sampling_rate=16000)
            
            seg_count = len(speech_timestamps) if speech_timestamps else 1
            chunks = [wav[s['start']:s['end']].numpy() for s in speech_timestamps] if speech_timestamps else [wav.numpy()]
            total_speech_dur = sum([len(c)/16000 for c in chunks])
            avg_seg_dur = total_speech_dur / seg_count if seg_count > 0 else 0
            
            # Transcription
            text_pieces = [adalat_pipe(c).get('text', '') for c in chunks]
            transcript = ' '.join(text_pieces).strip()
            
            inf_end_time = time.time()
            inference_time = inf_end_time - inf_start_time
            rtf = inference_time / row['duration'] if row['duration'] > 0 else 0
            
            res_dict = row.to_dict()
            res_dict.update({
                'adalat_transcript': transcript,
                'inference_time': inference_time,
                'segment_count': seg_count,
                'avg_segment_duration': avg_seg_dur,
                'total_speech_duration': total_speech_dur,
                'rtf': rtf
            })
            results_list.append(res_dict)
            
            # Save Checkpoint
            pd.DataFrame(results_list).to_csv(transcriptions_csv_path, index=False)
            
        except Exception as e:
            logger.error(f"Failed transcription for {clip_id}: {e}")
            res_dict = row.to_dict()
            res_dict.update({'adalat_transcript': '', 'inference_time': 0, 'segment_count': 0, 'avg_segment_duration': 0, 'total_speech_duration': 0, 'rtf': 0})
            results_list.append(res_dict)

    df_results = pd.DataFrame(results_list)
    
    # 7. Metrics Evaluation
    logger.info("Computing metrics...")
    wer_transform = jiwer.Compose([jiwer.ToLowerCase(), jiwer.RemoveMultipleSpaces(), jiwer.RemovePunctuation(), jiwer.Strip()])
    
    def compute_metrics(row):
        ref = str(row.get('sarvam_transcript', ''))
        hyp = str(row.get('adalat_transcript', ''))
        if not ref.strip(): return pd.Series([np.nan, np.nan, 0.0])
        try:
            ref_clean = wer_transform(ref)
            hyp_clean = wer_transform(hyp)
            wer = jiwer.wer(ref_clean, hyp_clean)
            cer = jiwer.cer(ref_clean, hyp_clean)
        except:
            wer = jiwer.wer(ref, hyp)
            cer = jiwer.cer(ref, hyp)
        sim = fuzz.ratio(ref, hyp)
        return pd.Series([wer, cer, sim])
        
    df_results[['wer', 'cer', 'similarity']] = df_results.apply(compute_metrics, axis=1)
    
    # Critical Term Recall (Heuristic)
    def analyze_critical_terms(row):
        ref_kw = extract_keywords(row.get('sarvam_transcript', ''))
        hyp_kw = extract_keywords(row.get('adalat_transcript', ''))
        if len(ref_kw) == 0: return pd.Series([1.0, 1.0, "", ""])
        matched = ref_kw.intersection(hyp_kw)
        missing = ref_kw - hyp_kw
        extra = hyp_kw - ref_kw
        recall = len(matched) / len(ref_kw) if len(ref_kw) > 0 else 1.0
        precision = len(matched) / len(hyp_kw) if len(hyp_kw) > 0 else 1.0
        return pd.Series([recall, precision, ", ".join(missing), ", ".join(extra)])
        
    df_results[['critical_recall', 'critical_precision', 'missing_critical', 'extra_critical']] = df_results.apply(analyze_critical_terms, axis=1)
    
    # Keyword Spotting (Levenshtein)
    def run_keyword_spotting(row):
        rec, matched, missed = compute_keyword_spotting(row.get('sarvam_transcript', ''), row.get('adalat_transcript', ''))
        return pd.Series([rec, matched, missed])
    df_results[['keyword_recall', 'matched_keywords', 'missed_keywords']] = df_results.apply(run_keyword_spotting, axis=1)

    eval_csv_path = os.path.join(RESULTS_DIR, "evaluation_results.csv")
    df_results.to_csv(eval_csv_path, index=False)
    
    crit_csv_path = os.path.join(RESULTS_DIR, "critical_term_analysis.csv")
    df_results[['clip_id', 'audio_bucket', 'sarvam_transcript', 'adalat_transcript', 'critical_recall', 'keyword_recall', 'missing_critical', 'missed_keywords']].to_csv(crit_csv_path, index=False)
    
    # 8. Summarization and Reporting
    avg_wer = df_results['wer'].mean()
    avg_cer = df_results['cer'].mean()
    avg_sim = df_results['similarity'].mean()
    avg_crit_recall = df_results['critical_recall'].mean()
    avg_kw_recall = df_results['keyword_recall'].mean()
    
    bucket_summary = df_results.groupby('audio_bucket').agg(
        Count=('clip_id', 'count'),
        Mean_WER=('wer', 'mean'),
        Mean_CER=('cer', 'mean'),
        Mean_Sim=('similarity', 'mean'),
        Mean_Crit_Recall=('critical_recall', 'mean'),
        Mean_KW_Recall=('keyword_recall', 'mean')
    )
    
    best_file_idx = df_results['wer'].idxmin()
    best_file = df_results.loc[best_file_idx] if not pd.isna(best_file_idx) else None
    
    worst_file_idx = df_results['wer'].idxmax()
    worst_file = df_results.loc[worst_file_idx] if not pd.isna(worst_file_idx) else None
    
    all_missing = []
    for m in df_results['missing_critical']:
        if pd.notna(m) and str(m).strip():
            all_missing.extend([w.strip() for w in str(m).split(',')])
    top_missing = pd.Series(all_missing).value_counts().head(10).to_dict()
    
    # Timing Stats
    avg_runtime = df_results['inference_time'].mean()
    avg_rtf = df_results['rtf'].mean()
    
    pipeline_end_time = time.time()
    total_pipeline_time = pipeline_end_time - pipeline_start_time
    total_files_processed = len(df_results)
    avg_end_to_end_time = total_pipeline_time / total_files_processed if total_files_processed > 0 else 0
    
    est_100 = avg_runtime * 100 / 60
    est_500 = avg_runtime * 500 / 60
    est_1000 = avg_runtime * 1000 / 60

    report = f"""
=========================================
      EVALUATION SUMMARY REPORT
=========================================

--- GLOBAL METRICS ---
Average WER: {avg_wer:.4f}
Average CER: {avg_cer:.4f}
Average Similarity: {avg_sim:.2f}%
Average Critical Term Recall: {avg_crit_recall*100:.2f}%
Average Keyword Recall (Levenshtein): {avg_kw_recall*100:.2f}%

--- BUCKET-WISE METRICS ---
{bucket_summary.to_string()}

--- EXTREMES ---
Best File (WER): {best_file['clip_id'] if best_file is not None else 'N/A'} (WER: {best_file['wer']:.4f} if best_file is not None else 'N/A')
Worst File (WER): {worst_file['clip_id'] if worst_file is not None else 'N/A'} (WER: {worst_file['wer']:.4f} if worst_file is not None else 'N/A')

--- TOP 10 MISSED CRITICAL TERMS ---
{json.dumps(top_missing, indent=2, ensure_ascii=False)}

--- TIMING & PROFILING (GPU) ---
Average Transcription Runtime per file: {avg_runtime:.2f} seconds
Average Real Time Factor (RTF): {avg_rtf:.4f}  (Time taken / Audio Duration)
(Note: RTF of 0.10 means 1 minute of audio takes 6 seconds to process)

Total Pipeline Execution Time: {total_pipeline_time:.2f} seconds
Average End-to-End Time per file: {avg_end_to_end_time:.2f} seconds (includes unzipping, profiling & prep)

--- ESTIMATED PROCESSING TIMES ---
For 100 files: ~{est_100:.2f} minutes
For 500 files: ~{est_500:.2f} minutes
For 1000 files: ~{est_1000:.2f} minutes

=========================================
"""
    report_path = os.path.join(RESULTS_DIR, "summary_report.txt")
    with open(report_path, "w", encoding="utf-8") as f:
        f.write(report)
        
    logger.info("Evaluation Complete. Results saved in 'results/' and 'logs/'.")
    logger.info(f"Summary:\n{report}")

if __name__ == "__main__":
    main()
