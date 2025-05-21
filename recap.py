import os
import glob
import json
import time
import threading
import srt
import pandas as pd
import re
import tempfile
from moviepy.editor import VideoFileClip, concatenate_videoclips, AudioFileClip
from summarizer.sbert import SBertSummarizer
from simple_diarizer.diarizer import Diarizer

RECAP_CONFIG_PATH = "recap_config.json"
RECAP_LOG_PATH = "OUT/recap.log"
RECAP_PROGRESS_PATH = "OUT/progress.json"

# Инициализация simple_diarizer
try:
    diarizer = Diarizer()
except Exception as e:
    print(f"[Diarizer init error] {e}")
    diarizer = None

def extract_audio_from_video(video_path: str) -> str:
    tmp_wav = tempfile.NamedTemporaryFile(delete=False, suffix=".wav").name
    try:
        clip = VideoFileClip(video_path)
        clip.audio.write_audiofile(tmp_wav, fps=16000, nbytes=2, codec='pcm_s16le', verbose=False, logger=None)
        clip.close()
    except Exception as e:
        print(f"[moviepy error] failed to extract audio from {video_path}: {e}")
        raise
    return tmp_wav

def diarize_audio(audio_path: str):
    diarization_result = diarizer.diarize(audio_path)
    return [(float(s['start']), float(s['end']), f"SPEAKER_{s['label']}") for s in diarization_result]

def find_speaker(start, end, diar_segments):
    overlaps = {}
    for s_start, s_end, speaker in diar_segments:
        overlap = max(0, min(end, s_end) - max(start, s_start))
        if overlap > 0:
            overlaps[speaker] = overlaps.get(speaker, 0) + overlap
    return max(overlaps, key=overlaps.get) if overlaps else "UNKNOWN"

def write_progress_json(progress, status="running", msg=""):
    with open(RECAP_PROGRESS_PATH, "w", encoding="utf-8") as pf:
        json.dump({"progress": progress, "status": status, "msg": msg, "ts": time.time()}, pf)

def parse_srt_by_diar(srt_path, diar_segments, video_name, file_idx):
    with open(srt_path, "r", encoding="utf-8") as f:
        subs = list(srt.parse(f))
    parsed = [(s.start.total_seconds(), s.end.total_seconds(), s.content.strip()) for s in subs if s.content.strip()]
    cleaned = [(s, e, re.sub(r"\s*\n\s*", " ", t)) for s, e, t in parsed]

    merged_subs = []
    buffer_text = ""
    buffer_start = None
    buffer_end = None
    for start, end, text in cleaned:
        if not buffer_text:
            buffer_start = start
        buffer_text += " " + text if buffer_text else text
        buffer_end = end
        if re.search(r"[.?!…]$", text.strip()):
            merged_subs.append((buffer_start, buffer_end, buffer_text.strip()))
            buffer_text = ""
            buffer_start = None
    if buffer_text:
        merged_subs.append((buffer_start, buffer_end, buffer_text.strip()))

    annotated = [(s, e, find_speaker(s, e, diar_segments), t) for s, e, t in merged_subs]
    blocks = []
    cur_spk = None
    buffer_text = ""
    buffer_start = None
    buffer_end = None
    for start, end, spk, text in annotated:
        if spk == cur_spk:
            buffer_text += " " + text
            buffer_end = end
        else:
            if cur_spk is not None:
                blocks.append((buffer_start, buffer_end, cur_spk, buffer_text.strip(), video_name, file_idx))
            cur_spk = spk
            buffer_text = text
            buffer_start = start
            buffer_end = end
    if cur_spk is not None:
        blocks.append((buffer_start, buffer_end, cur_spk, buffer_text.strip(), video_name, file_idx))

    return pd.DataFrame(blocks, columns=["start", "end", "speaker", "text", "file", "file_idx"])

def read_config():
    default = {
        "transition_duration": 1.0,
        "max_recap_duration": 120,
        "sbert_model": "paraphrase-MiniLM-L6-v2"
    }
    if not os.path.exists(RECAP_CONFIG_PATH):
        return default
    try:
        with open(RECAP_CONFIG_PATH, "r", encoding="utf-8") as f:
            cfg = json.load(f)
        for k, v in default.items():
            if k not in cfg:
                cfg[k] = v
        return cfg
    except Exception:
        return default

def pick_main_segments(df, model_name, max_duration=120):
    model = SBertSummarizer(model_name)
    texts = df["text"].tolist()
    if not texts:
        return []
    num_sentences = min(20, len(texts))
    summary_phrases = model(" ".join(texts), num_sentences=num_sentences, return_as_list=True)

    picked_indices = []
    for phrase in summary_phrases:
        for idx, row in df.iterrows():
            if phrase in row["text"] and idx not in picked_indices:
                picked_indices.append(idx)
                break

    result = []
    total = 0
    for idx in picked_indices:
        row = df.iloc[idx]
        dur = row["end"] - row["start"]
        if total + dur > max_duration:
            continue
        result.append(row)
        total += dur
    return result

def run_recap_with_logger(log, progress_cb=None, selected_files=None):
    IN_VIDEOS = "IN/videos"
    IN_SUBTITLES = "IN/subtitles"
    OUT_PATH = "OUT"
    os.makedirs(OUT_PATH, exist_ok=True)
    config = read_config()
    transition = float(config.get("transition_duration", 1.0))
    max_duration = int(config.get("max_recap_duration", 120))
    model_name = config.get("sbert_model", "paraphrase-MiniLM-L6-v2")

    log("Инициализация задачи", 0.0)
    write_progress_json(0.0, status="running", msg="Инициализация задачи")
    video_files = sorted(glob.glob(os.path.join(IN_VIDEOS, "*.mp4")))
    if selected_files:
        video_files = [f for f in video_files if os.path.basename(f) in selected_files]
    subtitle_files = [os.path.join(IN_SUBTITLES, os.path.splitext(os.path.basename(v))[0]+".srt") for v in video_files]
    video_sub_pairs = [(v, s) for v, s in zip(video_files, subtitle_files) if os.path.exists(s)]

    log(f"Найдено видеофайлов: {len(video_files)}", 0.01)
    log(f"Пары video+srt: {len(video_sub_pairs)}", 0.02)

    if not video_sub_pairs:
        msg = "Нет видео или субтитров для обработки!"
        log(msg, 1.0)
        write_progress_json(1.0, status="error", msg=msg)
        return

    write_progress_json(0.2, status="running", msg="Диаризация")
    all_dfs = []
    for idx, (vfile, sfile) in enumerate(video_sub_pairs):
        log(f"Диаризация аудио: {vfile}", 0.05 + idx * 0.02)
        audio_path = extract_audio_from_video(vfile)
        diar_segments = diarize_audio(audio_path)
        df = parse_srt_by_diar(sfile, diar_segments, os.path.basename(vfile), idx + 1)
        all_dfs.append(df)

    df = pd.concat(all_dfs, ignore_index=True)
    log(f"Всего реплик после объединения: {df.shape[0]}")
    write_progress_json(0.3, status="running", msg="Выбор ключевых фрагментов через SBert")
    picked = pick_main_segments(df, model_name, max_duration)
    if not picked:
        msg = "Не удалось выбрать ни одного фрагмента для рекапа!"
        log(msg, 1.0)
        write_progress_json(1.0, status="error", msg=msg)
        return

    recap_txt = os.path.join(OUT_PATH, "recap.txt")
    recap_json = os.path.join(OUT_PATH, "recap.json")
    recap_texts, recap_scenes = [], []
    for row in picked:
        recap_texts.append(f"{row['file']} [{row['speaker']}]: {row['text']}")
        recap_scenes.append({
            "file": row["file"], "file_idx": int(row["file_idx"]),
            "start": float(row["start"]), "end": float(row["end"]),
            "speaker": row["speaker"], "text": row["text"]
        })
    with open(recap_txt, "w", encoding="utf-8") as f:
        f.write("\n".join(recap_texts))
    with open(recap_json, "w", encoding="utf-8") as f:
        json.dump(recap_scenes, f, ensure_ascii=False, indent=2)
    log(f"Сохранён recap.txt и recap.json", 0.3)

    log("Сборка видеорекапа...", 0.4)
    clips = []
    for row in picked:
        vfile = os.path.join(IN_VIDEOS, row["file"])
        try:
            clip = VideoFileClip(vfile).subclip(float(row["start"]), float(row["end"]))
            clips.append(clip.fadein(transition).fadeout(transition))
        except Exception as e:
            log(f"Ошибка клипа {vfile}: {e}")
    if not clips:
        log("Нет клипов", 1.0)
        write_progress_json(1.0, status="error", msg="Нет клипов")
        return
    final = concatenate_videoclips(clips, method="compose")
    recap_mp4 = os.path.join(OUT_PATH, "recap.mp4")
    final.write_videofile(recap_mp4, codec="libx264", audio_codec="aac")
    log(f"Видеорекап сохранён в {recap_mp4}", 1.0)
    write_progress_json(1.0, status="done", msg="Готово")
    final.close()
    for c in clips:
        c.close()