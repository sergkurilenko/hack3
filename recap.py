import os
import glob
import json
import time
import threading
import srt
import pandas as pd
import re
from summarizer.sbert import SBertSummarizer
from moviepy.editor import VideoFileClip, concatenate_videoclips

RECAP_CONFIG_PATH = "recap_config.json"
RECAP_LOG_PATH = "OUT/recap.log"
RECAP_PROGRESS_PATH = "OUT/progress.json"

def read_config():
    default = {
        "genre": "",
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

def write_progress_json(progress, status="running", msg=""):
    with open(RECAP_PROGRESS_PATH, "w", encoding="utf-8") as pf:
        json.dump({"progress": progress, "status": status, "msg": msg, "ts": time.time()}, pf)

def parse_srt_to_sentences(srt_path, video_filename, file_idx):
    with open(srt_path, "r", encoding="utf-8") as f:
        subs = list(srt.parse(f))
    parsed_subs = [(s.start.total_seconds(), s.end.total_seconds(), s.content.strip()) for s in subs if s.content.strip()]
    cleaned_subs = [(start, end, re.sub(r"\s*\n\s*", " ", text)) for start, end, text in parsed_subs]
    merged_sentences = []
    buffer_text = ""
    buffer_start = None
    buffer_end = None

    for start, end, text in cleaned_subs:
        if not buffer_text:
            buffer_start = start
        if buffer_text:
            buffer_text += " " + text
        else:
            buffer_text = text
        buffer_end = end
        if re.search(r"[.?!…]$", text.strip()):
            merged_sentences.append((buffer_start, buffer_end, buffer_text.strip(), video_filename, file_idx))
            buffer_text = ""
            buffer_start = None
            buffer_end = None

    if buffer_text:
        merged_sentences.append((buffer_start, buffer_end, buffer_text.strip(), video_filename, file_idx))
    df = pd.DataFrame(merged_sentences, columns=["start", "end", "text", "file", "file_idx"])
    return df

def process_all_srt(video_files, subtitle_files):
    all_sentences = []
    for idx, (video_file, srt_file) in enumerate(zip(video_files, subtitle_files)):
        if not os.path.exists(srt_file):
            continue
        df = parse_srt_to_sentences(srt_file, os.path.basename(video_file), idx + 1)
        all_sentences.append(df)
    if not all_sentences:
        return pd.DataFrame(columns=["start", "end", "text", "file", "file_idx"])
    return pd.concat(all_sentences, ignore_index=True)

def pick_main_segments(df, model_name, max_duration=120):
    model = SBertSummarizer(model_name)
    texts = df["text"].tolist()
    if not texts:
        return []
    num_sentences = min(20, len(texts))
    #print("\n".join(texts))
   # texts1 = texts[:10]
    selected_text = model("\n".join(texts), num_sentences=num_sentences, return_as_list=True)
    print(selected_text)
    #summary_phrases = [s.strip() for s in selected_text.split('\n') if s.strip()]
    summary_phrases = selected_text
    if not summary_phrases:
        picked_indices = list(range(min(num_sentences, len(df))))
    else:
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
        seg_duration = row["end"] - row["start"]
        if total + seg_duration > max_duration:
            break
        result.append(row)
        total += seg_duration
    return result

def run_recap_with_logger(log, progress_cb=None):
    IN_VIDEOS = "IN/videos"
    IN_SUBTITLES = "IN/subtitles"
    OUT_PATH = "OUT"
    os.makedirs(OUT_PATH, exist_ok=True)
    config = read_config()
    transition = float(config.get("transition_duration", 1.0))
    max_duration = int(config.get("max_recap_duration", 120))
    model_name = config.get("sbert_model", "paraphrase-MiniLM-L6-v2")

    write_progress_json(0.0, status="running", msg="Инициализация задачи")

    video_files = sorted(glob.glob(os.path.join(IN_VIDEOS, "*.mp4")))
    subtitle_files = [os.path.join(IN_SUBTITLES, os.path.splitext(os.path.basename(v))[0]+".srt") for v in video_files]
    video_sub_pairs = [(v, s) for v, s in zip(video_files, subtitle_files) if os.path.exists(s)]
    if not video_sub_pairs:
        msg = "Нет видео или субтитров для обработки!"
        log(msg, 1.0)
        write_progress_json(1.0, status="error", msg=msg)
        return

    video_files = [v for v, s in video_sub_pairs]
    subtitle_files = [s for v, s in video_sub_pairs]

    log("Предобработка субтитров...", 0.05)
    write_progress_json(0.05, status="running", msg="Предобработка субтитров...")
    df = process_all_srt(video_files, subtitle_files)

    log(f"Всего предложений: {len(df)}", 0.10)
    log("Выбор главных фрагментов через SBert...", 0.20)
    write_progress_json(0.20, status="running", msg="Выбор главных фрагментов через SBert...")
    picked = pick_main_segments(df, model_name, max_duration)
    if not picked:
        msg = "Не удалось выбрать ни одного фрагмента для рекапа!"
        log(msg, 1.0)
        write_progress_json(1.0, status="error", msg=msg)
        return

    log("Формируется текстовый рекап...", 0.25)
    recap_texts = []
    recap_scenes = []
    for row in picked:
        recap_texts.append(f"{row['file']} [UNKNOWN]: {row['text']}")
        recap_scenes.append({
            "file": row["file"], "file_idx": int(row["file_idx"]),
            "start": float(row["start"]), "end": float(row["end"]),
            "speaker": "UNKNOWN", "text": row["text"]
        })
    recap_txt = os.path.join(OUT_PATH, "recap.txt")
    with open(recap_txt, "w", encoding="utf-8") as f:
        f.write("\n".join(recap_texts))
    log(f"Сохранён текстовый рекап: {recap_txt}", 0.3)
    recap_json = os.path.join(OUT_PATH, "recap.json")
    with open(recap_json, "w", encoding="utf-8") as f:
        json.dump(recap_scenes, f, ensure_ascii=False, indent=2)
    log(f"Сохранён JSON с событиями: {recap_json}", 0.35)

    log("Генерируется видео-рекап...", 0.4)
    write_progress_json(0.4, status="running", msg="Генерируется видео-рекап...")

    video_clips = []
    for row in picked:
        vfile = os.path.join(IN_VIDEOS, row["file"])
        try:
            clip = VideoFileClip(vfile).subclip(float(row["start"]), float(row["end"]))
            video_clips.append(clip)
        except Exception as e:
            log(f"Ошибка при обработке {vfile}: {e}")

    if not video_clips:
        msg = "Нет сцен для генерации видеорекапа."
        log(msg, 1.0)
        write_progress_json(1.0, status="error", msg=msg)
        return

    clips_with_transition = []
    for i, clip in enumerate(video_clips):
        clip = clip.fadein(transition) if i > 0 else clip
        clip = clip.fadeout(transition) if i < len(video_clips) - 1 else clip
        clips_with_transition.append(clip)
    final_clip = concatenate_videoclips(clips_with_transition, method="compose")
    recap_mp4 = os.path.join(OUT_PATH, "recap.mp4")
    log("Запущен процесс рендера итогового видео...", 0.55)
    write_progress_json(0.55, status="running", msg="Запущен процесс рендера итогового видео...")

    try:
        duration = final_clip.duration or 60
        finished = [False]
        def monitor_progress():
            start_time = time.time()
            while not finished[0]:
                elapsed = time.time() - start_time
                ratio = min(1.0, elapsed / (duration * 1.2))
                prog = 0.55 + (1.0 - 0.55) * ratio
                msg = f"Рендер видео... ({int(prog*100)}%)"
                write_progress_json(prog, status="running", msg=msg)
                time.sleep(1)
        t = threading.Thread(target=monitor_progress, daemon=True)
        t.start()
        final_clip.write_videofile(recap_mp4, codec="libx264", audio_codec="aac")
        finished[0] = True
        t.join(0.5)
        log(f"Сохранён видеорекап: {recap_mp4}", 1.0)
        write_progress_json(1.0, status="done", msg="Генерация полностью завершена!")
    except Exception as e:
        msg = f"Ошибка сохранения видео: {e}"
        log(msg, 1.0)
        write_progress_json(1.0, status="error", msg=msg)
        raise
    finally:
        final_clip.close()
        for clip in video_clips:
            clip.close()
