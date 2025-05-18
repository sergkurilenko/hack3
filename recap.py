import os
import glob
import json
import time
import threading
import pysrt
from moviepy.editor import VideoFileClip, concatenate_videoclips

RECAP_CONFIG_PATH = "recap_config.json"
RECAP_LOG_PATH = "OUT/recap.log"
RECAP_PROGRESS_PATH = "OUT/progress.json"

def read_config():
    if not os.path.exists(RECAP_CONFIG_PATH):
        return {"genre": "", "transition_duration": 1.0}
    with open(RECAP_CONFIG_PATH, "r", encoding="utf-8") as f:
        return json.load(f)

def parse_id(filename):
    base = os.path.splitext(os.path.basename(filename))[0]
    compilation_id, season_id, episode_id = base.split('_')
    return compilation_id, season_id, episode_id

def get_first_last_subtitle_times(srt_path):
    subs = pysrt.open(srt_path, encoding='utf-8')
    if len(subs) == 0:
        return None, None, None, None
    first = subs[0]
    last = subs[-1]
    start_first = first.start.hours*3600 + first.start.minutes*60 + first.start.seconds + first.start.milliseconds/1000
    end_first = first.end.hours*3600 + first.end.minutes*60 + first.end.seconds + first.end.milliseconds/1000
    start_last = last.start.hours*3600 + last.start.minutes*60 + last.start.seconds + last.start.milliseconds/1000
    end_last = last.end.hours*3600 + last.end.minutes*60 + last.end.seconds + last.end.milliseconds/1000
    return (start_first, end_first, first.text), (start_last, end_last, last.text)

def write_progress_json(progress, status="running", msg=""):
    with open(RECAP_PROGRESS_PATH, "w", encoding="utf-8") as pf:
        json.dump({"progress": progress, "status": status, "msg": msg, "ts": time.time()}, pf)

def run_recap_with_logger(log, progress_cb=None):
    IN_VIDEOS = "IN/videos"
    IN_SUBTITLES = "IN/subtitles"
    OUT_PATH = "OUT"
    os.makedirs(OUT_PATH, exist_ok=True)
    config = read_config()
    transition = float(config.get("transition_duration", 1.0))

    # Сброс прогресса в начале
    write_progress_json(0.0, status="running", msg="Инициализация задачи")

    video_files = sorted(glob.glob(os.path.join(IN_VIDEOS, "*.mp4")))
    total = len(video_files)
    recap_texts = []
    recap_scenes = []
    video_clips = []

    log("Запуск задачи", 0.0)
    write_progress_json(0.0, status="running", msg="Запуск задачи")
    log(f"Найдено {total} видео для обработки")

    if total == 0:
        msg = "Нет видеофайлов для обработки."
        log(msg, 1.0)
        write_progress_json(1.0, status="error", msg=msg)
        return

    log("Формируется текстовый рекап и JSON событий...", 0.1)
    write_progress_json(0.1, status="running", msg="Формируется текстовый рекап и JSON событий...")

    # === ОБРАБОТКА ФАЙЛОВ ===
    for i, video_path in enumerate(video_files):
        base_name = os.path.splitext(os.path.basename(video_path))[0]
        srt_path = os.path.join(IN_SUBTITLES, f"{base_name}.srt")
        step_progress = 0.1 + 0.4 * ((i+1)/total) if total else 0.5
        if not os.path.exists(srt_path):
            msg = f"[{i+1}/{total}] Субтитры не найдены для {base_name}, пропуск"
            log(msg, step_progress)
            write_progress_json(step_progress, status="running", msg=msg)
            continue
        compilation_id, season_id, episode_id = parse_id(video_path)
        first, last = get_first_last_subtitle_times(srt_path)
        if first is None or last is None:
            msg = f"[{i+1}/{total}] Нет субтитров в файле {srt_path}, пропуск"
            log(msg, step_progress)
            write_progress_json(step_progress, status="running", msg=msg)
            continue
        try:
            clip = VideoFileClip(video_path)
        except Exception as e:
            msg = f"[{i+1}/{total}] Не удалось открыть видео {video_path}: {e}"
            log(msg, step_progress)
            write_progress_json(step_progress, status="running", msg=msg)
            continue
        try:
            first_clip = clip.subclip(first[0], first[1])
            last_clip = clip.subclip(last[0], last[1])
            video_clips.extend([first_clip, last_clip])
            msg = f"[{i+1}/{total}] Сцены из {base_name} добавлены к видеорекапу."
            log(msg, step_progress)
        except Exception as e:
            msg = f"[{i+1}/{total}] Ошибка выделения сцен: {e}"
            log(msg, step_progress)
            write_progress_json(step_progress, status="running", msg=msg)
            continue
        recap_texts.append(f"{base_name}:\n{first[2]}\n...\n{last[2]}\n")
        recap_scenes.append({
            "compilation_id": compilation_id,
            "season_id": season_id,
            "episode_id": episode_id,
            "first": {"text": first[2], "start": first[0], "end": first[1]},
            "last": {"text": last[2], "start": last[0], "end": last[1]}
        })

    log("Текстовый рекап и JSON событий формируются...", 0.5)
    write_progress_json(0.5, status="running", msg="Текстовый рекап и JSON событий формируются...")

    recap_txt = os.path.join(OUT_PATH, "recap.txt")
    with open(recap_txt, "w", encoding="utf-8") as f:
        f.write("\n".join(recap_texts))
    log(f"Сохранён текстовый рекап: {recap_txt}")
    recap_json = os.path.join(OUT_PATH, "recap.json")
    with open(recap_json, "w", encoding="utf-8") as f:
        json.dump(recap_scenes, f, ensure_ascii=False, indent=2)
    log(f"Сохранён JSON с событиями: {recap_json}")

    log("Генерируется видео-рекап...", 0.5)
    write_progress_json(0.5, status="running", msg="Генерируется видео-рекап...")

    # === ГЕНЕРАЦИЯ ВИДЕО ===
    if video_clips:
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
    else:
        msg = "Нет сцен для генерации видеорекапа."
        log(msg, 1.0)
        write_progress_json(1.0, status="error", msg=msg)
