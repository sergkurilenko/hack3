import os
import json
import threading
from fastapi import FastAPI, Request, Form, UploadFile, File
from fastapi.responses import RedirectResponse, JSONResponse
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates

from recap import run_recap_with_logger, RECAP_CONFIG_PATH, RECAP_LOG_PATH, RECAP_PROGRESS_PATH

app = FastAPI(debug=True)
app.mount("/OUT", StaticFiles(directory="OUT"), name="out")
app.mount("/IN", StaticFiles(directory="IN"), name="in")
templates = Jinja2Templates(directory="templates")

def list_files(path, exts=None):
    try:
        files = os.listdir(path)
    except FileNotFoundError:
        files = []
    if exts is not None:
        files = [f for f in files if f.lower().endswith(tuple(exts))]
    return sorted(files)

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

@app.get("/")
async def index(request: Request):
    in_videos = list_files("IN/videos", [".mp4"])
    in_subs = list_files("IN/subtitles", [".srt"])
    out_files = list_files("OUT")
    config = read_config()
    return templates.TemplateResponse("index.html", {
        "request": request,
        "in_videos": in_videos,
        "in_subs": in_subs,
        "out_files": out_files,
        "config": config,
    })

@app.post("/set_config")
async def set_config(
    request: Request,
    genre: str = Form(""),
    transition_duration: float = Form(1.0),
    max_recap_duration: int = Form(120),
    sbert_model: str = Form("paraphrase-MiniLM-L6-v2")
):
    with open(RECAP_CONFIG_PATH, "w", encoding="utf-8") as f:
        json.dump({
            "genre": genre,
            "transition_duration": transition_duration,
            "max_recap_duration": max_recap_duration,
            "sbert_model": sbert_model
        }, f)
    return RedirectResponse("/", status_code=303)

@app.post("/run_recap")
async def run_recap_endpoint():
    # Очищаем лог-файл и progress-файл
    if os.path.exists(RECAP_LOG_PATH):
        os.remove(RECAP_LOG_PATH)
    if os.path.exists(RECAP_PROGRESS_PATH):
        os.remove(RECAP_PROGRESS_PATH)

    def logger(msg, progress=None):
        print(msg)
        with open(RECAP_LOG_PATH, "a", encoding="utf-8") as f:
            f.write(msg + "\n")

    threading.Thread(target=lambda: run_recap_with_logger(logger)).start()
    return JSONResponse({"ok": True})

@app.get("/recap_progress")
async def get_progress():
    if os.path.exists(RECAP_PROGRESS_PATH):
        with open(RECAP_PROGRESS_PATH, "r", encoding="utf-8") as f:
            return JSONResponse(json.load(f))
    return JSONResponse({"progress": 0.0, "status": "not_started", "msg": ""})

@app.post("/upload_file")
async def upload_file(
    video_file: UploadFile = File(None),
    subtitle_file: UploadFile = File(None)
):
    if video_file:
        filename = os.path.basename(video_file.filename)
        dest_path = os.path.join("IN/videos", filename)
        with open(dest_path, "wb") as f:
            content = await video_file.read()
            f.write(content)
    if subtitle_file:
        filename = os.path.basename(subtitle_file.filename)
        dest_path = os.path.join("IN/subtitles", filename)
        with open(dest_path, "wb") as f:
            content = await subtitle_file.read()
            f.write(content)
    return RedirectResponse("/", status_code=303)

@app.post("/clear_out")
async def clear_out(request: Request):
    for fname in os.listdir("OUT"):
        try:
            os.remove(os.path.join("OUT", fname))
        except Exception:
            pass
    return RedirectResponse("/", status_code=303)
