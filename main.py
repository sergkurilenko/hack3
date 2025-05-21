from fastapi import FastAPI, Request, Form, UploadFile, File
from fastapi.responses import HTMLResponse, FileResponse, RedirectResponse, JSONResponse
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates
import os, shutil, json, threading
from recap import run_recap_with_logger, RECAP_CONFIG_PATH, RECAP_LOG_PATH, RECAP_PROGRESS_PATH

app = FastAPI()
app.mount("/IN", StaticFiles(directory="IN"), name="in")
app.mount("/OUT", StaticFiles(directory="OUT"), name="out")
templates = Jinja2Templates(directory="templates")

@app.get("/", response_class=HTMLResponse)
async def index(request: Request):
    in_videos = sorted(os.listdir("IN/videos")) if os.path.exists("IN/videos") else []
    in_subs = sorted(os.listdir("IN/subtitles")) if os.path.exists("IN/subtitles") else []
    out_files = sorted(os.listdir("OUT")) if os.path.exists("OUT") else []
    config = {}
    if os.path.exists(RECAP_CONFIG_PATH):
        with open(RECAP_CONFIG_PATH, encoding="utf-8") as f:
            config = json.load(f)
    return templates.TemplateResponse("index.html", {"request": request, "in_videos": in_videos, "in_subs": in_subs, "out_files": out_files, "config": config})

@app.post("/run_recap")
async def run_recap_endpoint(request: Request):
    data = await request.json()
    selected = data.get("files", [])

    if os.path.exists(RECAP_LOG_PATH):
        os.remove(RECAP_LOG_PATH)
    if os.path.exists(RECAP_PROGRESS_PATH):
        os.remove(RECAP_PROGRESS_PATH)

    def logger(msg, progress=None):
        print(msg)
        with open(RECAP_LOG_PATH, "a", encoding="utf-8") as f:
            f.write(msg + "\n")

    threading.Thread(target=lambda: run_recap_with_logger(logger, selected_files=selected)).start()
    return JSONResponse({"ok": True})

@app.get("/recap_progress")
async def get_progress():
    if os.path.exists(RECAP_PROGRESS_PATH):
        with open(RECAP_PROGRESS_PATH, encoding="utf-8") as f:
            return JSONResponse(json.load(f))
    return JSONResponse({"progress": 0.0, "status": "idle", "msg": "Ожидание"})

@app.post("/set_config")
async def set_config(
    request: Request,
    transition_duration: float = Form(...),
    max_recap_duration: int = Form(...),
    sbert_model: str = Form("paraphrase-MiniLM-L6-v2")
):
    with open(RECAP_CONFIG_PATH, "w", encoding="utf-8") as f:
        json.dump({
            "transition_duration": transition_duration,
            "max_recap_duration": max_recap_duration,
            "sbert_model": sbert_model
        }, f)
    return RedirectResponse("/", status_code=303)

@app.post("/upload_file")
async def upload_file(request: Request, video_file: UploadFile = File(None), subtitle_file: UploadFile = File(None)):
    if video_file:
        with open(os.path.join("IN/videos", video_file.filename), "wb") as f:
            shutil.copyfileobj(video_file.file, f)
    if subtitle_file:
        with open(os.path.join("IN/subtitles", subtitle_file.filename), "wb") as f:
            shutil.copyfileobj(subtitle_file.file, f)
    return RedirectResponse("/", status_code=303)

@app.post("/clear_out")
async def clear_out():
    for f in os.listdir("OUT"):
        os.remove(os.path.join("OUT", f))
    return JSONResponse({"ok": True})

