"""FastAPI adapter for the existing AI Smart Blackboard engines."""
from __future__ import annotations
import base64, io, os, tempfile, uuid
from datetime import datetime, timezone
from pathlib import Path
from typing import Any
from fastapi import FastAPI, HTTPException
from fastapi.responses import FileResponse, HTMLResponse, Response
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel, Field
from app.ai.provider import AIProviderError
from app.services.gemini_service import GeminiService
from app.modeling.graph_engine import GraphEngine
from app.modeling.model_pipeline import ModelPipeline
from app.recognition.math_engine import MathEngine

ROOT = Path(__file__).resolve().parent
STATIC = ROOT / "static"
MODELS = Path("app/data/models").resolve()
SESSIONS: dict[str, dict[str, Any]] = {}
app = FastAPI(title="AI Smart Blackboard API", description="Web adapter for existing Smart Blackboard engines", version="1.1.0")
app.mount("/static", StaticFiles(directory=STATIC), name="static")

class SelectionRequest(BaseModel):
    image: str = Field(min_length=20, max_length=20_000_000)
    subject: str = "Mathematics"
    object_id: str | None = None
class PlotRequest(BaseModel): expression: str = Field(min_length=1, max_length=500)
class ExplainRequest(BaseModel):
    subject: str = "General"
    content: str = Field(default="", max_length=5000)
    image: str | None = Field(default=None, max_length=20_000_000)
class AIAnalyzeRequest(BaseModel):
    subject: str = "General"
    image: str = Field(min_length=20, max_length=20_000_000)
    local_result: dict[str, Any] | None = None
class AILectureRequest(BaseModel):
    subject: str = "General"
    content: str = Field(min_length=1, max_length=20_000)
    task: str = Field(default="summarize", max_length=100)
class LectureNotesRequest(BaseModel):
    content: str = Field(min_length=1, max_length=20_000)
    task: str = Field(default="summarize", max_length=100)
class LectureStartRequest(BaseModel): subject: str
class LectureCaptureRequest(BaseModel):
    image: str = Field(min_length=20, max_length=20_000_000)
    objects: list[dict[str, Any]] = Field(default_factory=list)
    note: str = Field(default="", max_length=1000)
class LectureSlide(BaseModel):
    position: int = Field(ge=1)
    title: str = Field(default="", max_length=200)
    image: str = Field(min_length=20, max_length=20_000_000)
    objects: list[dict[str, Any]] = Field(default_factory=list)
class LectureSlidesRequest(BaseModel):
    slides: list[LectureSlide] = Field(min_length=1, max_length=100)

def _selection_file(data_url: str) -> Path:
    try:
        header, payload = data_url.split(",", 1)
        if not header.startswith("data:image/"): raise ValueError()
        raw = base64.b64decode(payload, validate=True)
    except Exception as exc: raise HTTPException(422, "A valid canvas image is required.") from exc
    if not raw or len(raw) > 12_000_000: raise HTTPException(413, "Canvas selection is empty or too large.")
    handle = tempfile.NamedTemporaryFile(prefix="blackboard_selection_", suffix=".png", delete=False)
    handle.write(raw); handle.close()
    return Path(handle.name)
def _cleanup(path: Path) -> None:
    try: path.unlink(missing_ok=True)
    except OSError: pass

def _image_bytes(data_url: str) -> bytes:
    path = _selection_file(data_url)
    try:
        return path.read_bytes()
    finally:
        _cleanup(path)

def _truthy(name: str, default: bool) -> bool:
    value = os.getenv(name, str(default)).strip().lower()
    return value in {"1", "true", "yes", "on"}

def _gemini_threshold() -> float:
    try:
        value = float(os.getenv("GEMINI_RECOGNITION_THRESHOLD", "0.70"))
        return value if 0.0 <= value <= 1.0 else 0.70
    except ValueError:
        return 0.70

_HANDWRITING_RECOGNIZER: Any = None
_HANDWRITING_LOAD_ATTEMPTED = False

def _local_handwriting(image_path: Path) -> dict[str, Any] | None:
    """Use the existing 62-class CNN when its checkpoint/runtime is available."""
    global _HANDWRITING_RECOGNIZER, _HANDWRITING_LOAD_ATTEMPTED
    checkpoint = MODELS / "handwriting" / "checkpoint_best.pth"
    if not checkpoint.is_file():
        return None
    try:
        if not _HANDWRITING_LOAD_ATTEMPTED:
            _HANDWRITING_LOAD_ATTEMPTED = True
            from app.recognition.handwriting.inference import HandwritingRecognizer
            _HANDWRITING_RECOGNIZER = HandwritingRecognizer(str(checkpoint))
        if _HANDWRITING_RECOGNIZER is None:
            return None
        prediction = _HANDWRITING_RECOGNIZER.recognize(str(image_path), top_k=3)
        return {"label": prediction.get("label"), "confidence": float(prediction.get("confidence", 0.0)),
                "candidates": prediction.get("top_k", []), "model": "local_handwriting_cnn_62"}
    except Exception:
        # The legacy recognizer is optional; it must never make the board API unavailable.
        _HANDWRITING_RECOGNIZER = None
        return None

@app.get("/", response_class=HTMLResponse)
def index() -> HTMLResponse: return HTMLResponse((STATIC / "index.html").read_text(encoding="utf-8"))
@app.get("/health")
def health() -> dict[str, str]: return {"status": "healthy"}

@app.post("/api/recognize")
def recognize(request: SelectionRequest) -> dict[str, Any]:
    path = _selection_file(request.image)
    try:
        # Optional OCR dependencies should not prevent the browser UI from starting.
        from app.recognition.ocr import OCRProcessor
        result = OCRProcessor(save_processed_images=False).extract_text_with_confidence(path)
        text, confidence = str(result.get("text", "")), float(result.get("confidence", 0))
        math = MathEngine().analyze(text, confidence)
        local = _local_handwriting(path)
        local_confidence = float((local or {}).get("confidence", 0.0))
        final_text = str((local or {}).get("label") or text)
        final_confidence = local_confidence if local else confidence
        gemini_interpretation = None
        fallback_used = False
        if (_truthy("GEMINI_RECOGNITION_FALLBACK", True) and local_confidence < _gemini_threshold()
                and GeminiService().is_available):
            fallback_used = True
            try:
                gemini_interpretation = GeminiService().analyze_canvas(request.subject, path.read_bytes(), local)
            except AIProviderError:
                # Recognition still returns its local/OCR result if AI assistance is temporarily unavailable.
                gemini_interpretation = None
        return {"text": final_text, "confidence": final_confidence, "mode": result.get("mode", "unknown"),
                "math": {"parsed": math.parsed, "normalized": math.normalized, "error": math.error},
                "local_model_result": local, "gemini_interpretation": gemini_interpretation,
                "gemini_fallback_used": fallback_used}
    finally: _cleanup(path)

@app.post("/api/plot")
def plot(request: PlotRequest) -> dict[str, Any]:
    math = MathEngine().analyze(request.expression, 1.0)
    if not math.parsed: raise HTTPException(422, f"The selected expression cannot be plotted ({math.error}).")
    try:
        graph = GraphEngine().build(math.normalized)
        return {"kind": graph.kind, "expression": graph.expression, "x": graph.x.tolist(), "y": graph.y.tolist(), "values": graph.values.tolist()}
    except Exception as exc: raise HTTPException(422, f"The selected expression could not be plotted: {type(exc).__name__}.") from exc

@app.post("/api/model")
def model(request: SelectionRequest) -> dict[str, Any]:
    path = _selection_file(request.image)
    try:
        result = ModelPipeline(output_dir=MODELS).process(path)
        result.pop("mesh", None)  # trimesh objects are not JSON serializable.
        if not result.get("success"):
            stage = str(result.get("stage") or "unknown")
            error = str(result.get("error") or "3D model generation did not complete")
            status = 503 if "trimesh is not installed" in error.lower() else 422
            raise HTTPException(status, detail={"stage": stage, "error": error})
        stl = result.get("stl") or {}
        filename = stl.get("filename")
        if filename:
            stl["download_url"] = f"/api/stl/{filename}"
            stl["viewer_url"] = stl["download_url"]
            result["stl"] = stl
        return result
    finally: _cleanup(path)

@app.get("/api/stl/{filename}")
def download_stl(filename: str) -> FileResponse:
    path = (MODELS / filename).resolve()
    if path.parent != MODELS or not path.is_file() or path.suffix.lower() != ".stl": raise HTTPException(404, "STL file not found.")
    return FileResponse(path, media_type="model/stl", filename=path.name)

@app.post("/api/ai/explain")
@app.post("/api/explain", include_in_schema=False)  # Compatibility route for the existing frontend.
def explain(request: ExplainRequest) -> dict[str, str]:
    if not request.content.strip() and not request.image:
        raise HTTPException(422, "Provide selected text or a canvas image to explain.")
    try:
        image = _image_bytes(request.image) if request.image else None
        return {"explanation": GeminiService().explain(request.subject, request.content, image)}
    except AIProviderError as exc:
        raise HTTPException(503, str(exc)) from exc

@app.post("/api/ai/analyze")
def analyze_canvas(request: AIAnalyzeRequest) -> dict[str, str]:
    try:
        return {"interpretation": GeminiService().analyze_canvas(request.subject, _image_bytes(request.image), request.local_result)}
    except AIProviderError as exc:
        raise HTTPException(503, str(exc)) from exc

@app.post("/api/ai/lecture")
@app.post("/api/ai/summarize")
def lecture_assistance(request: AILectureRequest) -> dict[str, str]:
    try:
        return {"content": GeminiService().lecture_assistance(request.subject, request.content, request.task)}
    except AIProviderError as exc:
        raise HTTPException(503, str(exc)) from exc

@app.post("/api/lecture/start")
def lecture_start(request: LectureStartRequest) -> dict[str, Any]:
    lecture_id = str(uuid.uuid4()); SESSIONS[lecture_id] = {"subject": request.subject, "started": datetime.now(timezone.utc).isoformat(), "captures": [], "slides": [], "ended": None}
    return {"id": lecture_id, "subject": request.subject, "status": "recording"}
@app.post("/api/lecture/{lecture_id}/capture")
def lecture_capture(lecture_id: str, request: LectureCaptureRequest) -> dict[str, Any]:
    session = SESSIONS.get(lecture_id)
    if not session or session["ended"]: raise HTTPException(404, "Active lecture not found.")
    session["captures"].append({"at": datetime.now(timezone.utc).isoformat(), "image": request.image, "objects": request.objects, "note": request.note})
    return {"captures": len(session["captures"])}
@app.post("/api/lecture/{lecture_id}/slides")
def lecture_slides(lecture_id: str, request: LectureSlidesRequest) -> dict[str, Any]:
    session = SESSIONS.get(lecture_id)
    if not session or session["ended"]: raise HTTPException(404, "Active lecture not found.")
    ordered = sorted(request.slides, key=lambda slide: slide.position)
    if [slide.position for slide in ordered] != list(range(1, len(ordered) + 1)):
        raise HTTPException(422, "Slides must have consecutive positions starting at one.")
    for slide in ordered: _selection_file(slide.image).unlink(missing_ok=True)
    session["slides"] = [slide.model_dump() for slide in ordered]
    return {"slides": len(session["slides"])}
@app.post("/api/lecture/{lecture_id}/end")
def lecture_end(lecture_id: str) -> dict[str, Any]:
    session = SESSIONS.get(lecture_id)
    if not session: raise HTTPException(404, "Lecture not found.")
    session["ended"] = datetime.now(timezone.utc).isoformat(); return {"id": lecture_id, "captures": len(session["captures"]), "status": "ended"}
@app.post("/api/lecture/{lecture_id}/ai-notes")
def lecture_ai_notes(lecture_id: str, request: LectureNotesRequest) -> dict[str, str]:
    session = SESSIONS.get(lecture_id)
    if not session or session["ended"]:
        raise HTTPException(404, "Active lecture not found.")
    try:
        notes = GeminiService().lecture_assistance(session["subject"], request.content, request.task)
    except AIProviderError as exc:
        raise HTTPException(503, str(exc)) from exc
    session["ai_notes"] = notes
    return {"content": notes}
@app.get("/api/lecture/{lecture_id}/pdf")
def lecture_pdf(lecture_id: str) -> Response:
    session = SESSIONS.get(lecture_id)
    if not session or not session["ended"]: raise HTTPException(409, "End the lecture before generating its PDF.")
    from matplotlib.backends.backend_pdf import PdfPages
    from matplotlib.figure import Figure
    output = io.BytesIO()
    with PdfPages(output) as pdf:
        cover = Figure(figsize=(11.69, 8.27)); cover.text(.08,.88,"AI Smart Blackboard",fontsize=24,weight="bold"); cover.text(.08,.78,f"Lecture subject: {session['subject']}",fontsize=15); cover.text(.08,.70,f"Canvas captures: {len(session['captures'])}",fontsize=12); pdf.savefig(cover)
        pages = session["slides"] or session["captures"]
        for number, capture in enumerate(pages, 1):
            fig = Figure(figsize=(11.69,8.27)); axis = fig.add_subplot(111); raw = base64.b64decode(capture["image"].split(",",1)[1]); import matplotlib.image as mpimg; axis.imshow(mpimg.imread(io.BytesIO(raw),format="png")); note = capture.get("note") or capture.get("title", ""); axis.set_title(f"Lecture canvas {number}" + (f" — {note}" if note else "")); axis.axis("off"); pdf.savefig(fig)
        if session.get("ai_notes"):
            notes = Figure(figsize=(11.69, 8.27)); notes.text(.08, .92, "AI-assisted lecture notes", fontsize=18, weight="bold")
            notes.text(.08, .86, session["ai_notes"], fontsize=10, va="top", wrap=True); pdf.savefig(notes)
    return Response(output.getvalue(), media_type="application/pdf", headers={"Content-Disposition": f'attachment; filename="smart-blackboard-{lecture_id[:8]}.pdf"'})
