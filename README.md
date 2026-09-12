# AI Smart Blackboard

AI Smart Blackboard is an interactive digital canvas and AI-powered educational assistant designed to enhance classroom lectures, remote teaching, and interactive study sessions. It bridges the gap between traditional digital whiteboards and modern artificial intelligence by turning handwritten notes, mathematical expressions, and visual shapes into structured, actionable content in real time.

The system solves the friction of traditional teaching setups where instructors must constantly switch between disparate tools for drawing, plotting functions, rendering 3D models, recording audio lectures, and generating study materials. By integrating an intuitive drawing canvas with a FastAPI backend powered by Google Gemini AI, AI Smart Blackboard acts as a unified digital workstation for educators and students alike.

---

## Features

### Interactive Canvas & Drawing Tools

- **Pen & Freehand Drawing:** Smooth stroke rendering with customizable brush sizes and colors.
- **Eraser:** Precise point and object-level erasure.
- **Object Manipulation:** Single-object selection, multi-selection, moving, and resizing drawn elements across the canvas.
- **Shape Rendering:** Built-in drawing tools for standard geometric primitives (lines, rectangles, circles, arrows).
- **Multi-Slide Management:** Support for creating, switching, reordering, and clearing multiple slides/canvases in a single session.

### AI & Recognition Capabilities

- **Handwriting & Math Recognition:** Intelligent OCR processing to convert handwritten notes, text, and complex mathematical formulas into clean digital text and LaTeX math strings.
- **AI-Assisted Explanations:** Deep integration with Google Gemini AI to analyze canvas content, explain complex step-by-step problem solutions, and answer real-time contextual questions.
- **Subject Library Context:** Pre-configured prompts and domain modes tailored for Mathematics, Physics, Chemistry, and Engineering.

### Dynamic Generation & Visualization

- **Plotting & Graph Generation:** Automatically parses mathematical equations from the canvas and renders interactive 2D function plots and data visualizations.
- **3D Shape & Model Generation:** Converts visual spatial descriptions or geometry sketches into renderable 3D objects.
- **3D Canvas Viewer & STL Export:** Interactive embedded 3D viewer with rotation/zoom controls and direct export to `.stl` format for 3D printing or CAD use.

### Lecture Recording & Document Export

- **Lecture Mode & Voice Recording:** Built-in audio capture with pause, resume, and stop controls to sync voice explanations with canvas activity.
- **Audio Download:** Export recorded audio tracks directly as standard formats (`.wav` / `.mp3`).
- **Canvas Capture:** High-resolution snapshot tool to export individual slides as image files (`.png`).
- **PDF Document Generation:** Compiles all active slides and synchronized notes into a structured PDF document for quick distribution.

### Core Architecture

- **FastAPI Backend:** High-performance REST API handling asynchronous requests for AI processing, plotting, 3D mesh processing, and PDF compilation.
- **Gemini API Integration:** Direct integration with Google Gemini models via official SDKs for rapid multimodal visual and text reasoning.

---

## How the System Works

AI Smart Blackboard operates on a client-server architecture separating interactive user input from heavy AI computation and document rendering.

```text
User ──> Web Interface ──> Canvas API ──> FastAPI Backend ──> Service Layer ──> Output/Viewer
                                                                ├── Gemini AI API
                                                                ├── Matplotlib/Plotting
                                                                ├── 3D/Mesh Processing
                                                                └── PDF Engine
```

1. **User Interaction & Capture:** The user interacts with the HTML5 Canvas web interface using a stylus, mouse, or touch input. Drawn strokes, text, and voice signals are recorded in the client runtime.
2. **API Communication:** Upon triggering an action (e.g., "Recognize Math", "Explain Concept", "Generate 3D", or "Export PDF"), the frontend packages canvas image blobs, stroke vectors, or audio streams and sends them via asynchronous HTTP requests to the FastAPI backend.
3. **Backend Service Dispatch:** FastAPI routes the payload to dedicated service modules:
   - **Recognition & AI Service:** Formats image payload and query context to send to the Google Gemini API.
   - **Plotting Engine:** Parses math expressions into executable numpy/matplotlib code to produce vector/raster graphs.
   - **3D Modeling Service:** Generates volumetric geometry meshes and converts them to STL format.
   - **PDF Export Service:** Stitches canvas slide images into a single formatted document.
4. **Response Rendering:** Results are returned as JSON payloads, streamable media, or binary file downloads to be immediately rendered in the web UI.

---

## Project Architecture

```text
AI-Smart-Blackboard/
├── app/
│   ├── main.py                  # FastAPI application entry point & CORS configuration
│   ├── config.py                # Environment variables & system configuration settings
│   ├── api/
│   │   ├── router.py            # Main API router aggregating endpoints
│   │   ├── endpoints/
│   │   │   ├── canvas.py        # Canvas state & image payload endpoints
│   │   │   ├── ai.py            # Gemini recognition & explanation endpoints
│   │   │   ├── plot.py          # 2D graph generation endpoints
│   │   │   ├── modeling.py      # 3D object rendering & STL generation endpoints
│   │   │   └── export.py        # PDF compilation & lecture audio endpoints
│   ├── services/
│   │   ├── gemini_service.py    # Google Gemini API wrapper & prompt handlers
│   │   ├── plot_service.py      # Math parsing & Matplotlib chart generator
│   │   ├── stl_service.py       # 3D geometry processing & STL exporter
│   │   ├── pdf_service.py       # Multi-slide PDF document builder
│   │   └── audio_service.py     # Audio processing and handling helper
│   └── static/                  # Static assets & web interface frontend
│       ├── index.html           # Main digital blackboard interface
│       ├── css/
│       │   └── style.css        # Main application stylesheet
│       └── js/
│           ├── main.js          # Main app coordinator & event listeners
│           ├── canvas.js        # Core canvas drawing, tools, & slide engine
│           ├── ai.js            # Frontend Gemini AI interaction handlers
│           ├── viewer3d.js      # Three.js / STL web viewer integration
│           └── recorder.js      # Audio voice recording module
├── .env.example                 # Template for environment variables
├── .gitignore                   # Files excluded from version control
├── README.md                    # Project documentation
└── requirements.txt             # Python backend dependency manifest
```

---

## Getting Started

### Prerequisites

- **Python:** Version 3.10 or higher
- **Google Gemini API Key:** Required for AI vision, math recognition, and contextual explanations.

### Installation

1. **Clone the repository:**

   ```bash
   git clone https://github.com/M-Sameer-91/AI-Smart-Blackboard.git
   cd AI-Smart-Blackboard
   ```

2. **Create and activate a virtual environment:**

   - **Linux/macOS:**

     ```bash
     python3 -m venv venv
     source venv/bin/activate
     ```

   - **Windows:**

     ```cmd
     python -m venv venv
     venv\Scripts\activate
     ```

3. **Install backend dependencies:**

   ```bash
   pip install -r requirements.txt
   ```

4. **Set up environment variables:**

   Copy `.env.example` to `.env` and fill in your Gemini API key:

   ```bash
   cp .env.example .env
   ```

   Open `.env` and configure:

   ```env
   GEMINI_API_KEY=your_actual_gemini_api_key_here
   HOST=127.0.0.1
   PORT=8000
   ```

---

## Running the Application

1. **Start the FastAPI backend server:**

   ```bash
   uvicorn app.main:app --reload
   ```

2. **Access the application:**

   Open your browser and navigate to:

   ```text
   http://127.0.0.1:8000
   ```

3. **API Documentation:**

   Explore interactive API docs provided by FastAPI:

   - **Swagger UI:** `http://127.0.0.1:8000/docs`
   - **ReDoc:** `http://127.0.0.1:8000/redoc`
