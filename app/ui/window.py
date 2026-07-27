import customtkinter as ctk
from app.ui.canvas import DrawingCanvas


class SmartBlackboard(ctk.CTk):
    def __init__(self):
        super().__init__()

        # -----------------------------
        # Window Configuration
        # -----------------------------
        self.title("AI Smart Blackboard")
        self.geometry("1200x800")
        self.minsize(900, 600)

        ctk.set_appearance_mode("Dark")
        ctk.set_default_color_theme("blue")

        # -----------------------------
        # Toolbar
        # -----------------------------
        self.toolbar = ctk.CTkFrame(self, height=50)
        self.toolbar.pack(side="top", fill="x")

        self.pen_btn = ctk.CTkButton(self.toolbar, text="Pen")
        self.pen_btn.pack(side="left", padx=10, pady=10)

        self.eraser_btn = ctk.CTkButton(self.toolbar, text="Eraser")
        self.eraser_btn.pack(side="left", padx=10, pady=10)

        self.clear_btn = ctk.CTkButton(self.toolbar, text="Clear")
        self.clear_btn.pack(side="left", padx=10, pady=10)

        self.recognize_btn = ctk.CTkButton(self.toolbar, text="Recognize")
        self.recognize_btn.pack(side="left", padx=10, pady=10)

        # -----------------------------
        # Drawing Canvas
        # -----------------------------
        self.canvas = DrawingCanvas(self)
        self.canvas.pack(fill="both", expand=True)

        # -----------------------------
        # Status Bar
        # -----------------------------
        self.status = ctk.CTkLabel(
            self,
            text="Status : Ready",
            anchor="w"
        )
        self.status.pack(side="bottom", fill="x", padx=10, pady=5)