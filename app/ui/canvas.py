import tkinter as tk


class DrawingCanvas(tk.Canvas):
    def __init__(self, parent):
        super().__init__(
            parent,
            bg="black",
            width=900,
            height=600,
            highlightthickness=0
        )

        self.last_x = None
        self.last_y = None

        self.bind("<Button-1>", self.start_draw)
        self.bind("<B1-Motion>", self.draw)

    def start_draw(self, event):
        print("Mouse Pressed", event.x, event.y)
        self.last_x = event.x
        self.last_y = event.y

    def draw(self, event):
        self.create_line(
            self.last_x,
            self.last_y,
            event.x,
            event.y,
            fill="white",
            width=3,
            capstyle=tk.ROUND,
            smooth=True
        )

        self.last_x = event.x
        self.last_y = event.y