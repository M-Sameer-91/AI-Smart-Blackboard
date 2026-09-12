"""
main.py - Application Entry Point for AI Smart Blackboard
"""

import sys
import traceback
from typing import NoReturn


def check_dependencies() -> bool:
    """
    Check if all required dependencies are installed.
    """
    missing = []
    
    try:
        import customtkinter
    except ImportError:
        missing.append("customtkinter")
    
    try:
        import tkinter
    except ImportError:
        missing.append("tkinter")
    
    # Check optional dependencies (not required for basic functionality)
    optional_missing = []
    try:
        import torch
    except ImportError:
        optional_missing.append("torch (required for OCR)")
    
    try:
        import transformers
    except ImportError:
        optional_missing.append("transformers (required for OCR)")
    
    if missing:
        print("ERROR: Missing required dependencies:", file=sys.stderr)
        for dep in missing:
            print(f"  - {dep}", file=sys.stderr)
        print("\nPlease install them using:", file=sys.stderr)
        print("pip install customtkinter", file=sys.stderr)
        return False
    
    if optional_missing:
        print("WARNING: Missing optional dependencies:", file=sys.stderr)
        for dep in optional_missing:
            print(f"  - {dep}", file=sys.stderr)
        print("\nTo enable OCR functionality, install:", file=sys.stderr)
        print("pip install torch transformers pillow", file=sys.stderr)
        print("For CPU-only PyTorch: pip install torch --index-url https://download.pytorch.org/whl/cpu", file=sys.stderr)
    
    return True


def main() -> NoReturn:
    """
    Main application entry point.
    """
    # Check dependencies before attempting to run
    if not check_dependencies():
        sys.exit(1)
    
    try:
        from app.ui.window import SmartBlackboard
        
        app = SmartBlackboard()
        app.run()
        sys.exit(0)
        
    except KeyboardInterrupt:
        print("\nApplication terminated by user (KeyboardInterrupt)")
        sys.exit(130)
        
    except ImportError as e:
        error_msg = f"Import Error: {str(e)}\n\n"
        error_msg += "Please ensure all required packages are installed:\n"
        error_msg += "pip install customtkinter\n"
        error_msg += "pip install torch transformers pillow\n"
        
        print(error_msg, file=sys.stderr)
        
        try:
            import tkinter.messagebox as messagebox
            from tkinter import Tk
            
            root = Tk()
            root.withdraw()
            
            messagebox.showerror(
                "Import Error",
                f"Missing required module: {str(e)}\n\n"
                "Please install required dependencies:\n"
                "pip install customtkinter\n"
                "pip install torch transformers pillow"
            )
            root.destroy()
            
        except Exception:
            pass
        
        sys.exit(1)
        
    except Exception as e:
        error_msg = f"Fatal error during application startup or runtime:\n\n"
        error_msg += f"Exception: {type(e).__name__}\n"
        error_msg += f"Message: {str(e)}\n\n"
        error_msg += "Full traceback:\n"
        error_msg += traceback.format_exc()
        
        print(error_msg, file=sys.stderr)
        
        try:
            import tkinter.messagebox as messagebox
            from tkinter import Tk
            
            root = Tk()
            root.withdraw()
            
            messagebox.showerror(
                "Application Error",
                "The application failed to start due to an unexpected error.\n\n"
                f"Error: {type(e).__name__}: {str(e)}\n\n"
                "Please check the console for detailed error information."
            )
            root.destroy()
            
        except Exception:
            pass
        
        sys.exit(1)


if __name__ == "__main__":
    main()