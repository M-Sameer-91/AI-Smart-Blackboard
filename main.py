"""
main.py - Application Entry Point for AI Smart Blackboard

This module serves as the entry point for the AI Smart Blackboard application.
It is responsible for initializing and launching the main application window
with proper error handling and startup safety.

The module follows the principle of separation of concerns by delegating
all UI management, drawing operations, and application logic to their
respective modules (window.py, canvas.py, etc.).
"""

import sys
import traceback
from typing import NoReturn


def check_dependencies() -> bool:
    """
    Check if all required dependencies are installed.
    
    Returns:
        bool: True if all dependencies are available, False otherwise.
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
    
    if missing:
        print("ERROR: Missing required dependencies:", file=sys.stderr)
        for dep in missing:
            print(f"  - {dep}", file=sys.stderr)
        print("\nPlease install them using:", file=sys.stderr)
        print("pip install customtkinter", file=sys.stderr)
        return False
    
    return True


def main() -> NoReturn:
    """
    Main application entry point.
    
    Initializes and runs the Smart Blackboard application with comprehensive
    error handling. This function should be the only entry point for the
    application and should never return (it exits via sys.exit or the
    GUI event loop).
    
    The function handles:
        - Dependency checking
        - Application instance creation
        - GUI event loop execution
        - Graceful error handling and reporting
        - Clean application shutdown
    
    Returns:
        NoReturn: This function never returns normally.
        It either runs the GUI event loop until application exit
        or terminates via sys.exit() on error.
    
    Raises:
        SystemExit: Always raised when the application terminates,
                   either normally or with an error code.
    """
    # Check dependencies before attempting to run
    if not check_dependencies():
        sys.exit(1)
    
    try:
        # Import the main application class from the UI module
        from app.ui.window import SmartBlackboard
        
        # Create the main application window instance
        # The SmartBlackboard class handles all UI initialization
        # and window configuration
        app = SmartBlackboard()
        
        # Start the GUI event loop using the run() method
        app.run()
        
        # Normal application exit with success code
        sys.exit(0)
        
    except KeyboardInterrupt:
        # Handle Ctrl+C gracefully
        print("\nApplication terminated by user (KeyboardInterrupt)")
        sys.exit(130)
        
    except ImportError as e:
        # Handle missing module imports specifically
        error_msg = f"Import Error: {str(e)}\n\n"
        error_msg += "Please ensure all required packages are installed:\n"
        error_msg += "pip install customtkinter\n"
        
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
                "pip install customtkinter"
            )
            root.destroy()
            
        except Exception:
            pass
        
        sys.exit(1)
        
    except Exception as e:
        # Catch all unexpected exceptions for graceful error handling
        error_msg = f"Fatal error during application startup or runtime:\n\n"
        error_msg += f"Exception: {type(e).__name__}\n"
        error_msg += f"Message: {str(e)}\n\n"
        error_msg += "Full traceback:\n"
        error_msg += traceback.format_exc()
        
        # Print to stderr for logging/debugging
        print(error_msg, file=sys.stderr)
        
        # Display error in a message box if possible
        try:
            import tkinter.messagebox as messagebox
            from tkinter import Tk
            
            # Create temporary root for message box
            root = Tk()
            root.withdraw()  # Hide the root window
            
            messagebox.showerror(
                "Application Error",
                "The application failed to start due to an unexpected error.\n\n"
                f"Error: {type(e).__name__}: {str(e)}\n\n"
                "Please check the console for detailed error information."
            )
            root.destroy()
            
        except Exception as msgbox_error:
            # Fallback: print error if tkinter message box fails
            print(f"\nCould not display error dialog: {msgbox_error}", file=sys.stderr)
        
        # Exit with non-zero error code to indicate failure
        sys.exit(1)


if __name__ == "__main__":
    """
    Application entry point when script is executed directly.
    
    This block ensures the application runs only when the script is
    executed directly (not imported as a module), following Python's
    best practices for executable modules.
    """
    main()