"""
Script to continuously scroll down
Press ESC to stop
"""

from pynput.mouse import Controller
from pynput import keyboard
import time

# Initialize mouse controller
mouse = Controller()

# Flag to control the script
running = True

def on_press(key):
    """Stop the script when ESC is pressed"""
    global running
    if key == keyboard.Key.esc:
        print("\nESC pressed - stopping...")
        running = False
        return False

def main():
    global running
    
    print("Continuously scrolling down...")
    print("Press ESC to stop\n")
    
    # Start keyboard listener
    listener = keyboard.Listener(on_press=on_press)
    listener.start()
    
    # Continuously scroll down until ESC is pressed
    try:
        while running:
            mouse.scroll(0, -3)  # Scroll down (negative value = down)
            time.sleep(0.05)  # Small delay between scrolls (adjust for speed)
    finally:
        print("Scrolling stopped.")

if __name__ == "__main__":
    main()