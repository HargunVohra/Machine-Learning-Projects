import tkinter as tk
from tkinter import ttk
from PIL import Image, ImageTk


def face_expression():
    import mainface

def hand_volume():
    import handvolumemain

def hand_mouse():
    import mousefinger
    mousefinger
def parking_space_detection():
    import maincar 
    maincar()

def Hand_keyboard():
    import mainpainter

# Create main window
root = tk.Tk()
root.title("Main Menu")

# Set window size
root.geometry("750x422")

root.resizable(False, False)

# Load the background image
bg_image = Image.open("IMAGE/bg.jpg")
bg_photo = ImageTk.PhotoImage(bg_image)

# Create a label with the background image
bg_label = tk.Label(root, image=bg_photo)
bg_label.place(x=0, y=0, relwidth=1, relheight=1)

# Create a frame for the main menu options
menu_frame = ttk.Frame(root, padding="20")
menu_frame.place(relx=0.5, rely=0.5, anchor=tk.CENTER)
menu_frame.configure(style='Transparent.TFrame')  # Set transparent style

# Define main menu options
options = [
    ("Face Expression", face_expression),
    ("Hand Volume", hand_volume),
    ("Hand Mouse", hand_mouse),
    ("Parking Space Detection", parking_space_detection),
    ("Hand Keyboard", Hand_keyboard)
]

# Create and layout main menu buttons
for text, command in options:
    button = ttk.Button(menu_frame, text=text, command=command, width=60)
    button.pack(side="top", pady=5, fill="none")
    button_style = ttk.Style()
    button_style.configure('Custom.TButton', padding=(10, 60))  # Adjust the padding to set the height
    button_style.configure('Custom.TButton', background=root.cget('bg'))  # Set button background to window background color



# Create a custom style to make the frame and buttons transparent
style = ttk.Style()
style.theme_use('default')
style.configure('Transparent.TFrame', background=root.cget('bg'))  # Set frame background to window background color
style.configure('Transparent.TButton', background=root.cget('bg'), foreground='black')  # Set button background to window background color

root.mainloop()
