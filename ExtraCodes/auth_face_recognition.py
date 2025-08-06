import face_recognition
import cv2
import pickle
import os
import numpy as np
import tkinter as tk
from tkinter import messagebox
from pathlib import Path
import requests


class DlibFaceUnlock:
    def __init__(self):
        self.labels_path = 'labels.pickle'
        self.faces_path = 'KnownFace.pickle'
        self.image_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'images')
        self.labels_ids = {}
        self.known_faces = []

        if os.path.exists(self.labels_path):
            with open(self.labels_path, 'rb') as f:
                self.og_labels = pickle.load(f)
        else:
            self.og_labels = {}

        self._load_or_update_faces()

    def _load_or_update_faces(self):
        current_id = 0

        for root, dirs, files in os.walk(self.image_dir):
            for file in files:
                if file.endswith(('png', 'jpg')):
                    label = os.path.basename(os.path.dirname(os.path.join(root, file))).replace(' ', '-').lower()
                    if label not in self.labels_ids:
                        self.labels_ids[label] = current_id
                        current_id += 1

        if self.labels_ids != self.og_labels:
            with open(self.labels_path, 'wb') as f:
                pickle.dump(self.labels_ids, f)

            self.known_faces.clear()
            for label in self.labels_ids:
                user_folder = os.path.join(self.image_dir, label)
                for img_file in os.listdir(user_folder):
                    img_path = os.path.join(user_folder, img_file)
                    img = face_recognition.load_image_file(img_path)
                    encodings = face_recognition.face_encodings(img)
                    if encodings:
                        self.known_faces.append([label, encodings[0]])

            with open(self.faces_path, 'wb') as f:
                pickle.dump(self.known_faces, f)
        else:
            if os.path.exists(self.faces_path):
                with open(self.faces_path, 'rb') as f:
                    self.known_faces = pickle.load(f)

    def ID(self):
        cap = cv2.VideoCapture(0)
        matched_names = []

        while True:
            ret, frame = cap.read()
            if not ret:
                break

            rgb_frame = frame[:, :, ::-1]
            face_locations = face_recognition.face_locations(rgb_frame)
            face_encodings = face_recognition.face_encodings(rgb_frame, face_locations)

            for encoding in face_encodings:
                for label, known_encoding in self.known_faces:
                    match = face_recognition.compare_faces([known_encoding], encoding)[0]
                    if match:
                        matched_names.append(label)
                        cap.release()
                        cv2.destroyAllWindows()
                        return matched_names

            cv2.imshow("Face Login", frame)
            if cv2.waitKey(1) & 0xFF == 27:
                break

        cap.release()
        cv2.destroyAllWindows()
        return matched_names


def sanitize_email(email):
    return email.replace('@', '_at_').replace('.', '_dot_')


def desanitize_email(sanitized):
    return sanitized.replace('_at_', '@').replace('_dot_', '.')


def register():
    user_email = email.get().strip().lower()
    if not user_email:
        messagebox.showwarning("Warning", "Please enter an email.")
        return

    sanitized_email = sanitize_email(user_email)
    user_path = os.path.join("images", sanitized_email)
    Path(user_path).mkdir(parents=True, exist_ok=True)
    number_of_files = len(os.listdir(user_path)) + 1

    cam = cv2.VideoCapture(0)
    cv2.namedWindow("Register")

    while True:
        ret, frame = cam.read()
        if not ret:
            break
        cv2.imshow("Register", frame)
        k = cv2.waitKey(1)
        if k % 256 == 27:  # ESC key
            break
        elif k % 256 == 32:  # SPACE key
            img_name = f"{number_of_files}.png"
            cv2.imwrite(os.path.join(user_path, img_name), frame)
            break

    cam.release()
    cv2.destroyAllWindows()

    # Send to Node.js API
    try:
        response = requests.post("http://localhost:8080/authUser/signup", json={
            "email": user_email,
            "authType": "face",
        })
        print("Response JSON:", response.json())
        if response.status_code in [200, 201]:
            messagebox.showinfo("Info", response.json().get("message", "Registration successful."))
        else:
            messagebox.showerror("Error", f"Registration failed.\nCode: {response.status_code}\nMessage: {response.json().get('message')}")
    except Exception as e:
        messagebox.showerror("Error", f"API request failed: {e}")

    raise_frame(login_frame)


def login():
    recognizer = DlibFaceUnlock()
    user = recognizer.ID()
    if not user:
        messagebox.showerror("Alert", "Face Not Recognised")
    else:
        logged_in_user.set(desanitize_email(user[0]))
        raise_frame(user_menu_frame)


def raise_frame(frame):
    frame.tkraise()


# GUI Setup
root = tk.Tk()
root.title("Face Recognition Login")
root.geometry("800x500")

# Frames
login_frame = tk.Frame(root, bg="white")
reg_frame = tk.Frame(root, bg="white")
user_menu_frame = tk.Frame(root, bg="white")

for frame in (login_frame, reg_frame, user_menu_frame):
    frame.grid(row=0, column=0, sticky='news')

# Variables
email = tk.StringVar()
logged_in_user = tk.StringVar()

# Login Frame
tk.Label(login_frame, text="Face Recognition", font=("Courier", 40), bg="white").grid(row=0, column=0, columnspan=2, pady=20)
tk.Button(login_frame, text="Login", font=("Arial", 25), command=login).grid(row=1, column=1, pady=20)
tk.Button(login_frame, text="Register", font=("Arial", 25), command=lambda: raise_frame(reg_frame)).grid(row=1, column=0, pady=20)

# Register Frame
tk.Label(reg_frame, text="Register", font=("Courier", 40), bg="white").grid(row=0, column=0, columnspan=2, pady=20)
tk.Label(reg_frame, text="Email:", font=("Arial", 20), bg="white").grid(row=1, column=0, pady=10)
tk.Entry(reg_frame, textvariable=email, font=("Arial", 20)).grid(row=1, column=1, pady=10)
tk.Button(reg_frame, text="Register", font=("Arial", 25), command=register).grid(row=2, column=1, pady=20)
tk.Button(reg_frame, text="Back", font=("Arial", 25), command=lambda: raise_frame(login_frame)).grid(row=2, column=0, pady=20)

# User Menu Frame
tk.Label(user_menu_frame, text="Hello,", font=("Courier", 40), bg="white").grid(row=0, column=0, pady=20)
tk.Label(user_menu_frame, textvariable=logged_in_user, font=("Courier", 40), bg="white", fg="red").grid(row=0, column=1, pady=20)
tk.Button(user_menu_frame, text="Back", font=("Arial", 25), command=lambda: raise_frame(login_frame)).grid(row=1, column=0, columnspan=2, pady=20)

# Start GUI
raise_frame(login_frame)
root.mainloop()
