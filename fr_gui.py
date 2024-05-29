import cv2
import numpy as np
import face_recognition as fr
import os
import pickle
from tkinter import Tk, Label, Button, Entry, Canvas, NW, filedialog
from PIL import Image, ImageTk

class FaceRecognitionApp:
    def __init__(self, master):
        self.master = master
        master.title("Face Recognition App")

        self.known_face_encodings = []
        self.known_face_names = []
        self.face_locations = []
        self.face_encodings = []
        self.face_names = []
        self.process_this_frame = True

        self.load_known_faces()

        self.camera = cv2.VideoCapture(0)
        self.camera_width = int(self.camera.get(cv2.CAP_PROP_FRAME_WIDTH))
        self.camera_height = int(self.camera.get(cv2.CAP_PROP_FRAME_HEIGHT))

        self.create_widgets()

        self.video_loop()

    def create_widgets(self):
        self.canvas = Canvas(self.master, width=self.camera_width, height=self.camera_height)
        self.canvas.grid(row=0, column=0, columnspan=2)

        self.name_label = Label(self.master, text="Nome:")
        self.name_label.grid(row=1, column=0)

        self.name_entry = Entry(self.master)
        self.name_entry.grid(row=1, column=1)

        self.scan_button = Button(self.master, text="Escanear Rosto", command=self.scan_face)
        self.scan_button.grid(row=2, column=0)

        self.load_button = Button(self.master, text="Carregar Imagem", command=self.load_image)
        self.load_button.grid(row=2, column=1)

    def video_loop(self):
        ret, frame = self.camera.read()
        if ret:
            self.process_frame(frame)
            self.show_frame(frame)
        self.master.after(10, self.video_loop)

    def process_frame(self, frame):
        small_frame = cv2.resize(frame, (0, 0), fx=0.25, fy=0.25)
        rgb_small_frame = small_frame[:, :, ::-1]

        if self.process_this_frame:
            self.face_locations = fr.face_locations(rgb_small_frame)
            self.face_encodings = fr.face_encodings(rgb_small_frame, self.face_locations)

            self.face_names = []
            for face_encoding in self.face_encodings:
                print('face encoding:', face_encoding)
                print('known face encodings:', self.known_face_encodings)
                matches = fr.compare_faces(self.known_face_encodings, face_encoding)
                name = "Desconhecido"

                face_distances = fr.face_distance(self.known_face_encodings, face_encoding)
                best_match_index = np.argmin(face_distances)
                if matches[best_match_index]:
                    name = self.known_face_names[best_match_index]

                self.face_names.append(name)

        self.process_this_frame = not self.process_this_frame

    def show_frame(self, frame):
        for (top, right, bottom, left), name in zip(self.face_locations, self.face_names):
            top *= 4
            right *= 4
            bottom *= 4
            left *= 4

            cv2.rectangle(frame, (left, top), (right, bottom), (0, 0, 255), 2)
            cv2.rectangle(frame, (left, bottom - 35), (right, bottom), (0, 0, 255), cv2.FILLED)
            font = cv2.FONT_HERSHEY_DUPLEX
            cv2.putText(frame, name, (left + 6, bottom - 6), font, 1.0, (255, 255, 255), 1)

        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        photo = ImageTk.PhotoImage(image=Image.fromarray(rgb_frame))
        self.canvas.create_image(0, 0, image=photo, anchor=NW)
        self.canvas.photo = photo

    def scan_face(self):
        ret, frame = self.camera.read()
        if ret:
            face_locations = fr.face_locations(frame)
            if face_locations:
                face_encoding = fr.face_encodings(frame, face_locations)[0]
                name = self.name_entry.get()
                if name:  # Ensure name is not empty
                    self.known_face_encodings.append(face_encoding)
                    self.known_face_names.append(name)
                    self.save_known_faces()
                    self.name_entry.delete(0, 'end')
                else:
                    print("Nome não pode ser vazio.")
            else:
                print("Nenhum rosto detectado.")
        else:
            print("Erro ao capturar imagem da câmera.")

    def load_image(self):
        file_path = filedialog.askopenfilename(
            initialdir="/",
            title="Selecione uma imagem",
            filetypes=(("Arquivos de imagem", "*.jpg;*.jpeg;*.png"), ("Todos os arquivos", "*.*"))
        )
        if file_path:
            image = fr.load_image_file(file_path)
            face_locations = fr.face_locations(image)
            if face_locations:
                face_encoding = fr.face_encodings(image, face_locations)[0]
                name = os.path.splitext(os.path.basename(file_path))[0]  # Use the filename without extension as the name
                self.known_face_encodings.append(face_encoding)
                self.known_face_names.append(name)
                self.save_known_faces()
            else:
                print("Nenhum rosto detectado na imagem.")

    def load_known_faces(self):
        try:
            with open("known_faces.dat", "rb") as f:
                self.known_face_encodings, self.known_face_names = pickle.load(f)
        except FileNotFoundError:
            print("Arquivo de rostos conhecidos não encontrado.")
        except EOFError:
            print("Arquivo de rostos conhecidos está vazio.")

    def save_known_faces(self):
        with open("known_faces.dat", "wb") as f:
            pickle.dump((self.known_face_encodings, self.known_face_names), f)

if __name__ == "__main__":
    root = Tk()
    app = FaceRecognitionApp(root)
    root.mainloop()
