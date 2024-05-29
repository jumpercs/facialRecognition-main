import cv2
import numpy as np
import face_recognition as fr
import os

# Definindo variáveis globais
known_face_encodings = []
known_face_names = []

import cv2
import face_recognition as fr

def show_webcam(mirror=False, cam=0):
    # Obter a lista de câmeras disponíveis
    available_cameras = get_available_cameras()  # Assuma que você tenha uma função get_available_cameras() implementada
    print(available_cameras)
    # Usar a primeira câmera disponível
    cam = cv2.VideoCapture(available_cameras[cam])

    # Inicializa o tempo para calcular o FPS
    start_time = 0
    frame_count = 0

    while True:
        ret_val, img = cam.read()
        if not ret_val:
            break

        # Adicionar quadrado ao rosto detectado na imagem e o nome da pessoa se estiver na lista known_face_encodings
        face_locations = fr.face_locations(img)
        face_encodings = fr.face_encodings(img, face_locations)
        for (top, right, bottom, left), face_encoding in zip(face_locations, face_encodings):
            matches = fr.compare_faces(known_face_encodings, face_encoding)  # known_face_encodings e known_face_names devem ser definidos
            name = "Unknown"
            if True in matches:
                first_match_index = matches.index(True)
                name = known_face_names[first_match_index]
            cv2.rectangle(img, (left, top), (right, bottom), (0, 0, 255), 2)
            cv2.rectangle(img, (left, bottom - 35), (right, bottom), (0, 0, 255), cv2.FILLED)
            font = cv2.FONT_HERSHEY_DUPLEX
            cv2.putText(img, name, (left + 6, bottom - 6), font, 1.0, (255, 255, 255), 1)

        # Calcular o frametime e o FPS
        end_time = cv2.getTickCount()
        # Tempo de processamento por frame (em segundos)
        frame_time = (end_time - start_time) / cv2.getTickFrequency()
        # Atualiza o tempo inicial para o próximo frame
        start_time = end_time
        # Aumenta a contagem de frames
        frame_count += 1
        # Calcula o FPS
        fps = 1.0 / frame_time
        # Mostra o FPS na tela
        cv2.putText(img, "FPS: {:.2f}".format(fps), (10, 30), font, 1.0, (255, 255, 255), 1)

        if mirror:
            img = cv2.flip(img, 1)
        cv2.imshow('my webcam', img)
        if cv2.waitKey(1) == 27:
            break  # esc para sair
    cv2.destroyAllWindows()
    cam.release()

def get_available_cameras():
    # Tentar obter uma lista de todas as câmeras em 5 segundos
    index = 0
    arr = []
    while True:
        cap = cv2.VideoCapture(index)
        if not cap.read()[0]:
            break
        else:
            arr.append(index)
            print("Camera {} is available".format(index))
        cap.release()
        index += 1
    return arr

def list_ports():
    """
    Testar as portas e retornar uma tupla com as portas disponíveis e as que estão funcionando.
    """
    is_working = True
    dev_port = 0
    working_ports = []
    available_ports = []
    while is_working:
        camera = cv2.VideoCapture(dev_port)
        if not camera.isOpened():
            is_working = False
            print("Port %s is not working." % dev_port)
        else:
            is_reading, img = camera.read()
            w = camera.get(3)
            h = camera.get(4)
            if is_reading:
                print("Port %s is working and reads images (%s x %s)" % (dev_port, h, w))
                working_ports.append(dev_port)
            else:
                print("Port %s for camera (%s x %s) is present but does not read." % (dev_port, h, w))
                available_ports.append(dev_port)
        dev_port += 1
    return available_ports, working_ports

def generate_face_encodings():
    global known_face_encodings, known_face_names
    # Carregar imagens de exemplo na pasta pessoas e armazenar o nome da pessoa em known_face_names e as codificações faciais em known_face_encodings
    for file in os.listdir("pessoas"):
        print(file)
        if file.endswith(".jpg"):
            print("Carregando imagem: " + file)
            image = fr.load_image_file("pessoas/" + file)
            face_encoding = fr.face_encodings(image)[0]
            print(face_encoding)
            known_face_encodings.append(face_encoding)
            known_face_names.append(file.split(".")[0])

def main():
    generate_face_encodings()
    show_webcam()

if __name__ == '__main__':
    main()
