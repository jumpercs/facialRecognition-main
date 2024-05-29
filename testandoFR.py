import cv2
import numpy as np
import face_recognition as fr



def show_webcam(mirror=False, cam=1):
    print("Iniciando webcam")
    cam = cv2.VideoCapture(1)
    print(cam.get(cv2.CAP_PROP_FRAME_WIDTH))
    while True:
        ret_val, img = cam.read()
        print(img.shape)

        #img = cv2.resize(img, (0,0), fx=0.5, fy=0.5)
        #print(img.shape)

        face_locations = fr.face_locations(img)
    
        print("I found {} face(s) in this photograph.".format(len(face_locations)))


        for face_location in face_locations:
                
                top, right, bottom, left = face_location
                print("A face is located at pixel location Top: {}, Left: {}, Bottom: {}, Right: {}".format(top, left, bottom, right))
    
                # You can access the actual face itself like this:
                face_image = img[top:bottom, left:right]
                cv2.rectangle(img, (left, top), (right, bottom), (0, 0, 255), 2)
                

                #cv2.imshow('face', face_image)
                #cv2.waitKey(0)
                #cv2.destroyAllWindows()
                #cv2.imshow('face', face_image)
                #cv2.waitKey(0)
                #cv2.destroyAllWindows()


 
        if mirror: 
            img = cv2.flip(img, 1)
        cv2.imshow('my webcam', img)
        if cv2.waitKey(1) == 27: 
            break  # esc to quit
    cv2.destroyAllWindows()


def get_available_cameras():
    # try to get a list of all cameras in 5 seconds
    index = 0
    arr = []
    while True:
        cap = cv2.VideoCapture(index)
        print(cap.read())
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
    Test the ports and returns a tuple with the available ports 
    and the ones that are working.
    """
    is_working = True
    dev_port = 0
    working_ports = []
    available_ports = []
    while is_working:
        camera = cv2.VideoCapture(dev_port)
        if not camera.isOpened():
            is_working = False
            print("Port %s is not working." %dev_port)
        else:
            is_reading, img = camera.read()
            w = camera.get(3)
            h = camera.get(4)
            if is_reading:
                print("Port %s is working and reads images (%s x %s)" %(dev_port,h,w))
                working_ports.append(dev_port)
            else:
                print("Port %s for camera ( %s x %s) is present but does not reads." %(dev_port,h,w))
                available_ports.append(dev_port)
        dev_port +=1
    return available_ports,working_ports



def generate_face_encodings():
    print("Generating face encodings")
    cam = cv2.VideoCapture(1)
    while True:
        ret_val, img = cam.read()
        face_locations = fr.face_locations(img)
        face_encodings = fr.face_encodings(img, face_locations)

        for face_encoding in face_encodings:
            print(face_encoding)
            print("Face encoding generated")
            print("Size: {}".format(len(face_encoding)))
            print("Type: {}".format(type(face_encoding)))
            print("Shape: {}".format(face_encoding.shape))
            print("Dtype: {}".format(face_encoding.dtype))
            print("Itemsize: {}".format(face_encoding.itemsize))
            print("Nbytes: {}".format(face_encoding.nbytes))
            print("Strides: {}".format(face_encoding.strides))
            print("Flags: {}".format(face_encoding.flags))
            print("Contiguous: {}".format(face_encoding.flags.contiguous))
            print("C: {}".format(face_encoding.flags.c_contiguous))
            print("F: {}".format(face_encoding.flags.f_contiguous))
            print("O: {}".format(face_encoding.flags.owndata))
            print("W: {}".format(face_encoding.flags.writeable))
            print("A: {}".format(face_encoding.flags.aligned) )

        if cv2.waitKey(1) == 27:
            break
    cv2.destroyAllWindows()
    




def main():
    show_webcam()


if __name__ == '__main__':
    main()

