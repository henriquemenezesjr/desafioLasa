import numpy as np
import cv2

#Iniciando o Classificador Haar Cascade
face_cascade = cv2.CascadeClassifier('haarcascade_frontalface_default.xml')

#Capturando o vídeo da Webcam
video_webcam = cv2.VideoCapture(0)

while True:
    #Capturando frame por frame do vídeo
    ret, frame = video_webcam.read()

    #Convertendo o frame de RGB para escala em cinza
    imagem_cinza = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    #Utilizando o classificador no frame 
    rostos = face_cascade.detectMultiScale(imagem_cinza, 1.3, 5)

    #Desenhando retângulos onde foram localizados os rostos
    for (x, y, w, h) in rostos:
        cv2.rectangle(frame, (x, y), (x+w, y+h), (255, 0, 0), 2)

    #Exibindo o frame de resultado
    cv2.imshow('Video', frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

video_webcam.release()
cv2.destroyAllWindows()
