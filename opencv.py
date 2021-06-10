#libreria opencv para deteccion de autos
import cv2

#Captura de marcos del video
camara = cv2.VideoCapture('/Users/oscarfrancisco/Downloads/Car-detection-master/video.avi')
camara.open('/Users/oscarfrancisco/Downloads/Car-detection-master/video.avi')
#Xml entrenado para la deteccion de autos
car_cascade = cv2.CascadeClassifier('/Users/oscarfrancisco/Documents/cars.xml')

#Loop cuando el ciclo haya inicializado
while True:
    #Se leen los marcos del video
    (grabbed,frame) = camara.read()
    #Se convierte el video a escala de grises
    video_gris = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    #Se detectan los autos de diferentes medidas en cualquier parte del video
    autos = car_cascade.detectMultiScale(video_gris, 1.1, 1)
    for (x,y,w,h) in autos:
        #Se  dibujan los rectangulos alrededor de los autos en formato rgb
         cv2.rectangle(frame,(x,y),(x+w,y+h),(0,255,0),2)
         cv2.imshow("video",frame)
    #Se despliga ventana en pantalla y se cierra presionando Q o automaticamente al finalizar el video
    if cv2.waitKey(1)== ord('q'):
        break
camara.release()
cv2.destroyAllWindows()

