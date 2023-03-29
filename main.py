import cv2
# Carga el clasificador de cascada Haar para detectar rostros
face_cascade = cv2.CascadeClassifier('haarcascade_frontalface_default.xml')

# Abre la cámara
cap = cv2.VideoCapture(0)

# Ciclo para procesar cada frame de la cámara
while True:
    # Lee un frame de la cámara
    ret, frame = cap.read()

    # Verifica si se leyó correctamente el frame
    if not ret:
        print("No se pudo leer el frame.")
        break

    # Convierte el frame a escala de grises
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    # Detecta las caras en el frame
    faces = face_cascade.detectMultiScale(
        gray, scaleFactor=1.3, minNeighbors=5)

    # Dibuja un cuadrado alrededor de cada cara detectada
    for (x, y, w, h) in faces:
        cv2.rectangle(frame, (x, y), (x+w, y+h), (0, 255, 0), 2)
    # Convierte el frame a escala de grises
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    # Aplica un filtro Gaussiano para reducir el ruido
    # Espera por la tecla 'q' para salir del ciclo
    gray = cv2.GaussianBlur(gray, (5, 5), 0)
    # Muestra el frame con los cuadrados alrededor de las caras detectadas
    cv2.imshow('VENTANA', frame)

    if cv2.waitKey(1) == ord('q'):
        break

# Libera la cámara y cierra todas las ventanas
cap.release()
cv2.destroyAllWindows()
