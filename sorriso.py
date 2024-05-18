import cv2

# Carregar os classificadores pré-treinados para detecção de rostos e sorrisos
face_cascade_path = "haarcascade_frontalface_default.xml"
smile_cascade_path = "haarcascade_smile.xml"

# Carregar os classificadores Haar Cascade
face_cascade = cv2.CascadeClassifier(face_cascade_path)
smile_cascade = cv2.CascadeClassifier(smile_cascade_path)

# Função para detectar rostos e sorrisos em tempo real na câmera
def detect_smile_camera():
    # Inicializar a captura de vídeo
    video_capture = cv2.VideoCapture(0)

    while True:
        # Capturar o quadro da câmera
        ret, frame = video_capture.read()
        
        # Verificar se o quadro foi capturado corretamente
        if not ret:
            continue

        # Converter o quadro para escala de cinza para melhorar a detecção
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

        # Detectar rostos na imagem
        faces = face_cascade.detectMultiScale(gray, scaleFactor=1.3, minNeighbors=5)
        
        # Para cada rosto detectado, verificar se há um sorriso
        for (x, y, w, h) in faces:
            roi_gray = gray[y:y+h, x:x+w]
            roi_color = frame[y:y+h, x:x+w]

            # Detectar sorrisos na região do rosto
            smiles = smile_cascade.detectMultiScale(
                roi_gray, 
                scaleFactor=1.1, 
                minNeighbors=15, 
                minSize=(8, 8)
            )

            # Depuração: imprimir o número de sorrisos detectados
            print(f"Rostos detectados: {len(faces)}, Sorrisos detectados: {len(smiles)}")

            # Se um sorriso for detectado, exibir uma mensagem indicando que está sorrindo
            if len(smiles) > 0:
                cv2.putText(frame, 'Sorrindo', (x, y-10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 255, 0), 2)
                cv2.rectangle(frame, (x, y), (x+w, y+h), (0, 255, 0), 2)  # Verde para sorriso
            else:
                cv2.rectangle(frame, (x, y), (x+w, y+h), (255, 0, 0), 2)  # Azul para sem sorriso

        # Exibir o quadro com as detecções
        cv2.imshow('Sorriso Detector', frame)

        # Pressione 'q' para sair do loop
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    # Liberar a captura de vídeo
    video_capture.release()
    cv2.destroyAllWindows()

# Chamar a função para detectar sorrisos na câmera em tempo real
detect_smile_camera()