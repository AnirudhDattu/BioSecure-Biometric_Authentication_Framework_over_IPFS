import cv2

# Test different indices and backends
for index in [0, 1]:
    for backend in [cv2.CAP_ANY, cv2.CAP_DSHOW]:
        cap = cv2.VideoCapture(index, backend)
        cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
        cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
        
        if not cap.isOpened():
            print(f"Camera index {index} with backend {backend} failed to open")
            continue
        
        ret, frame = cap.read()
        if ret and frame is not None:
            print(f"Camera index {index} with backend {backend} works! Frame shape: {frame.shape}")
            while True:
                ret, frame = cap.read()
                if not ret:
                    print("Failed to grab frame")
                    break
                cv2.imshow("Camera Test", frame)
                if cv2.waitKey(1) & 0xFF == ord('q'):
                    break
        else:
            print(f"Camera index {index} with backend {backend} opened but failed to grab frame")
        
        cap.release()
cv2.destroyAllWindows()