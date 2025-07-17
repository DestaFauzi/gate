import cv2
import pytesseract

def capture_and_detect():
    camera = cv2.VideoCapture(0)
    ret, frame = camera.read()
    cv2.imwrite('plate.jpg', frame)
    camera.release()
    
    image = cv2.imread('plate.jpg')
    plate_text = pytesseract.image_to_string(image, config='--psm 8')
    return plate_text.strip()

if __name__ == "__main__":
    plate = capture_and_detect()
    print(f"Detected Plate: {plate}")