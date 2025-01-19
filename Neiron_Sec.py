import cv2
import pytesseract
import sqlite3
import time
from fuzzywuzzy import fuzz
import matplotlib.pyplot as plt
from twilio.rest import Client

#Укажите свой путь к pytesseract!
pytesseract.pytesseract.tesseract_cmd = r'path-to-tesseract'

#Сюда вставьте свои данные Twilio
account_sid = 'your_sid'
auth_token = 'your_token'
twilio_phone_number = 'your_twilio number'
destination_phone_number = 'destination_phone_number'

def create_db():
    conn = sqlite3.connect('car_plates.db')
    cursor = conn.cursor()
    cursor.execute('''
    CREATE TABLE IF NOT EXISTS plates (
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        plate_number TEXT UNIQUE NOT NULL
    )''')
    conn.commit()
    conn.close()

def add_plate_to_db(plate_number):
    conn = sqlite3.connect('car_plates.db')
    cursor = conn.cursor()
    try:
        cursor.execute('''
        INSERT INTO plates (plate_number) VALUES (?)
        ''', (plate_number,))
        conn.commit()
    except sqlite3.IntegrityError:
        pass
    conn.close()

def make_call(to_number):
    client = Client(account_sid, auth_token)

    call = client.calls.create(
        to=to_number,
        from_=twilio_phone_number,
        url='http://demo.twilio.com/docs/voice.xml'
    )

    print(f"Звонок с Twilio отправлен на номер: {to_number}")

def check_plate_in_db(plate_number):
    conn = sqlite3.connect('car_plates.db')
    cursor = conn.cursor()
    cursor.execute('SELECT plate_number FROM plates')
    all_plates = cursor.fetchall()

    plate_number = plate_number.strip().replace(" ", "").upper()
    print(f"Проверяем номер: {plate_number}")

    for row in all_plates:
        db_plate = row[0].strip().replace(" ", "").upper()
        print(f"Сравниваем с базой данных: ")
        
        similarity = fuzz.ratio(plate_number, db_plate)
        if similarity >= 70:
            print(f"Номер {plate_number} совпадает с {db_plate} на {similarity}%")
            make_call(destination_phone_number)
            return True

    return False

def open_img(img_path):
    """Открытие и отображение изображения"""
    carplate_img = cv2.imread(img_path)
    if carplate_img is None:
        print(f"Ошибка: не удалось загрузить изображение {img_path}")
        return None
    carplate_img = cv2.cvtColor(carplate_img, cv2.COLOR_BGR2RGB)
    plt.axis('off')
    plt.imshow(carplate_img)
    return carplate_img


def carplate_extract(image, carplate_haar_cascade):
    """Извлечение номерного знака с использованием Haar Cascade"""
    if image is None:
        return None

    carplate_rects = carplate_haar_cascade.detectMultiScale(image, scaleFactor=1.1, minNeighbors=5)

    if len(carplate_rects) == 0:
        print("Номерной знак не найден")
        return None

    for x, y, w, h in carplate_rects:
        carplate_img = image[y+15:y+h-10, x+15:x+w-20]

    return carplate_img


def enlarge_img(image, scale_percent):
    """Изменение размера изображения"""
    if image is None or image.size == 0:
        print("Ошибка: пустое изображение")
        return None

    width = int(image.shape[1] * scale_percent / 100)
    height = int(image.shape[0] * scale_percent / 100)
    dim = (width, height)
    resized_image = cv2.resize(image, dim, interpolation=cv2.INTER_AREA)
    return resized_image


def preprocess_img(image):
    """Предобработка изображения для улучшения качества OCR"""
    gray_image = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
    _, thresh_image = cv2.threshold(gray_image, 150, 255, cv2.THRESH_BINARY)
    return thresh_image


# Основная функция
def main():
    cap = cv2.VideoCapture(0)
    frame_count = 0

    #Вставьте сюда путь к haar cascad'у
    carplate_haar_cascade = cv2.CascadeClassifier(r'path_to_haar_cascade')

    if carplate_haar_cascade.empty():
        print("Ошибка загрузки каскада! Проверь путь к файлу XML.")
        exit()

    if not cap.isOpened():
        print("Не удается открыть камеру")
        exit()

    while True:
        ret, frame = cap.read()
        if not ret:
            print("Не удалось получить кадр")
            break
        
        frame_count += 1
        current_time = time.strftime("%Y-%m-%d %H:%M:%S")
        fps = cap.get(cv2.CAP_PROP_FPS)
        fps_text = f"FPS: {fps:.2f}" if fps != 0 else "FPS: Unknown"

        cv2.putText(frame, fps_text, (10, frame.shape[0] - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
        cv2.putText(frame, current_time, (10, frame.shape[0] - 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)

        if frame_count % 10 == 0:
            carplate_img_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

            carplate_extract_img = carplate_extract(carplate_img_rgb, carplate_haar_cascade)

            if carplate_extract_img is not None:
                carplate_extract_img = enlarge_img(carplate_extract_img, 150)

                if carplate_extract_img is not None:
                    carplate_extract_img_processed = preprocess_img(carplate_extract_img)

                    plate_number = pytesseract.image_to_string(
                        carplate_extract_img_processed,
                        config='--psm 6 --oem 3 -c tessedit_char_whitelist=ABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789'
                    )

                    plate_number = plate_number.strip().replace(" ", "").upper()
                    print(f"Извлечённый номер: {plate_number}")
                    
                    if check_plate_in_db(plate_number):
                        print("Номер найден в базе данных.")
                    else:
                        print("Номер не найден в базе данных.")
                else:
                    print("Не удалось изменить размер изображения номерного знака.")
            else:
                print('Номерной знак не найден')

        cv2.imshow("Camera Feed", frame)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()

if __name__ == '__main__':
    create_db()
    main()

    #Можете добавить сюда любые номера
    add_plate_to_db('C227HA69')
    add_plate_to_db('К777ОТ555')