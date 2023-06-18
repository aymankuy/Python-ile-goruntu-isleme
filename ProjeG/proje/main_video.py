import cv2
from simple_facerec import SimpleFacerec

# klasörden resim çekme
sfr = SimpleFacerec()
sfr.load_encoding_images("images/")

# kamera açma
cap = cv2.VideoCapture(0)


while True:
    ret, frame = cap.read()

    # yüz tanıma işlemi yapılıyor
    face_locations, face_names = sfr.detect_known_faces(frame) # yüz tanıma sistemini çağırır
    for face_loc, name in zip(face_locations, face_names): # yüz ve isim için eşleşme yapan döngüsü
        y1, x2, y2, x1 = face_loc[0], face_loc[1], face_loc[2], face_loc[3] # faceloc yüzün koordinatlarını belirten komut

        cv2.putText(frame, name,(x1, y1 - 10), cv2.FONT_HERSHEY_DUPLEX, 1, (0, 0, 200), 2) # görüntüye isim yazma
        cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 0, 200), 4) # yüzü tanımlayan kutu şekli için kullanıldı

    cv2.imshow("Frame", frame) #görüntüyü ekranda oluşturur 

    key = cv2.waitKey(1)  # bir tuşa basıldığında pencere kapanır
    if key == 27:
        break

cap.release()
cv2.destroyAllWindows() # bütün pencerelerin kapanma komutu