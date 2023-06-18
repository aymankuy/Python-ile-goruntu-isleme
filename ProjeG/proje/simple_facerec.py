import face_recognition
import cv2
import os
import glob
import numpy as np # gerekli kütüphaneleri import ettik

class SimpleFacerec:
    def __init__(self):
        self.known_face_encodings = [] # bilinen yüzlerin kodlamasını içerir
        self.known_face_names = [] #bilinen yüzlerin isimlerini içerir

        # görüntüyü boyutlandırma işlemi 
        self.frame_resizing = 0.25

    def load_encoding_images(self, images_path): #bu dizinde görüntüler, isimler çekilir ve ilgili listeye eklenir 
        """
        Load encoding images from path
        :param images_path:
        :return:
        """ 
        # görüntü yükleme 
        images_path = glob.glob(os.path.join(images_path, "*.*"))

        print("{} encoding images found.".format(len(images_path)))

        # görüntü ve isimlerin saklanması işlemi
        for img_path in images_path:
            img = cv2.imread(img_path)
            rgb_img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

            # başlangıç aşamasından sadece dosya adı çekilir
            basename = os.path.basename(img_path)
            (filename, ext) = os.path.splitext(basename)
           
            img_encoding = face_recognition.face_encodings(rgb_img)[0]

           # 1.satırdaki kod 2.satırdakine görüntü kodlamasını ekler 
            self.known_face_encodings.append(img_encoding) 
            self.known_face_names.append(filename)
        print("Encoding images loaded")

    def detect_known_faces(self, frame):
        small_frame = cv2.resize(frame, (0, 0), fx=self.frame_resizing, fy=self.frame_resizing)
        # mevcut video karesindeki tüm yüzleri ve kodlamalarını bulmak için yazılır
        # renk formatına dönüştürme işlemi 
        rgb_small_frame = cv2.cvtColor(small_frame, cv2.COLOR_BGR2RGB)
        face_locations = face_recognition.face_locations(rgb_small_frame)
        face_encodings = face_recognition.face_encodings(rgb_small_frame, face_locations)

        face_names = []
        for face_encoding in face_encodings:
            # yüzlerin eşleşip eşleşmediğini kontrol et 
            matches = face_recognition.compare_faces(self.known_face_encodings, face_encoding)
            name = "Unknown" #eşleşmediği durumda ekrana unknown yaz 

    # bu kod parçası, bir yüz kodlamasının bilinen yüz kodlamaları ile karşılaştırılarak 
    # en iyi eşleşmenin bulunmasını ve ilgili yüz için bir isim belirlenmesini sağlar.
            face_distances = face_recognition.face_distance(self.known_face_encodings, face_encoding) 
            # yüz kodlamaları arası mesafeyi hesaplar ve kaydeder
            best_match_index = np.argmin(face_distances) #en iy eşleşen yüzün konumunu belirtir
            if matches[best_match_index]: #değerin True olup olmadığı kontrol edilir
                name = self.known_face_names[best_match_index] #eşleşen en iyi yüzün ismini ifade eder
            face_names.append(name) # elde edilen değeri listeye ekler. bu işlem eklenen her yüz için devam eder sonuçları saklar

        # hızlı bir şekilde çerçeveyi yeniden boyutlandırmak için koordinatları düzenlemek için numpy dizisine dönüştürür 
        face_locations = np.array(face_locations)
        face_locations = face_locations / self.frame_resizing
        return face_locations.astype(int), face_names
