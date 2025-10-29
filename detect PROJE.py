import cv2
from sahi import AutoDetectionModel
from sahi.predict import get_sliced_prediction  #videoyu dilimleyerek tespiti yapan ana fonksiyon

detection_model = AutoDetectionModel.from_pretrained(
    model_type='yolov11',
    model_path="yolo11n.pt",
    confidence_threshold=0.3,  #Emin olma skorunu %30 yaptım.
    device="cpu"  # cpu ya da gpu kullanmayı belirtiriz. gpu daha hızlı çalışabilir.
)

video_yolu = "C:/Users/reybe/Desktop/staj/YOLOv8 foto/sokaktaki insanlar.mp4"
cap = cv2.VideoCapture(video_yolu) #OpenCV'ye video dosyasını açmasını söyledik.
Pencere_genisligi = 1280

while True:
    ret, frame = cap.read()
    #eğer ret false olursa(video biterse ya da dosya bozuksa) video oynatmasını durdur.
    if not ret:
        break

    #Orijinal kareyi(frame'i)alır, detection_model'i kullanarak dilimler halinde işler.
    result = get_sliced_prediction(
        frame,
        detection_model,#Yukarıda SAHI için belirttiğimiz model.
        slice_height=640,
        slice_width=640,  #Görüntüyü 640 piksellik dilimlere ayırdık. genişik ve yükseklik olarak.
        overlap_height_ratio=0.2,  #Dilimlerin kenarda nesne kaçırmaması için %20 üst üste binmesini sağlar.
        overlap_width_ratio=0.2 
    )

    #SAHI'nin result içinde döndürdüğü tüm tespit edilen nesnelerin listesinde döngüye girelim.
    for pred in result.object_prediction_list:
        # Eğer bulunan nesnenin sınıfı person değilse bu nesneyi atlayacak.
        if pred.category.name != "person":
            continue
        #person olarak tespit edilen nesnenin kutu koordinatlarını alalım.
        x1, y1, x2, y2 = int(pred.bbox.minx), int(pred.bbox.miny), int(pred.bbox.maxx), int(pred.bbox.maxy)
        
        #OpenCV'ye, dikdörtgen çizmesini söyledik.
        cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
        
        #OpenCV'ye, sınıfın adını(insan olacak)ve güven skorunu yazmasını söyledik.
        cv2.putText(frame, f"{pred.category.name} {pred.score.value:.2f}", (x1, y1 - 10), 
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)

    (h, w) = frame.shape[:2] #Üzerine çizim yapılmış karenin mevcut yüksekliğini (h) ve genişliğini (w) aldık.
    r = Pencere_genisligi / float(w) #Görüntüyü Pencere_genisligi'ne (1280px) sığdırmak için küçültme oranını hesapladık.
    boyutlandırma = (Pencere_genisligi, int(h * r)) #En-boy oranını koruyarak yeni boyutları hesapladık.
    frame_resized = cv2.resize(frame, boyutlandırma , interpolation=cv2.INTER_AREA)  #kareyi hesaplanan yeni boyutlara dim'e göre yeniden boyutlandır

    cv2.imshow("SAHI ile YOLO (Başarılı)", frame_resized) #Yeniden boyutlandırdığımız karesyi pencerede göster dedik.


    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release() #Döngü bittiğinde, cap tarafından açılan video dosyasını kapatıyoruz.
cv2.destroyAllWindows()