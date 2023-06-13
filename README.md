# Deep-Learning---Face-Classifier
Cnn ile Yüz tanıma

Projenin amacı yüz tanıma ve sınflandırma işlemidir.
Proje fonksiyonel programlama mantığı ile yapılmıştır. Pyhton diliyle ve kütüphaneleriyle çalıştım.
Kullandığım yöntemler ve algoritmalar şöyledir.

-Model eğitimi ve sınıflandırma için Convolutional Neural Network (CNN)
-Yüz tespiti için OpenCv’ nin sağladağı CascadeClassifier kullandım.
-Görüntü üzerinden “Person” yakalama işlemi yolo ile yapıldı.



Klasörler ve Py Dosyaları Hakkında Bilgi 
-TrainSet (Klasör): Bu klasör altında sınıflandırma yapacağımız kişilerin
(Kaggle:Golden Foot Football Image Dataset ) dataları mevcut. Birçok futbolcu
 var bu datalarda ancak ben kısa sürmesi için birkaç futbolcu ekledim. 
Siz isterseniz modeli baştan eğitip kendi datalarınızı ekleyebilirsiniz.
https://www.kaggle.com/datasets/balabaskar/golden-foot-football-players-image-dataset

-yolo_face.py: Bu python kodu verilen görüntü üzerinden nesne tespitini yolo ile yapar.
 Görüntülerin TrainSet içerisinde olması gerekmektedir. Kod içerisinde “detect_person” 
isimli bir method (fonksiyon)Vardır. Parametre olarak TrainSet altındaki klasör isimleri verilir.
 Fonksiyonun çalışıp elde ettiği görüntüleri kaydetmesi için “save” parametresi 1 olarak verilir. (Default 1) 
Örn: detect_person("luka_modric",1)

-faces(klasör): detect_person fonksiyonu çalıştırıldıktan sonra elde edilen görüntüler bu klasör altına yazılır.

-face_segmentation.py: face_detect adında iki parametreli bir fonksiyona sahiptir. Birinci parametre faces klasörü
 altında olan sadece “person” ile sınırlandırılmış görüntülerin klasör adları yer almalıdır. İkinci parametre save=1 default.
 Bu fonksion Frontal ve Profil Cascade classifier ile yüz koordinatlarını buluyor ve real_face klasörü altına kaydediliyor.

-real_face(klasör): Burdaki klasörler yüz koordinatlarını bulunup sadece yüz görüntülerinin kaydedildiği klasördür.

-mode_train.py: Bu parçada ise real_face altındaki tüm verileri okuyan ve Cnn algoritması ile sınıflandırma işlemi yapan python kodudur.
 Oluşturulan model anaklasör altında yeni.h5 olarak gelecektir. Oluşturulan modeli test klasörü altına almamız gerekiyor
 (İlerleyen dönemlerde bu gibi durumlar otomatik yapılacaktır. Şuanlık bir eksiklik maalesef). Model eğitiminde kullandığım
 yöntemleri görmek için açıp inceleyebilirsiniz.
Model eğitimi yapmak istemiyorsanız zaten hali hazırda ben test klasörü altına bu modeli ekledim.

-test(klasör): Buraya yeni.h5 modelimizi atıyoruz ve eğitimde kullanmadığımız dışarıdan görsellerimizi atıyoruz istediğimiz kadar. 

-test.py: Test klasörü altındaki jpg uzantılı dosyaları okuyarak modele göndermeden önce belli başlı ön işlemlere tabi tutar
 Burda yine cascade classifierlar yardımıyla yüz tespiti yapılır ancak herhangi bir kayıt işlemi gerçekleşmez. Döngü için de her
 görüntü için bu işlem yapılır ve modelle test edilir. Tespit edilen kişiler bir kare içerine alınır ve kimlikleri yazılır.

-data ve dataset (klasör): Bu klasörler hazır yolo kullanımı için eklenmiştir. İçerisinde coco_classes yani nesne tespiti yaptığı
 nesnelerin labelleri vardır.

Ekstra Bilgi: real_face altındaki görüntüler eşit sayıda olmadığı ve bazı fotoğrafların tam yüz olarak seçilememinden dolayı birkaç
görüntü manuel olarak silinmiştir. İlerleyen dönemler de bunları düzelteceğim.

Eksiklikler-Farkındalıklar: Şu aşamada test işlemi sadece fotoğraf olarak çalışıyor ve sınıflandırıyor. Model eğitimine dahil olmayan 
yüzleri tespit ettiği zaman yanlış sınıflandırıyor. Bunları gidereceğim ve video ile sınıflandırma ve DeepSORT ile takip işlemlerini aktif edeceğim.

Proje büyük boyutta olduğu için rar dosyasını indirmeniz gerekiyor. İsterseniz indirmeden sadece kod kısmını da inceleyebilirsiniz.
