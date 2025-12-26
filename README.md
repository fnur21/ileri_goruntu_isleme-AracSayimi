# 🚗 YOLOv8 ile Gerçek Zamanlı Araç Tespiti ve Sayımı

Bu proje, **İleri Görüntü İşleme** dersi kapsamında geliştirilmiş bir nesne tespiti uygulamasıdır. Derin öğrenme tabanlı **YOLOv8 (You Only Look Once)** algoritması kullanılarak; araba, otobüs, kamyon ve motosiklet gibi araç türleri tespit edilmekte ve sayılmaktadır.

Proje, kullanıcıların kolayca video veya resim yükleyip sonuçları görebileceği **Flask** tabanlı, kullanıcı dostu bir web arayüzüne sahiptir.

---

## 🖥️ Proje Arayüzü

Aşağıda, geliştirilen web arayüzünün ve tespit işleminin bir örneği görülmektedir:
![Demo Görseli](./2025-12-23%20174758.png)



## ✨ Özellikler

* **Gerçek Zamanlı Tespit:** YOLOv8'in hızı sayesinde videolarda yüksek FPS ile tespit.
* **Çoklu Sınıf Tespiti:** 4 farklı araç kategorisini tanıma:
    * 🚗 Araba (Car)
    * 🚌 Otobüs (Bus)
    * 🚛 Kamyon (Truck)
    * 🏍️ Motosiklet (Motorcycle)
* **Web Tabanlı Kullanım:** Flask ile geliştirilmiş modern ve sade arayüz.
* **GPU Hızlandırma:** Eğitim ve tespit süreçleri NVIDIA GPU (CUDA) desteği ile optimize edilmiştir.

---

## 🛠️ Kullanılan Teknolojiler

* **Dil:** Python 3.x
* **Model:** [YOLOv8 (Ultralytics)](https://github.com/ultralytics/ultralytics)
* **Web Çerçevesi:** Flask
* **Görüntü İşleme:** OpenCV
* **Derin Öğrenme Kütüphanesi:** PyTorch
* **Ön Yüz:** HTML5, CSS3, (Bootstrap kullanılabilir)

---

## 🧠 Model Eğitimi ve Veri Seti

* **Model:** Projede hız ve performans dengesi için **YOLOv8 Nano (yolov8n)** modeli tercih edilmiştir.
* **Veri Seti:** Özel olarak toplanmış ve etiketlenmiş binlerce araç görüntüsü kullanılmıştır (Roboflow formatında düzenlenmiştir).
* **Eğitim Süreci:** Model, GPU üzerinde **100 epoch** boyunca eğitilmiştir. Eğitim parametreleri ve veri yolları `data.yaml` dosyasında, eğitim betiği ise `train.py` dosyasında bulunmaktadır.

*(İsteğe bağlı: Buraya eğitim sonucunda elde edilen mAP50 başarı grafiğini (results.png) de ekleyebilirsiniz)*

---

## 🚀 Kurulum ve Çalıştırma

Projeyi kendi bilgisayarınızda çalıştırmak için aşağıdaki adımları izleyin:

**1. Projeyi Klonlayın:**
```bash
git clone [https://github.com/fnur21/ileri_goruntu_isleme-AracSayimi.git](https://github.com/fnur21/ileri_goruntu_isleme-AracSayimi.git)
cd ileri_goruntu_isleme-AracSayimi
