import os
from ultralytics import YOLO
import torch

if __name__ == '__main__':
    # --- Donanım Kontrolü ---
    # Eğer GPU (Ekran Kartı) yoksa işlemi durduruyorum, CPU ile vakit kaybetmek istemiyorum.
    if not torch.cuda.is_available():
        raise EnvironmentError("HATA: GPU bulunamadı! İşlem CPU üzerinden yapılmak istenmiyor.")

    # Mevcut GPU'nun adını alıp konsola yazdırıyorum.
    device_name = torch.cuda.get_device_name(0)
    print(f"Eğitim şu donanım üzerinde başlatılıyor: {device_name}")

    # --- Model Hazırlığı ---
    # YOLOv8 nano modelini başlangıç ağırlıklarıyla yüklüyorum.
    model = YOLO('yolov8n.pt')

    # --- Eğitim Süreci ---
    print("Eğitim parametreleri yüklendi, işlem başlıyor...")

    model.train(
        data='data.yaml',  # Veri seti konfigürasyon dosyası
        epochs=100,  # Maksimum 100 epoch (Erken durdurma aktif olabilir)
        imgsz=640,  # Standart giriş görüntü boyutu
        batch=16,  # GPU belleğine uygun batch boyutu
        device=0,  # Birincil ekran kartını kullan
        workers=2,  # Veri yükleme işlemi için iş parçacığı sayısı
        project='runs/detect',  # Sonuçların kaydedileceği ana dizin
        name='arac_modelim',  # Bu eğitimin özel klasör adı
        exist_ok=True,  # Klasör varsa üzerine yazmaya izin ver
        verbose=True  # Eğitim detaylarını konsolda göster
    )