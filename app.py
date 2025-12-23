import os
import cv2
import datetime
import time
from flask import Flask, render_template, Response, jsonify, request, redirect, url_for
from ultralytics import YOLO
from werkzeug.utils import secure_filename

app = Flask(__name__)

# --- DOSYA YOLLARI  ---
BASE_DIR = os.path.abspath(os.path.dirname(__file__))
UPLOAD_FOLDER = os.path.join(BASE_DIR, 'static', 'uploads')
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER
os.makedirs(UPLOAD_FOLDER, exist_ok=True)


model_path = os.path.join(BASE_DIR, 'runs', 'detect', 'train', 'weights', 'best.pt')

if os.path.exists(model_path):
    print(f"✅ Özel model yüklendi: {model_path}")
    model = YOLO(model_path)
else:
    print(f"⚠️ DİKKAT: best.pt bulunamadı! Standart yolov8n.pt kullanılıyor.")
    model = YOLO('yolov8n.pt')


counts = {'araba': 0, 'otobus': 0, 'kamyon': 0, 'motor': 0}
logs = []
counted_ids = set()
loop_video = False
current_source = None


CLASS_NAMES = {0: 'otobus', 1: 'araba', 2: 'motor', 3: 'kamyon'}


def generate_frames():
    global counts, logs, counted_ids, current_source

    if not current_source:
        return

    cap = cv2.VideoCapture(current_source)

    if not cap.isOpened():
        print("❌ HATA: Video dosyası okunamadı.")
        return

    while True:
        success, frame = cap.read()
        if not success:
            if loop_video:
                cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
                # Döngüde aynı araçları tekrar saymamak için ID'leri temizlemiyoruz.
                # Eğer her döngüde sıfırdan saysın istersen: counted_ids.clear()
                continue
            else:
                break

        # --- GÖRÜNTÜ İŞLEME ---
        h, w, _ = frame.shape
        line_y = int(h * 0.65)  # Çizgi ekranın %65 aşağısında

        # Takip (Tracking) Modu
        results = model.track(frame, persist=True, verbose=False, classes=[0, 1, 2, 3])

     # sanal çizgi
        cv2.line(frame, (0, line_y), (w, line_y), (0, 255, 255), 2)

        if results[0].boxes.id is not None:
            boxes = results[0].boxes.xyxy.cpu().numpy()
            track_ids = results[0].boxes.id.int().cpu().tolist()
            cls_ids = results[0].boxes.cls.int().cpu().tolist()

            for box, track_id, cls_id in zip(boxes, track_ids, cls_ids):
                x1, y1, x2, y2 = map(int, box)
                label = CLASS_NAMES.get(cls_id, 'bilinmeyen')

                # Aracın merkezi
                cy = int((y1 + y2) / 2)
                cx = int((x1 + x2) / 2)

                # --- SAYIM MANTIĞI ---
                # Araç çizgiye değiyor mu? (+- 15 piksel tolerans)
                if line_y - 15 < cy < line_y + 15:
                    # Daha önce bu ID sayıldı mı?
                    if track_id not in counted_ids:
                        counted_ids.add(track_id)  # Artık sayıldı işaretle

                        if label in counts:
                            counts[label] += 1
                            # Loglara ekle
                            timestamp = datetime.datetime.now().strftime("%H:%M:%S")
                            # Log listesinin en başına ekle (yeni gelen üstte dursun)
                            logs.insert(0, {'time': timestamp, 'type': label, 'id': track_id})

                # Kutu Çizimi
                color = (0, 255, 0) if track_id in counted_ids else (100, 100, 255)
                cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
                cv2.putText(frame, f"{label} ID:{track_id}", (x1, y1 - 10),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)
                cv2.circle(frame, (cx, cy), 4, (0, 0, 255), -1)

        ret, buffer = cv2.imencode('.jpg', frame)
        frame_bytes = buffer.tobytes()
        yield (b'--frame\r\n' b'Content-Type: image/jpeg\r\n\r\n' + frame_bytes + b'\r\n')


# --- ROTALAR (Senin HTML Dosyanın İstediği Adresler) ---

@app.route('/')
def index():
    return render_template('index.html')


@app.route('/video_feed')
def video_feed():
    return Response(generate_frames(), mimetype='multipart/x-mixed-replace; boundary=frame')


# HTML'deki fetch('/data') buraya istek atıyor
@app.route('/data')
def data():
    return jsonify({'counts': counts, 'logs': logs})


# HTML'deki fetch('/settings') buraya istek atıyor
@app.route('/settings', methods=['POST'])
def settings():
    global loop_video
    data = request.json
    loop_video = data.get('loop', False)
    print(f"🔄 Döngü Ayarı Değişti: {loop_video}")
    return jsonify({'status': 'success', 'loop': loop_video})


# HTML'deki form action="/upload" buraya dosya yolluyor
@app.route('/upload', methods=['POST'])
def upload():
    global current_source, counted_ids, counts, logs

    if 'file' not in request.files:
        return redirect(url_for('index'))

    file = request.files['file']
    if file.filename == '':
        return redirect(url_for('index'))

    if file:
        filename = secure_filename(file.filename)
        path = os.path.join(app.config['UPLOAD_FOLDER'], filename)
        file.save(path)

        print(f"📂 Yeni dosya yüklendi: {path}")

        # Sistemi Sıfırla
        current_source = path
        counted_ids.clear()
        counts = {'araba': 0, 'otobus': 0, 'kamyon': 0, 'motor': 0}
        logs = []

    return redirect(url_for('index'))


if __name__ == "__main__":
    # Host='0.0.0.0' yaparak yerel ağda da erişilebilir yaptık (Opsiyonel)
    app.run(debug=True, port=5000)