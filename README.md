# Cheat Detection Pipeline (Eye Gaze + Environment + Body Pose + RNN/LSTM + MIL)

โครงสร้างโค้ดนี้ตรงกับ 3 flow ในไดอะแกรม: **Inference**, **Training**, **Label by MIL**

## 1. ติดตั้ง

```bash
cd cheat_detection
python -m venv venv
source venv/bin/activate   # Windows: venv\Scripts\activate
pip install -r requirements.txt
```

## 2. โครงสร้างโปรเจกต์

```
cheat_detection/
├── config.py                  # path/hyperparameter ทั้งหมด แก้ที่นี่ที่เดียว
├── models/
│   ├── eye_gaze_model.py       # Eye Gaze Classification Model : MobileOne
│   ├── environment_model.py    # Environment Model : Yolo11s
│   ├── body_pose_model.py      # Body Pose Model
│   └── sequence_model.py       # RNN/LSTM + Classifier head + MIL Attention head
├── utils/
│   ├── preprocessing.py        # extract 30fps->5fps, crop head, resize env frame
│   ├── dataset.py               # โหลด Dataset/subjectX/cheat|Non_cheat/*.jpg (ตรงกับรูปที่ 2)
│   └── fusion.py                 # รวม embedding 3 branch -> เข้า RNN/LSTM
├── inference.py                 # (1) Inference
├── train.py                      # (2) Training
├── label_by_mil.py               # (3) Label by MIL
└── requirements.txt
```

วาง dataset ตามรูปที่ 2 ไว้ที่ `Dataset/subject1/cheat/`, `Dataset/subject1/Non_cheat/`, `Dataset/subject2/cheat/` ...

## 3. การใช้งานแต่ละส่วน

### (1) Inference — วิดีโอ 30fps -> เลือกเฟรม 5fps -> Model -> Prediction
```bash
python inference.py --video path/to/video.mp4
python inference.py --video path/to/video.mp4 --onnx   # ใช้โมเดล INT8 ONNX (เร็วกว่า ตามที่โน้ตในรูปแนะนำ)
```
ถ้าต้องการ export เป็น ONNX INT8 ก่อน ให้เรียก `export_onnx()` ใน `inference.py` (ต้องมี `checkpoints/sequence_rnn.pt` จากการเทรนก่อน)

### (2) Training — Labeled data (manual + MIL) -> 3 branch -> RNN/LSTM -> Prediction
```bash
python train.py
python train.py --epochs 50 --batch-size 4
python train.py --data-dir Dataset_MIL         # เทรนต่อด้วยชุดข้อมูลที่ label อัตโนมัติจาก MIL
python train.py --finetune-branches            # fine-tune backbone ของ 3 branch ด้วย (ค่า default จะ freeze ไว้)
```
โดย default จะ freeze น้ำหนักของ Eye Gaze / Environment / Body Pose ไว้ (ใช้เป็นแค่ feature extractor) แล้วเทรนเฉพาะ RNN/LSTM + classifier head ซึ่งเหมาะกับข้อมูลจำนวนน้อย

### (3) Label by MIL — Video ดิบ -> 3 branch -> RNN/LSTM -> MIL -> Labeled data
```bash
python label_by_mil.py --video raw_videos/subject3.mp4 --subject subject3
python label_by_mil.py --video-dir raw_videos/          # batch ทั้งโฟลเดอร์
```
ผลลัพธ์จะถูกเขียนออกมาเป็นโครงสร้างเดียวกับรูปที่ 2 ที่ `Dataset_MIL/subjectX/<cheat|Non_cheat>/subjectX_frameNNNN.jpg` โดยจะเก็บเฉพาะเฟรมที่ attention weight ของ MIL head สูงกว่า `--attn-threshold` (ลด label noise)

## หมายเหตุ

- ต้องมี weight ที่เทรนแล้วใน `checkpoints/` ก่อนถึงจะรัน `inference.py` หรือ `label_by_mil.py` ได้ผลลัพธ์ที่มีความหมาย — เริ่มจาก `train.py` ก่อนเสมอ
- `models/eye_gaze_model.py` ใช้ MobileOne จาก `timm`, `environment_model.py` ใช้ YOLO11s จาก `ultralytics` — ถ้ามี weight ที่ fine-tune เฉพาะทางของคุณเองแล้ว ให้ใส่ path ไว้ใน `config.py`
- `body_pose_model.py` ใช้ mediapipe pose landmarks เป็นตัวอย่าง สามารถเปลี่ยนเป็นโมเดล pose อื่นได้โดยแก้เฉพาะไฟล์นี้ (interface เดิม: `embed_batch(list_of_bgr_frames) -> [T, POSE_FEAT_DIM]`)
- ตัว face detector ใน `crop_head()` เป็น Haar cascade แบบง่าย ๆ ไว้ก่อน แนะนำเปลี่ยนเป็น mediapipe face detection หรือ RetinaFace สำหรับ production
