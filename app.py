import gradio as gr
import cv2
from ultralytics import YOLO
import numpy as np
import rules

model = YOLO('E:/Yolov8/Nien_luan1/runs/detect/weights/last.pt')

# Vẽ tên lớp và độ chính xác lên ảnh
def draw_class_text(image, class_name, confidence, x1, y1, y2):
    text = f"{class_name} {confidence}%"
    text_width, text_height = cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, fontScale=0.5, thickness=2)[0] #chỉ lấy phần kích cỡ của văn bảng(width,height), không lấy khung văn bảng (x,y,w,h)
    text_position_x = x1 + 10
    text_position_y = y1 - 5
    # Kiểm tra tọa độ góc trên bên phải của văn bản
    x = x1 + text_width + 5
    y = y1 - text_height - 10
    if y < 0:
        text_position_y = y2 + text_height + 5
    elif x > image.shape[1]: #shape : height width
        text_position_x = x1 - (x - image.shape[1]) # đoạn cần dời ảnh
    elif y < 0 and x > image.shape[1]:
        text_position_x = x1 - (x - image.shape[1])
        text_position_y = y2 + text_height + 5

    image = cv2.putText(image, text, (text_position_x, text_position_y), cv2.FONT_HERSHEY_SIMPLEX,
                        fontScale=0.5, color=(255, 0, 0), thickness=2)
    return image

# Hàm dự đoán với ảnh
def predict_image(image):
    image = image.astype(np.uint8)
    results = model.predict(source=image, imgsz=640, conf=0.6)
    predicted_image = results[0].orig_img
    text_furniture = ""
    # Kiểm tra xem có hộp giới hạn được phát hiện trong ảnh hay không
    if not results[0].boxes:
        return image, text_furniture

    names = set()
    #Vẽ hộp giới hạn của từng đối tượng đươc dự đoán
    for bbox, cls, conf in zip(results[0].boxes.xyxy.tolist(), results[0].boxes.cls.tolist(),
                               results[0].boxes.conf.tolist()):
        # Lấy tọa độ của hộp giới hạn
        x1, y1, x2, y2 = map(int, bbox[:4])
        # Vẽ hộp giới hạn lên ảnh
        predicted_image = cv2.rectangle(predicted_image, (x1, y1), (x2, y2), color=(0,0,0), thickness=2)
        # Lấy tên lớp và độ chính xác
        class_name = model.names[int(cls)]
        confidence = round(conf * 100, 2)
        # Thêm tên của đối tượng vào lnames
        names.add(class_name)
        # Vẽ tên lớp và độ chính xác lên ảnh
        predicted_image = draw_class_text(predicted_image, class_name, confidence, x1, y1, y2)

    names_list = list(names)
    for i in range(0, len(names_list)-1):
        text_furniture += names_list[i] + ", "
    text_furniture += names_list[len(names_list) - 1]
    return predicted_image, text_furniture

# Hàm dự đoán với video
def predict_video(video):
    cap = cv2.VideoCapture(video)
    # Kiểm tra nếu video được mở thành công
    if not cap.isOpened():
        print("Không thể mở video")
        return

    # Lấy kích thước của video
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = cap.get(cv2.CAP_PROP_FPS)

    # Tạo video writer để lưu video dự đoán
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out = cv2.VideoWriter('predicted_video.mp4', fourcc, fps, (width, height))

    count_frame = 0
    skip_frame = 15
    detail = ""
    furnitures = set()
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:  # Kiểm tra xem khung hình có được đọc thành công không
            break
        # Dự đoán trên từng khung hình và trả về ảnh đã dự đoán
        predicted_frame, text = predict_image(frame)

        #lưu lại tất cả các đồ nội thất có trong video
        temp = text.split(", ")
        furnitures.update(temp)

        # Ghi khung hình dự đoán vào file video
        out.write(predicted_frame)
        if count_frame % skip_frame == 0:
            # Lấy thời gian tính bằng mili giây của khung hình hiện tại
            current_time_msec = cap.get(cv2.CAP_PROP_POS_MSEC)
            detail += "Giây thứ " + str(round(current_time_msec/1000,2)) + ": "
            detail += text + chr(10)
        count_frame += 1

    #chuyển furnitures về chuỗi
    result = rules.apply_rules(furnitures)

    # Giải phóng video writer và cửa sổ hiển thị
    cap.release()
    out.release()
    cv2.destroyAllWindows()

    # Trả về video đã dự đoán
    return "predicted_video.mp4", detail, result

demo_image = gr.Interface(
    fn=predict_image,
    inputs=gr.Image(label="Tải ảnh lên"),
    # outputs="image",
    outputs=[gr.Image(label="Ảnh đã dự đoán"), gr.Textbox(label="Các đồ nội thất: ")],
    title="Phát hiện đồ nội thất với YOLOv8",
    description="Các đồ nội thất sẽ được phát hiện."
)

demo_video = gr.Interface(
    fn=predict_video,
    inputs=gr.Video(label="Tải video lên"),
    outputs=[gr.File(label="Video đã dự đoán"),gr.Textbox(label="Chi tiết dự đoán"), gr.Textbox(label="Luật sinh")],
    title="Phát hiện đồ nội thất với YOLOv8",
    description="Tải video lên và các đồ nội thất trong video sẽ được phát hiện."
)

# demo_image.launch(share=True)
demo_video.launch(share=True)
