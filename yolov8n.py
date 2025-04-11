import os
import streamlit as st
from PIL import Image
import tempfile
import numpy as np
from ultralytics import YOLO
import torch
import cv2
from ultralytics.nn.tasks import DetectionModel

torch.serialization.add_safe_globals([DetectionModel])

st.set_page_config(page_title="이미지 자동 크롭기", layout="centered")
st.title("📐 이미지 자동 크롭 + 분할 (YOLO 기반)")

# SNS 비율 옵션
target_ratios = {
    "인스타그램 - 정사각형 (1:1)": (1, 1),
    "인스타그램 - 세로 (4:5)": (4, 5),
    "인스타그램 - 가로 (1.91:1)": (1.91, 1),
    "인스타그램 - 릴스/스토리 (9:16)": (9, 16),
    "페이스북 - 피드 이미지 (1.91:1)": (1.91, 1),
    "페이스북 - 정사각형 (1:1)": (1, 1),
    "페이스북 - 스토리 (9:16)": (9, 16),
    "페이스북 - 이벤트 커버 (16:9)": (16, 9)
}

uploaded_file = st.file_uploader("이미지 파일 업로드 (JPG, PNG)", type=["jpg", "jpeg", "png"])

if uploaded_file is not None:
    with tempfile.TemporaryDirectory() as temp_dir:
        file_path = os.path.join(temp_dir, uploaded_file.name)
        with open(file_path, "wb") as f:
            f.write(uploaded_file.read())

        try:
            img = Image.open(file_path).convert("RGB")
            img_width, img_height = img.size
            st.success(f"✅ 현재 이미지 해상도: {img_width} x {img_height} px")

            selected_ratio_label = st.selectbox("📱 크롭할 SNS 비율을 선택하세요:", list(target_ratios.keys()))
            ratio_w, ratio_h = target_ratios[selected_ratio_label]

            crop_width = st.number_input("크롭할 가로 픽셀 수 입력", min_value=100, max_value=img_width, value=1080, step=10)
            crop_height = int(crop_width * (ratio_h / ratio_w))
            st.markdown(f"🔍 자동 계산된 세로 크기: **{crop_height}px** (비율 {ratio_w}:{ratio_h})")

            if st.button("✂️ 크롭 하기 (YOLO 자동 객체 중심)"):
                model = YOLO("yolov8n.pt")  # 신뢰된 사전학습 모델 사용 가정
                results = model(file_path)

                if not results or results[0].boxes is None or len(results[0].boxes) == 0:
                    st.warning("객체를 인식하지 못했습니다. 이미지 중앙을 기준으로 크롭합니다.")
                    center_x, center_y = img_width // 2, img_height // 2
                else:
                    boxes = results[0].boxes.xyxy.cpu().numpy()
                    areas = [(x2 - x1) * (y2 - y1) for x1, y1, x2, y2 in boxes]
                    biggest = boxes[np.argmax(areas)]
                    x1, y1, x2, y2 = biggest
                    center_x = int((x1 + x2) / 2)
                    center_y = int((y1 + y2) / 2)

                left = max(0, center_x - crop_width // 2)
                upper = max(0, center_y - crop_height // 2)
                right = min(img_width, left + crop_width)
                lower = min(img_height, upper + crop_height)

                cropped = img.crop((left, upper, right, lower))
                output_path = os.path.join(temp_dir, f"cropped_{uploaded_file.name}")
                cropped.save(output_path, quality=100, subsampling=0)

                st.image(cropped, caption="크롭된 이미지 (YOLO 인식 중심)", use_column_width=True)
                with open(output_path, "rb") as f:
                    st.download_button(
                        label="📥 크롭 이미지 다운로드",
                        data=f,
                        file_name=f"cropped_{uploaded_file.name}",
                        mime="image/jpeg"
                    )

        except Exception as e:
            st.error(f"이미지 처리 중 오류 발생: {e}")
