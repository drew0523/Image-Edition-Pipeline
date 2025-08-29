# Image-Edition-Pipeline
image edit: deepfake using face detection, stable diffusion api



# Environment Settings
```
git clone https://github.com/drew0523/Image-Edition-Pipeline.git
conda create -n myenv
pip install -r requirements.txt
$env:GOOGLE_GENAI_API_KEY = "YOUR API KEY"
```

# Folder Composition
```
#onnx_weights, input_images, input_images_bg 폴더만 따로 생성 나머지는 자동으로 생성됨

project-root/
├── onnx_weights/               # ONNX 모델 파일 저장
│   └── segface_swinv2.onnx
├── input_images/               # 입력 이미지(대상) 폴더
│   └── example1.jpg
├── input_images_bg/            # 입력 이미지(배경 대상) 폴더
│   └── example_bg1.jpg
├── onnx_outputs/               # ONNX 추론 결과
│   └── mask_results/
│       └── swinv2_base_512/
│           └── onnx_preds/     # 추론 마스크 PNG 결과
├── results_rgba/               # 원본+마스크 합성 RGBA 결과
├── final_results/              # InsightFace 기반 합성 결과(1차 합성)
├── geminai_results/            # Gemini API 후편집 결과
├── infer_segface_onnx.py       # ONNX 추론 스크립트
├── mask_rgba.py                # 마스크→RGBA 합성 모듈
├── overlay.py                  # InsightFace 오버레이 모듈
├── picture_edit.py             # Gemini API 이미지 편집 모듈
├── run_pipeline.py             # 전체 이미지 합성 파이프라인 실행
└── requirements.txt            # pip 의존성 리스트
```

# Run
```
python run_pipeline.py --image1 '{합성하고 싶은 대상 이미지}' --image2 '{합성하고 싶은 배경 이미지}'

# ex)
python run_pipeline.py --image1 './input_images/face (5048).jpeg' --image2 '.\input_images_bg\astro.png'
```

# Demo

<p align="center">
  <figure style="display:inline-block; text-align:center; margin:10px;">
    <img src="https://github.com/user-attachments/assets/4021d75d-658c-463a-9f23-602bfa7f77f4" width="280" height="280">
  </figure>
  <figure style="display:inline-block; text-align:center; margin:10px;">
    <img src="https://github.com/user-attachments/assets/64cab94c-7354-494c-a64c-d8737da7fc9a" width="280" height="280">
  </figure>
  <figure style="display:inline-block; text-align:center; margin:10px;">
    <img src="https://github.com/user-attachments/assets/d0e5875e-ecc7-43ba-8ac2-a4e0b651b66c" width="280" height="280">
  </figure>
</p>










