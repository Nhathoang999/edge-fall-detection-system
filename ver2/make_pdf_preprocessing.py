import base64
import json
import requests
import subprocess
import os

mermaid_code = """flowchart TD
    classDef raw node,stroke:#333,stroke-width:2px,fill:#ffe0b2
    classDef extract node,stroke:#333,stroke-width:2px,fill:#fff9c4
    classDef npy node,stroke:#333,stroke-width:2px,fill:#e1f5fe
    classDef norm node,stroke:#333,stroke-width:2px,fill:#e8f5e9
    classDef model node,stroke:#333,stroke-width:2px,fill:#fce4ec

    subgraph Phase1 [Giai đoạn 1: Tiền xử lý tĩnh - Static Preprocessing]
        A[Video thô MP4<br/>Hành vi ngã/bình thường]:::raw --> B[Pose Estimation Model<br/>MediaPipe / YOLO-Pose]:::extract
        B --> C[Trích xuất tọa độ cơ thể<br/>17 keypoints x, y, visibility]:::extract
        C --> D[Phân mảnh chuỗi thời gian<br/>Trích xuất 30 frames liên tiếp]:::extract
        D --> E[Lưu thành dataset<br/>File .npy Shape: 30, 51]:::npy
    end

    subgraph Phase2 [Giai đoạn 2: Chuẩn hóa động - Dynamic Normalization]
        E --> F[Data Loader<br/>Đọc file .npy vào bộ nhớ]:::npy
        F --> G{Xác định tọa độ tham chiếu<br/>Mid-Hip & Mid-Shoulder}:::norm
        G --> H[Dịch chuyển gốc tọa độ Centering<br/>Trừ đi tọa độ Mid-Hip]:::norm
        H --> I[Chuẩn hóa tỷ lệ Scaling<br/>Chia cho khoảng cách Vai-Hông]:::norm
        I --> J[Xử lý nhiễu<br/>Bỏ qua khớp có độ tin cậy thấp]:::norm
        J --> K[Dữ liệu đã chuẩn hóa<br/>Normalized Sequence]:::norm
    end

    K --> L((Đưa vào Mô hình<br/>Transformer / LSTM)):::model
"""

state = {
  "code": mermaid_code,
  "mermaid": '{"theme": "default"}'
}
state_b64 = base64.urlsafe_b64encode(json.dumps(state).encode('utf-8')).decode('utf-8')
img_url = f"https://mermaid.ink/svg/{state_b64}"

print(f"Downloading SVG from mermaid.ink...")
res = requests.get(img_url)

svg_b64 = base64.b64encode(res.content).decode('utf-8')
img_src = f"data:image/svg+xml;base64,{svg_b64}"

html_content = f"""
<!DOCTYPE html>
<html>
<head>
    <meta charset="utf-8">
    <style>
        body {{ font-family: Arial, sans-serif; margin: 40px; color: #333; line-height: 1.6; }}
        h1 {{ text-align: center; font-size: 24px; }}
        h2 {{ font-size: 18px; margin-top: 30px; }}
        .diagram {{ text-align: center; margin: 20px 0; }}
        img {{ max-width: 100%; height: auto; }}
        ul {{ margin-top: 0; }}
        li {{ margin-bottom: 10px; }}
    </style>
</head>
<body>
    <h1>Quy trình tiền xử lý dữ liệu (Data Preprocessing)</h1>
    
    <h2>1. Sơ đồ quy trình (Preprocessing Flow)</h2>
    <div class="diagram">
        <img src="{img_src}" alt="Preprocessing Diagram">
    </div>

    <h2>2. Giải thích các bước tiền xử lý</h2>
    
    <h3>Giai đoạn 1: Tiền xử lý tĩnh (Tạo Dataset)</h3>
    <ul>
        <li><strong>Trích xuất điểm neo:</strong> Chạy video qua mô hình Pose Estimation để lấy tọa độ của 17 khớp xương quan trọng.</li>
        <li><strong>Tạo cửa sổ thời gian (Sliding Window):</strong> Tách luồng video thành các phân đoạn 30 frames (timesteps). Mỗi frame chứa 17 x 3 = 51 đặc trưng.</li>
        <li><strong>Kết quả:</strong> Lưu thành dạng mảng NumPy (<code>.npy</code>) siêu nhẹ. Lúc này dữ liệu tuy gọn nhưng vẫn phụ thuộc vào vị trí người đứng trong camera.</li>
    </ul>

    <h3>Giai đoạn 2: Chuẩn hóa động (Dynamic Normalization)</h3>
    <p>Được chạy trong file <code>src/skeleton.py</code>, đây là bước "chuẩn hóa không gian" giúp mô hình không bị thiên lệch bởi camera:</p>
    <ul>
        <li><strong>Centering (Dịch gốc tọa độ):</strong> Lấy điểm <strong>Mid-Hip</strong> (trung điểm 2 hông) làm gốc (0,0). Mọi tọa độ khác sẽ được trừ đi Mid-Hip. Nhờ đó, dù người đứng ở góc trái hay phải video, dữ liệu đưa vào mô hình đều giống nhau.</li>
        <li><strong>Scaling (Chuẩn hóa tỷ lệ):</strong> Tính <code>reference_height</code> là khoảng cách từ vai (Mid-Shoulder) đến hông (Mid-Hip). Chia mọi tọa độ cho số này. Việc này giúp hệ thống chống chịu được việc người đứng xa hay đứng gần camera.</li>
        <li><strong>Khử nhiễu:</strong> Các điểm bị mờ, khuất (Confidence thấp) sẽ bị loại bỏ tầm ảnh hưởng để tránh làm sai lệch phép tính gốc tọa độ.</li>
    </ul>
</body>
</html>
"""

html_path = os.path.abspath("temp_prep.html")
pdf_path = os.path.abspath("Data_Preprocessing_Diagram.pdf")

with open(html_path, "w", encoding="utf-8") as f:
    f.write(html_content)

print("HTML generated, invoking MS Edge to print PDF...")
edge_exe = r"C:\Program Files (x86)\Microsoft\Edge\Application\msedge.exe"
cmd = [edge_exe, "--headless", "--disable-gpu", f"--print-to-pdf={pdf_path}", "--no-pdf-header-footer", html_path]

subprocess.run(cmd, check=True)
print(f"PDF successfully generated at: {pdf_path}")
