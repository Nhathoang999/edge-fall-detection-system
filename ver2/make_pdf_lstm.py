import base64
import json
import requests
import subprocess
import os

mermaid_code = """flowchart TD
    classDef input node,stroke:#333,stroke-width:2px,fill:#e3f2fd,color:#000
    classDef lstm node,stroke:#333,stroke-width:2px,fill:#fff3e0,color:#000
    classDef dropout node,stroke:#333,stroke-width:1px,fill:#f5f5f5,stroke-dasharray: 5 5,color:#000
    classDef dense node,stroke:#333,stroke-width:2px,fill:#e8f5e9,color:#000
    classDef output node,stroke:#333,stroke-width:2px,fill:#ffebee,color:#000

    A["Input Layer<br>Shape: (30 frames, 51 features)"]:::input
    
    B["LSTM Layer 1<br>Units: 64, return_sequences=True<br>Output Shape: (30, 64)"]:::lstm
    C["Dropout Layer 1<br>Rate: 0.2"]:::dropout
    
    D["LSTM Layer 2<br>Units: 32<br>Output Shape: (32)"]:::lstm
    E["Dropout Layer 2<br>Rate: 0.2"]:::dropout
    
    F["Dense Layer (Hidden)<br>Units: 32, Activation: ReLU"]:::dense
    G["Dense Layer (Output)<br>Units: 1, Activation: Sigmoid"]:::output
    
    H["Fall Probability<br>(0.0 to 1.0)"]:::output

    A -->|"Skeleton Sequence"| B
    B -->|"Temporal Features"| C
    C -->|"Regularized Features"| D
    D -->|"Final Temporal State"| E
    E -->|"Regularized State"| F
    F -->|"Non-linear Features"| G
    G -->|"Probability Score"| H
"""

html_content = f"""<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>LSTM Architecture Diagram</title>
    <script type="module">
        import mermaid from 'https://cdn.jsdelivr.net/npm/mermaid@10/dist/mermaid.esm.min.mjs';
        mermaid.initialize({{ startOnLoad: true, theme: 'base' }});
    </script>
    <style>
        body {{
            font-family: Arial, sans-serif;
            display: flex;
            justify-content: center;
            align-items: center;
            height: 100vh;
            margin: 0;
            background-color: white;
        }}
        .mermaid {{
            transform: scale(1.5);
            transform-origin: center top;
        }}
    </style>
</head>
<body>
    <div class="mermaid">
        {mermaid_code}
    </div>
    <script>
        setTimeout(() => {{
            window.status = 'ready';
        }}, 2000);
    </script>
</body>
</html>
"""

with open("temp_lstm.html", "w", encoding="utf-8") as f:
    f.write(html_content)

edge_paths = [
    r"C:\Program Files (x86)\Microsoft\Edge\Application\msedge.exe",
    r"C:\Program Files\Microsoft\Edge\Application\msedge.exe"
]

edge_path = next((p for p in edge_paths if os.path.exists(p)), None)

if edge_path:
    print(f"Using Edge at: {edge_path}")
    pdf_path = os.path.abspath("LSTM_Architecture_Diagram.pdf")
    html_path = f"file:///{os.path.abspath('temp_lstm.html').replace(chr(92), '/')}"
    
    cmd = [
        edge_path,
        "--headless",
        "--disable-gpu",
        "--run-all-compositor-stages-before-draw",
        f"--print-to-pdf={pdf_path}",
        html_path
    ]
    
    try:
        subprocess.run(cmd, check=True, capture_output=True, text=True)
        print(f"Successfully generated {pdf_path}")
    except subprocess.CalledProcessError as e:
        print(f"Error generating PDF: {e.stderr}")
else:
    print("Could not find MS Edge.")
