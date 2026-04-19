import sys
sys.path.insert(0, "d:/Du_Lieu_Hoc/KLTN/THESIS_CS/ver1/huggingface_space")

try:
    import cv2
    print("✓ cv2 imported")
except ImportError as e:
    print(f"✗ cv2: {e}")

try:
    import mediapipe as mp
    print("✓ mediapipe imported")
except ImportError as e:
    print(f"✗ mediapipe: {e}")

try:
    import numpy as np
    print("✓ numpy imported")
except ImportError as e:
    print(f"✗ numpy: {e}")

try:
    import tensorflow as tf
    print("✓ tensorflow imported")
except ImportError as e:
    print(f"✗ tensorflow: {e}")

try:
    import gradio as gr
    print("✓ gradio imported")
except ImportError as e:
    print(f"✗ gradio: {e}")

print("\nKiểm tra syntax file app.py...")
try:
    import py_compile
    py_compile.compile("d:/Du_Lieu_Hoc/KLTN/THESIS_CS/ver1/huggingface_space/app.py", doraise=True)
    print("✓ app.py syntax OK")
except py_compile.PyCompileError as e:
    print(f"✗ Syntax error: {e}")
except Exception as e:
    print(f"✗ Error: {e}")
