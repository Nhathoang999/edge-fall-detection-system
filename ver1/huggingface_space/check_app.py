import importlib,sys,traceback,os
print("CWD", os.getcwd())
print("PYTHONPATH", sys.path[:5])
try:
    m = importlib.import_module("app")
    print("OK", hasattr(m,"process_video_for_gradio"))
except Exception:
    traceback.print_exc()
