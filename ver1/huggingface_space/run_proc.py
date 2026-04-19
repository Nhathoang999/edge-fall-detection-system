import importlib, traceback, os,sys
m = importlib.import_module('app')
input_path = 'fall_example_1.mp4'
print('Input:', os.path.abspath(input_path))
try:
    out_path, summary = m.process_video_for_gradio(input_path)
    print('OUT:', out_path)
    print('SUMMARY:', (summary or '')[:500])
    if out_path:
        print('OUT exists:', os.path.exists(out_path), 'size=', os.path.getsize(out_path) if os.path.exists(out_path) else 'NA')
except Exception:
    traceback.print_exc()
    sys.exit(1)
