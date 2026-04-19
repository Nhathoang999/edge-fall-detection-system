import os,importlib,traceback
cwd = os.getcwd()
base = 'fall_example_1.mp4'
pattern_suffix = '_' + base
before = sorted([f for f in os.listdir(cwd) if f.endswith(pattern_suffix)])
print('BEFORE copies:', before)
try:
    m = importlib.import_module('app')
    importlib.reload(m)
    out, summary = m.process_video_for_gradio(base)
    print('OUT:', out)
    print('SUMMARY PREVIEW:', (summary or '')[:200])
except Exception:
    traceback.print_exc()
    raise
after = sorted([f for f in os.listdir(cwd) if f.endswith(pattern_suffix)])
print('AFTER copies:', after)
new = set(after) - set(before)
print('NEW COPIES CREATED:', new)
