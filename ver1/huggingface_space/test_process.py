import importlib, traceback, os,sys
print('CWD', os.getcwd())
try:
    m = importlib.import_module('app')
    importlib.reload(m)
    tests = [
        'fall_example_1.mp4',
        ['fall_example_1.mp4'],
        {'name': 'fall_example_1.mp4'},
        {'tmp_path': 'fall_example_1.mp4'}
    ]
    for t in tests:
        print('\n--- Test input:', type(t), t if not isinstance(t, (list,dict)) else str(t)[:80])
        try:
            out, summary = m.process_video_for_gradio(t)
            print('-> OUT:', out, 'exists=', os.path.exists(out) if out else None, 'size=', os.path.getsize(out) if out and os.path.exists(out) else None)
            print('-> summary preview:', (summary or '')[:200])
        except Exception:
            traceback.print_exc()
            sys.exit(1)
    print('\nAll tests done')
except Exception:
    traceback.print_exc()
    sys.exit(1)
