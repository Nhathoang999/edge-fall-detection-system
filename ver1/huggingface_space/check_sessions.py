import importlib, time
m = importlib.import_module('app')
importlib.reload(m)
print('Checking session landmarks and drawing...')
import cv2, os
cap = cv2.VideoCapture('fall_example_1.mp4')
frames = []
for i in range(6):
    ok,f = cap.read()
    if not ok: break
    frames.append(f)
cap.release()
print('read', len(frames))
state = None
saved = 0
for i,f in enumerate(frames*10):
    out, state = m.process_frame_for_realtime(f, state)
    sid = state[0]
    s = m.SESSION_STORE.get(sid)
    if s:
        has_land = 'last_landmarks' in s and s['last_landmarks']
    else:
        has_land = False
    if i % 5 == 0:
        print('iter', i, 'has_landmarks', bool(has_land), 'last_label', s.get('last_label') if s else None)
    # save first few display frames to disk
    if saved < 3 and out is not None:
        p = f'test_out_{i}.jpg'
        cv2.imwrite(p, out)
        print('wrote', p)
        saved += 1
    time.sleep(0.02)
print('SESSION_STORE keys:', list(m.SESSION_STORE.keys()))
