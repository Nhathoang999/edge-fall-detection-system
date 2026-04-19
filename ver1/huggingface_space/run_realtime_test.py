import importlib, time
m = importlib.import_module('app')
importlib.reload(m)
print('SESSION_STORE before:', list(m.SESSION_STORE.keys()))
import cv2
cap = cv2.VideoCapture('fall_example_1.mp4')
frames = []
for i in range(8):
    ok,f = cap.read()
    if not ok: break
    frames.append(f)
cap.release()
print('read', len(frames))
state = None
for i,f in enumerate(frames*6):
    out, state = m.process_frame_for_realtime(f, state)
    sid = state[0]
    if i % 3 == 0:
        s = m.SESSION_STORE.get(sid)
        qsize = None
        if s and 'in_queue' in s:
            try:
                qsize = s['in_queue'].qsize()
            except Exception:
                qsize = 'err'
        print('iter', i, 'sid', sid[:8], 'worker_running', s.get('running') if s else None, 'in_q', qsize, 'last_label', s.get('last_label') if s else None, 'last_conf', s.get('last_conf') if s else None, 'last_center', s.get('last_center') if s else None)
    time.sleep(0.03)
print('SESSION_STORE after:', list(m.SESSION_STORE.keys()))
