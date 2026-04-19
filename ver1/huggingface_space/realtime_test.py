import importlib,traceback,os,sys
m = importlib.import_module('app')
importlib.reload(m)
print('SESSION_STORE keys before:', list(m.SESSION_STORE.keys())[:5])
# Simulate webcam frames
cap = __import__('cv2').VideoCapture('fall_example_1.mp4')
frames = []
for i in range(4):
    ok,f = cap.read()
    if not ok: break
    frames.append(f)
cap.release()
print('read', len(frames))
state = None
for i,f in enumerate(frames*5):
    out, state = m.process_frame_for_realtime(f, state)
    if i%3==0:
        print('iter', i, 'state preview:', state[0], state[1], state[2][:30], state[3], state[4], state[5])
print('SESSION_STORE keys after:', list(m.SESSION_STORE.keys())[:5])
print('done')
