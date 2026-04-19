import importlib,traceback,os,sys,time
m = importlib.import_module('app')
importlib.reload(m)
# Read a few frames from example video and call realtime function repeatedly to simulate webcam
cap = None
try:
    vid = 'fall_example_1.mp4'
    cap = __import__('cv2').VideoCapture(vid)
    frames = []
    for i in range(6):
        ok, f = cap.read()
        if not ok: break
        frames.append(f)
    print('Read', len(frames), 'frames')
    state = None
    t0 = time.time()
    for i,f in enumerate(frames*10):
        out, state = m.process_frame_for_realtime(f, state)
        if i%5==0:
            print('iter', i, 'state preview:', state[0], state[3], state[4], state[5])
    t1 = time.time()
    print('Processed', len(frames)*10, 'calls in', round(t1-t0,3), 's')
except Exception:
    traceback.print_exc()
    sys.exit(1)
finally:
    try:
        if cap: cap.release()
    except:
        pass
