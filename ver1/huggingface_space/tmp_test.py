import importlib, time
m = importlib.import_module('app')
importlib.reload(m)
print('module reloaded')
# load a sample frame
import cv2
cap = cv2.VideoCapture('fall_example_1.mp4')
ok, f = cap.read()
cap.release()
print('have_frame', bool(ok))
# call new function with toggle False then True
state = None
out, state = m.process_frame_for_realtime(f, state, False)
print('called with False -> state preview:', state[0], state[1], state[4], state[5])
out2, state = m.process_frame_for_realtime(f, state, True)
print('called with True -> state preview:', state[0], state[1], state[4], state[5])
# ensure no exceptions
print('done')
