import codecs
p = r'c:\KLTN\Fall-Detection\deployment\huggingface_space\app.py'
txt = codecs.open(p, 'r', 'utf-8').read()
new = '''import tensorflow as tf
tflite = tf.lite'''
txt = txt.replace('import tflite_runtime.interpreter as tflite', new)
codecs.open(p, 'w', 'utf-8').write(txt)
