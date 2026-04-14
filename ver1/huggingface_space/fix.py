import codecs
p = r'c:\KLTN\Fall-Detection\deployment\huggingface_space\app.py'
txt = codecs.open(p, 'r', 'utf-8').read()

old_text = '    video_writer.release()'
new_text = '''    video_writer.release()
    try:
        import subprocess, imageio_ffmpeg, os
        ffmpeg_exe = imageio_ffmpeg.get_ffmpeg_exe()
        web_mp4 = "web_" + output_temp_video_path
        subprocess.run([ffmpeg_exe, "-y", "-i", output_temp_video_path, "-vcodec", "libx264", web_mp4], stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
        if os.path.exists(web_mp4):
            os.remove(output_temp_video_path)
            output_temp_video_path = web_mp4
    except Exception as e:
        print(f"Không thể re-encode video H264: {e}")'''

if old_text in txt:
    txt = txt.replace(old_text, new_text)
    codecs.open(p, 'w', 'utf-8').write(txt)
    print('REPLACED')
else:
    print('NOT FOUND!')
