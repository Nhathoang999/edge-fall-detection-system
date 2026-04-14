import codecs, re
p = r'c:\KLTN\Fall-Detection\deployment\huggingface_space\app.py'
txt = codecs.open(p, 'r', 'utf-8').read()

txt = txt.replace('Đang gom đủ mẫu...', 'Buffering frames...')
txt = txt.replace('gom đủ mẫu', 'Buffering frames')
txt = txt.replace('Lỗi bộ phân tích thuật toán', 'Model inference error')
txt = txt.replace('Lỗi', 'Error')
txt = txt.replace('ĐÃ CÓ NGƯỜI NGÃ', 'FALL DETECTED')
txt = txt.replace('NGUY CẤP', 'CRITICAL')

# Fix 2: FFmpeg libx264 conversion
old_text = '    video_writer.release()'
new_text = '''    video_writer.release()
    try:
        import subprocess, imageio_ffmpeg
        ffmpeg_exe = imageio_ffmpeg.get_ffmpeg_exe()
        web_mp4 = "web_" + output_temp_video_path
        subprocess.run([ffmpeg_exe, "-y", "-i", output_temp_video_path, "-vcodec", "libx264", web_mp4], stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
        import os
        if os.path.exists(web_mp4):
            os.remove(output_temp_video_path)
            output_temp_video_path = web_mp4
    except Exception as e:
        print(f"Không thể re-encode video H264: {e}")'''
txt = txt.replace(old_text, new_text)

codecs.open(p, 'w', 'utf-8').write(txt)
