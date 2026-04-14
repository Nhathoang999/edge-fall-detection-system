import codecs
p = r'c:\KLTN\Fall-Detection\deployment\huggingface_space\app.py'
txt = open(p, 'r', encoding='utf-8').read()
new = 'overall_status = f"CẢNH BÁO TÉ NGÃ - Frame {frame_count}\\n" + str(overall_status)'
import re
txt = re.sub(r'overall_status = f\"CẢNH BÁO TÉ NGÃ - Frame \{frame_count\}\n\" \+ str\(overall_status\)', new, txt)
open(p, 'w', encoding='utf-8').write(txt)
print('FIXED')
