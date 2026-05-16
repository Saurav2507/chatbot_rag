import urllib.request
import re

html = urllib.request.urlopen('https://abetlen.github.io/llama-cpp-python/whl/cpu').read().decode('utf-8')
urls = re.findall(r'href=[\"\']([^\"\']+)[\"\']', html)
for w in urls[:20]:
    print(w)

