import os
import sys

os.environ['HF_ENDPOINT'] = 'https://hf-mirror.com'

# Fix stdout encoding for Windows
if sys.platform == 'win32':
    import io
    sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8')

try:
    from ingest import ingest
    vs = ingest(force_rebuild=True)
    print('Vectorstore:', type(vs))
except Exception as e:
    import traceback
    traceback.print_exc()
    raise
