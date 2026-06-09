import time, sys
start = time.time()

from rag_engine import load_engine
engine = load_engine()

elapsed = time.time() - start
stats = engine.get_kb_stats() if engine else None
print(f"耗时: {elapsed:.1f}秒")
print(f"统计: {stats}")
