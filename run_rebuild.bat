@echo off
cd /d "%~dp0"
echo 开始重建索引，请稍候...
python _rebuild_index.py > rebuild_output.txt 2>&1
echo 完成！查看 rebuild_output.txt 确认结果
pause
