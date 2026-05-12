@echo off
chcp 65001 >nul
echo 正在检测 API 连接...
python -c "from consciousness_sim.check_api import check; import sys; sys.exit(check())"
if %errorlevel% neq 0 (
    echo.
    echo 请修改 consciousness_sim\config.json 中的 api_key 和 base_url 后重试。
    pause
    exit /b 1
)
echo.
echo 正在安装依赖...
pip install -r requirements.txt -q
echo 正在启动 Web UI...
streamlit run consciousness_sim/app.py --server.port 8501
pause
