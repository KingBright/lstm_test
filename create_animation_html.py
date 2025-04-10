# create_animation_html.py

import pandas as pd
import numpy as np
import json
import os
import config  # Import config to get file paths and parameters
import utils   # Import utils for safe_text

# --- Configuration ---
DATA_FILE_TO_ANIMATE = "train_data_sep.csv" # 默认使用验证集文件 (通常较小)
# DATA_FILE_TO_ANIMATE = config.TRAIN_DATA_FILE # 取消注释这行来使用训练集文件 (可能非常大!)
# NUM_FRAMES_TO_ANIMATE = 1000 # <<<--- 移除帧数限制 ---<<<
OUTPUT_HTML_FILE = "pendulum_animation.html"
PENDULUM_LENGTH_PIXELS = 150 # Visual length in pixels on canvas
CANVAS_WIDTH = 400
CANVAS_HEIGHT = 350 # Enough height for swing
MAX_SPEED_MULTIPLIER = 20 # Max speed for slider control

# --- HTML Template ---
# (HTML_TEMPLATE 保持不变，与上一个版本相同)
HTML_TEMPLATE = """
<!DOCTYPE html>
<html lang="zh">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>单摆运动动画</title>
    <script src="https://cdn.tailwindcss.com"></script>
    <style>
        body {{ margin: 0; display: flex; justify-content: center; align-items: center; min-height: 100vh; background-color: #f0f4f8; }}
        canvas {{ background-color: #ffffff; border-radius: 8px; box-shadow: 0 4px 6px rgba(0,0,0,0.1); }}
        .controls {{ margin-top: 1rem; text-align: center; width: {canvas_width}px; }}
        button {{ padding: 0.5rem 1rem; margin: 0 0.5rem; background-color: #4a90e2; color: white; border: none; border-radius: 5px; cursor: pointer; transition: background-color 0.2s; }}
        button:hover {{ background-color: #357abd; }}
        #timeLabel {{ font-family: monospace; margin-top: 0.5rem; color: #333; }}
        .speed-control {{ margin-top: 0.5rem; display: flex; justify-content: center; align-items: center; }}
        input[type=range] {{ width: 60%; margin: 0 0.5rem; cursor: pointer; }}
    </style>
</head>
<body class="flex flex-col items-center justify-center min-h-screen bg-gray-100">
    <h1 class="text-2xl font-bold mb-4 text-gray-700">{title_text}</h1>
    <canvas id="pendulumCanvas" width="{canvas_width}" height="{canvas_height}"></canvas>

    <div class="controls">
        <div>
            <button id="playPauseBtn">播放</button>
            <button id="resetBtn">重置</button>
        </div>
        <div class="speed-control">
            <label for="speedSlider" class="text-sm text-gray-600">速度:</label>
            <input type="range" id="speedSlider" min="1" max="{max_speed}" value="1" step="1">
            <span id="speedValue" class="text-sm text-gray-600 font-mono">1x</span>
        </div>
        <div id="timeLabel">Time: 0.00s</div>
    </div>

    <script>
        const canvas = document.getElementById('pendulumCanvas');
        const ctx = canvas.getContext('2d');
        const pivotX = canvas.width / 2;
        const pivotY = 50;
        const pendulumLength = {pendulum_length};
        const bobRadius = 10;
        let animationFrameId = null;
        let currentFrame = 0;
        let isPlaying = false;
        let speedMultiplier = 1;

        // --- Embedded Data ---
        const timeData = {time_data_json};
        const thetaData = {theta_data_json};
        const numFrames = thetaData.length; // <<<--- 使用全部数据的长度 ---<<<

        // --- Get Control Elements ---
        const playPauseBtn = document.getElementById('playPauseBtn');
        const resetBtn = document.getElementById('resetBtn');
        const timeLabel = document.getElementById('timeLabel');
        const speedSlider = document.getElementById('speedSlider');
        const speedValueSpan = document.getElementById('speedValue');

        function drawPendulum(theta) {{
            const bobX = pivotX + pendulumLength * Math.sin(theta);
            const bobY = pivotY + pendulumLength * Math.cos(theta);
            ctx.clearRect(0, 0, canvas.width, canvas.height);
            ctx.beginPath(); ctx.arc(pivotX, pivotY, 5, 0, 2 * Math.PI); ctx.fillStyle = '#333'; ctx.fill(); // Pivot
            ctx.beginPath(); ctx.moveTo(pivotX, pivotY); ctx.lineTo(bobX, bobY); ctx.strokeStyle = '#555'; ctx.lineWidth = 3; ctx.stroke(); // Rod
            ctx.beginPath(); ctx.arc(bobX, bobY, bobRadius, 0, 2 * Math.PI); ctx.fillStyle = '#e53e3e'; ctx.fill(); // Bob
        }}

        function animate() {{
            if (!isPlaying || currentFrame >= numFrames) {{
                isPlaying = false; playPauseBtn.textContent = '播放';
                if (currentFrame >= numFrames) currentFrame = numFrames - 1;
                if (currentFrame >= 0 && currentFrame < numFrames) {{
                   drawPendulum(thetaData[currentFrame]);
                   timeLabel.textContent = `Time: ${{timeData[currentFrame].toFixed(2)}}s`;
                }}
                cancelAnimationFrame(animationFrameId); return;
            }}
            if (currentFrame < 0) currentFrame = 0;
            if (currentFrame >= numFrames) {{ // Should not happen if logic above is correct
                currentFrame = numFrames - 1; isPlaying = false; animate(); return;
            }}

            const theta = thetaData[currentFrame];
            drawPendulum(theta);
            timeLabel.textContent = `Time: ${{timeData[currentFrame].toFixed(2)}}s`;
            currentFrame += speedMultiplier;
            animationFrameId = requestAnimationFrame(animate);
        }}

        // --- Event Listeners ---
        playPauseBtn.addEventListener('click', () => {{
            if (isPlaying) {{ isPlaying = false; playPauseBtn.textContent = '播放'; cancelAnimationFrame(animationFrameId); }}
            else {{
                if (currentFrame >= numFrames -1) {{ currentFrame = 0; }}
                isPlaying = true; playPauseBtn.textContent = '暂停';
                cancelAnimationFrame(animationFrameId); animationFrameId = requestAnimationFrame(animate);
            }}
        }});
        resetBtn.addEventListener('click', () => {{
            isPlaying = false; playPauseBtn.textContent = '播放'; cancelAnimationFrame(animationFrameId);
            currentFrame = 0; speedMultiplier = 1; speedSlider.value = 1; speedValueSpan.textContent = '1x';
            if (numFrames > 0) {{ drawPendulum(thetaData[0]); timeLabel.textContent = `Time: ${{timeData[0].toFixed(2)}}s`; }}
            else {{ ctx.clearRect(0, 0, canvas.width, canvas.height); timeLabel.textContent = `Time: 0.00s`; }}
        }});
        speedSlider.addEventListener('input', () => {{
            speedMultiplier = parseInt(speedSlider.value, 10); speedValueSpan.textContent = `${{speedMultiplier}}x`;
        }});

        // Initial draw
        if (numFrames > 0) {{ drawPendulum(thetaData[0]); timeLabel.textContent = `Time: ${{timeData[0].toFixed(2)}}s`; }}
        else {{ ctx.font = "16px Arial"; ctx.textAlign = "center"; ctx.fillText("无数据显示", canvas.width/2, canvas.height/2); timeLabel.textContent = `Time: 0.00s`; }}
    </script>
</body>
</html>
"""

def create_animation():
    """Loads data and generates the animation HTML file using all data points."""
    print(f"开始生成动画 HTML 文件...")
    print(f"将从以下文件加载 *全部* 数据: {DATA_FILE_TO_ANIMATE}")
    print("警告：如果数据文件非常大，生成的 HTML 文件可能很大且浏览器运行缓慢！")

    # --- Load Data ---
    try:
        df = pd.read_csv(DATA_FILE_TO_ANIMATE)
        if df.empty: print(f"错误: 数据文件 '{DATA_FILE_TO_ANIMATE}' 为空。"); return
        num_points = len(df) # <<<--- 使用全部数据点 ---<<<
        print(f"数据加载成功，将使用全部 {num_points} 个数据点生成动画。")
    except FileNotFoundError: print(f"错误: 数据文件 '{DATA_FILE_TO_ANIMATE}' 未找到。"); return
    except Exception as e: print(f"加载数据时出错: {e}"); return

    if num_points <= 0: print("错误: 没有数据点用于动画。"); return

    # Use the entire DataFrame for animation
    anim_df = df # <<<--- 直接使用整个 DataFrame ---<<<

    # Extract data and convert to lists for JSON embedding
    try:
        time_data = anim_df['time'].tolist(); theta_data = anim_df['theta'].tolist()
        time_json = json.dumps(time_data); theta_json = json.dumps(theta_data)
    except KeyError as ke:
        print(f"错误: 数据文件中缺少必需的列: {ke}")
        return
    except Exception as e:
        print(f"处理数据时出错: {e}")
        return

    # --- Generate HTML ---
    title_text = utils.safe_text(f"单摆动画 - {os.path.basename(DATA_FILE_TO_ANIMATE)} (全部 {num_points} 点)",
                                f"Pendulum Animation - {os.path.basename(DATA_FILE_TO_ANIMATE)} (All {num_points} points)")

    html_content = HTML_TEMPLATE.format(
        title_text=title_text,
        canvas_width=CANVAS_WIDTH,
        canvas_height=CANVAS_HEIGHT,
        pendulum_length=PENDULUM_LENGTH_PIXELS,
        max_speed=MAX_SPEED_MULTIPLIER,
        time_data_json=time_json,
        theta_data_json=theta_json
    )

    # --- Write HTML File ---
    try:
        with open(OUTPUT_HTML_FILE, 'w', encoding='utf-8') as f: f.write(html_content)
        print(f"动画文件已成功生成: {OUTPUT_HTML_FILE}")
        print(f"请用你的网页浏览器打开此文件查看动画。")
    except Exception as e: print(f"写入 HTML 文件时出错: {e}")

if __name__ == "__main__":
    create_animation()
