#!/usr/bin/env python3
"""
把当前目录下的图片整体变暗（模拟“变脏”）
"""

import os
from PIL import Image, ImageEnhance

# ======================
# 可调参数（核心）
# ======================
# 亮度因子：越小越黑（0.0 = 全黑，1.0 = 原图）
BRIGHTNESS = 0.55

# 对比度因子：越小越灰（模拟污垢感）
CONTRAST = 0.85

# 输出后缀
SUFFIX = "_dirty"
# ======================

def darken_image(img_path, out_path):
    img = Image.open(img_path).convert("RGB")

    # 降低亮度（模拟变脏）
    img = ImageEnhance.Brightness(img).enhance(BRIGHTNESS)

    # 降低对比度（更像污垢，而不是单纯变黑）
    img = ImageEnhance.Contrast(img).enhance(CONTRAST)

    img.save(out_path)
    print(f"✅ {img_path} -> {out_path}")


if __name__ == "__main__":
    exts = (".png", ".jpg", ".jpeg", ".bmp")
    files = [f for f in os.listdir(".") if f.lower().endswith(exts)]

    if not files:
        print("❌ 当前目录未找到图片")
        exit(1)

    for f in files:
        name, ext = os.path.splitext(f)
        out = f"{name}{SUFFIX}{ext}"
        darken_image(f, out)
