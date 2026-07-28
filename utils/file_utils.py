from typing import Tuple
import base64, mimetypes


def image_to_base64(img: str) -> str:
    """将图片转换为base64编码"""
    try:
        mime = mimetypes.guess_type(img)[0] or "image/png"
        with open(img, "rb") as f:
            b64 = base64.b64encode(f.read()).decode("utf-8")
        return f"data:{mime};base64,{b64}"
    except Exception as e:
        print(f"[图片] 本地文件转 base64 失败：{e}")
        return ""
