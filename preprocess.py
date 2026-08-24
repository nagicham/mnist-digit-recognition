import numpy as np
from PIL import Image, ImageOps

BINARIZE_THRESHOLD = 0.5
DIGIT_BOX_SIZE = 20
CANVAS_SIZE = 28

def preprocess_image(image, invert=True):
    # 1. グレースケール化
    img = image.convert("L")

    # 2. 色の反転(白背景・黒文字 ー> 黒背景・白文字へ)
    if invert:
        img = ImageOps.invert(img)

    img = img.resize((CANVAS_SIZE, CANVAS_SIZE), Image.LANCZOS)

    arr = np.asarray(img, dtype=np.float32) / 255.0
    arr = np.where(arr > BINARIZE_THRESHOLD, 1.0, 0.0).astype(np.float32)
    
    return arr.reshape(1, 1, CANVAS_SIZE, CANVAS_SIZE)

    """
    
    # 3. 画素値を 0.0 ~ 1.0 にする
    arr = np.asarray(img, dtype=np.float32) / 255.0

    # 4. 数字部分のバウンディングボックスを検出して切り出す
    mask = arr > 0.1
    if mask.any():
        rows = np.any(mask, axis=1)
        cols = np.any(mask, axis=0)
        top, bottom = np.where(rows)[0][[0, -1]]
        left, right = np.where(cols)[0][[0, -1]]
        digit = arr[top : bottom + 1, left : right + 1]
    else:
        digit = arr

    # 5. アスペクト比を保ったまま、20×20に収まるようにリサイズ
    digit_img = Image.fromarray((digit * 255).astype(np.uint8))
    h, w = digit.shape
    if h > w:
        new_h = DIGIT_BOX_SIZE
        new_w = max(1, round(w * DIGIT_BOX_SIZE / h))
    else:
        new_w = DIGIT_BOX_SIZE
        new_h = max(1, round(w * DIGIT_BOX_SIZE / w))
    digit_img = digit_img.resize((new_w, new_h), Image.LANCZOS)

    # 6. 28×28の黒キャンパスの中央に貼り付ける
    canvas = Image.new("L", (CANVAS_SIZE, CANVAS_SIZE), color=0)
    paste_x = (CANVAS_SIZE - new_w) // 2
    paste_y = (CANVAS_SIZE - new_h) // 2
    canvas.paste(digit_img, (paste_x, paste_y))

    # 7. 正規化してから、学習時と同じ閾値で二値化
    final_arr = np.asarray(canvas, dtype=np.float32) / 255.0
    final_arr = np.where(final_arr > BINARIZE_THRESHOLD, 1.0, 0.0).astype(np.float32)
    """




