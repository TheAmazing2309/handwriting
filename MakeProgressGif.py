from PIL import Image, ImageDraw, ImageFont
import glob
import re
import os

FRAME_GLOB = "Graphs/epoch*batch100.png"
OUT_PATH = "assets/training_progress.gif"
FRAME_DURATION_MS = 350
MAX_WIDTH = 900

def loadFrames():
    frameRe = re.compile(r"epoch(\d+)batch100\.png$")
    frames = []
    for path in glob.glob(FRAME_GLOB):
        m = frameRe.search(path)
        if m:
            frames.append((int(m.group(1)), path))
    frames.sort()
    return frames

def captionFont(size=28):
    try:
        return ImageFont.truetype("arial.ttf", size)
    except OSError:
        return ImageFont.load_default()

def addCaption(img, text, font):
    img = img.convert("RGB")
    draw = ImageDraw.Draw(img)
    bbox = draw.textbbox((0, 0), text, font=font)
    pad = 8
    boxW, boxH = bbox[2] - bbox[0] + pad * 2, bbox[3] - bbox[1] + pad * 2
    draw.rectangle([0, 0, boxW, boxH], fill=(252, 252, 251))
    draw.text((pad, pad), text, font=font, fill=(11, 11, 11))
    return img

def main():
    frames = loadFrames()
    if not frames:
        print(f"No frames found matching {FRAME_GLOB}")
        return
    font = captionFont()
    images = [addCaption(Image.open(path), f"Epoch {epoch}", font) for epoch, path in frames]
    baseW, baseH = images[0].size
    scale = MAX_WIDTH / baseW
    targetSize = (MAX_WIDTH, round(baseH * scale))
    images = [im.resize(targetSize).convert("P", palette=Image.ADAPTIVE, colors=128) for im in images]

    os.makedirs("assets", exist_ok=True)
    images[0].save(
        OUT_PATH, save_all=True, append_images=images[1:],
        duration=FRAME_DURATION_MS, loop=0, optimize=True,
    )
    print(f"Saved {len(images)} frames (epochs {frames[0][0]}-{frames[-1][0]}) to {OUT_PATH}")

if __name__ == "__main__":
    main()
