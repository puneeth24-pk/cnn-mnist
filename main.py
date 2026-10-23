import base64
import io
import os

import numpy as np
import tensorflow as tf
from fastapi import FastAPI, HTTPException
from fastapi.responses import FileResponse
from PIL import Image
from pydantic import BaseModel
from scipy.ndimage import center_of_mass, shift

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
MODEL_PATH = os.path.join(BASE_DIR, "mnist_model.keras")

if not os.path.exists(MODEL_PATH):
    raise FileNotFoundError(f"Put your trained model here: {MODEL_PATH}")

L = tf.keras.layers

# Rebuild the same architecture used in training, then load only the weights.
# (Avoids the "quantization_config" error caused by Keras version mismatch.)
model = tf.keras.Sequential([
    L.Input(shape=(28, 28, 1)),
    L.Conv2D(32, (3, 3), activation="relu"),
    L.MaxPooling2D((2, 2)),
    L.Conv2D(64, (3, 3), activation="relu"),
    L.MaxPooling2D((2, 2)),
    L.Flatten(),
    L.Dense(64, activation="relu"),
    L.Dense(32, activation="relu"),
    L.Dense(10, activation="softmax"),
])
model.load_weights(MODEL_PATH)

app = FastAPI(title="MNIST CNN Visualizer")


class DrawingRequest(BaseModel):
    image: str  # data URL: "data:image/png;base64,...."


# ---------------------------------------------------------------- preprocessing
def preprocess(data_url: str) -> np.ndarray:
    """Canvas PNG (white digit on black) -> MNIST-style (28, 28) float array."""
    try:
        image_bytes = base64.b64decode(data_url.split(",", 1)[1])
        image = Image.open(io.BytesIO(image_bytes)).convert("L")  # no invert
    except Exception:
        raise HTTPException(status_code=400, detail="Invalid image data.")

    pixels = np.array(image, dtype=np.uint8)
    rows, cols = np.where(pixels > 30)

    if len(rows) == 0:
        raise HTTPException(status_code=400, detail="Please draw a digit first.")

    cropped = pixels[rows.min():rows.max() + 1, cols.min():cols.max() + 1]

    h, w = cropped.shape
    scale = 20.0 / max(h, w)
    new_w = max(1, int(round(w * scale)))
    new_h = max(1, int(round(h * scale)))
    digit = Image.fromarray(cropped).resize((new_w, new_h), Image.Resampling.LANCZOS)

    canvas = Image.new("L", (28, 28), 0)
    canvas.paste(digit, ((28 - new_w) // 2, (28 - new_h) // 2))

    array = np.array(canvas, dtype=np.float32) / 255.0

    cy, cx = center_of_mass(array)
    if np.isfinite(cy) and np.isfinite(cx):
        array = shift(
            array,
            (int(round(14 - cy)), int(round(14 - cx))),
            order=1,
            mode="constant",
            cval=0,
        )

    return np.clip(array, 0, 1)


# ---------------------------------------------------------------- helpers
def layer_kind(layer) -> str:
    if isinstance(layer, L.Conv2D):
        return "conv"
    if isinstance(layer, L.MaxPooling2D):
        return "pool"
    if isinstance(layer, L.Flatten):
        return "flatten"
    if isinstance(layer, L.Dense):
        return "dense"
    return "other"


def pack_activation(kind: str, out: np.ndarray, is_last: bool):
    """Convert a layer output into compact JSON-friendly data for the frontend."""
    a = out[0]

    if kind in ("conv", "pool"):
        # (h, w, c) -> (c, h, w), each feature map scaled 0-255 on its own
        a = np.transpose(a, (2, 0, 1))
        mx = a.max(axis=(1, 2), keepdims=True)
        mx[mx == 0] = 1
        return (a / mx * 255).astype(int).tolist()

    # flatten / dense -> 1D list
    if is_last:
        return [round(float(v), 4) for v in a]  # softmax probabilities
    mx = float(a.max())
    mx = mx if mx > 0 else 1.0
    return [round(float(v) / mx, 3) for v in a]


# ---------------------------------------------------------------- routes
@app.get("/")
def home():
    return FileResponse(os.path.join(BASE_DIR, "index.html"))


@app.get("/filters")
def filters():
    """Learned 3x3 kernels of the first conv layer, scaled to -1..1."""
    first_conv = next(l for l in model.layers if isinstance(l, L.Conv2D))
    kernels = first_conv.get_weights()[0]            # (3, 3, 1, 32)
    kernels = np.transpose(kernels[:, :, 0, :], (2, 0, 1))  # (32, 3, 3)
    m = float(np.abs(kernels).max()) or 1.0
    return {"filters": (kernels / m).round(3).tolist()}


@app.post("/predict")
def predict(req: DrawingRequest):
    array = preprocess(req.image)
    x = tf.convert_to_tensor(array.reshape(1, 28, 28, 1))

    # run the image through the network layer by layer and keep every output
    layers_out = []
    n = len(model.layers)
    for i, layer in enumerate(model.layers):
        x = layer(x)
        out = x.numpy()
        kind = layer_kind(layer)
        layers_out.append({
            "name": layer.name,
            "kind": kind,
            "shape": list(out.shape[1:]),
            "data": pack_activation(kind, out, is_last=(i == n - 1)),
        })

    probs = x.numpy()[0]
    digit = int(np.argmax(probs))

    return {
        "digit": digit,
        "confidence": round(float(probs[digit]) * 100, 2),
        "probabilities": [round(float(p) * 100, 2) for p in probs],
        "processed": array.round(3).tolist(),
        "layers": layers_out,
    }


# Run:  uvicorn main:app --reload