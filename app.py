import time
import numpy as np
from PIL import Image, ImageOps
import torch
from fastai.vision.all import *
import gradio as gr
import matplotlib.pyplot as plt

# =====================================================
# MODEL INITIALIZATION
# =====================================================
dummy_imgs = torch.zeros((1, 28, 28))
dummy_lbls = torch.tensor([0])

mnist_block = DataBlock(
    blocks=(ImageBlock(cls=PILImageBW), CategoryBlock(vocab=list(range(10)))),
    get_items=lambda x: range(len(dummy_imgs)),
    splitter=FuncSplitter(lambda x: False),
    get_x=lambda i: dummy_imgs[i],
    get_y=lambda i: int(dummy_lbls[i].item())
)

dls_dummy = mnist_block.dataloaders(range(len(dummy_imgs)), bs=1)

learn = vision_learner(
    dls=dls_dummy,
    arch=models.resnet18,
    loss_func=CrossEntropyLossFlat(),
    metrics=[accuracy],
    n_out=10
)

learn.load("mnist_best_weights_only")

# =====================================================
# PREPROCESSING FUNCTIONS
# =====================================================
def preprocess_image(img):
    if img is None:
        return None
    if isinstance(img, dict):
        img = img.get("composite", None)
    if img is None:
        return None
    if isinstance(img, Image.Image):
        img = np.array(img)
    if img.dtype != np.uint8:
        img = (img * 255).astype(np.uint8)
    if len(img.shape) == 3:
        img = np.mean(img, axis=2).astype(np.uint8)
    img = (img - img.min()) / (img.max() - img.min() + 1e-5)
    img = (img * 255).astype(np.uint8)
    threshold = np.mean(img)
    img_bin = (img > threshold).astype(np.uint8) * 255
    corners = [img_bin[0,0], img_bin[0,-1], img_bin[-1,0], img_bin[-1,-1]]
    if np.mean(corners) > 127:
        img_bin = 255 - img_bin
    coords = np.column_stack(np.where(img_bin > 0))
    if len(coords) > 0:
        y_min, x_min = coords.min(axis=0)
        y_max, x_max = coords.max(axis=0)
        img_crop = img_bin[y_min:y_max+1, x_min:x_max+1]
    else:
        img_crop = img_bin
    h, w = img_crop.shape
    size = max(h, w)
    padded = np.zeros((size, size), dtype=np.uint8)
    y_offset = (size - h) // 2
    x_offset = (size - w) // 2
    padded[y_offset:y_offset+h, x_offset:x_offset+w] = img_crop
    img_pil = Image.fromarray(padded).resize((28,28), Image.Resampling.BILINEAR).convert("L")
    return img_pil

def add_frame(img, frame_color=(255,255,255), bg_color=(240,240,240), size=(120,120)):
    img = img.convert("RGB")
    img_bg = Image.new("RGB", size, bg_color)
    img = ImageOps.expand(img, border=2, fill=frame_color)
    img_bg.paste(img, ((size[0]-img.width)//2, (size[1]-img.height)//2))
    return img_bg

def preprocessing_steps_preview(data):
    img = data.get("composite") if isinstance(data, dict) else data
    if img is None:
        return [None]*5
    if isinstance(img, Image.Image):
        img = np.array(img)

    preview_orig = add_frame(Image.fromarray(img.astype(np.uint8) if img.dtype != np.uint8 else img))
    img_gray = np.mean(img, axis=2).astype(np.uint8) if len(img.shape) == 3 else img.copy()
    img_gray = ((img_gray - img_gray.min()) / (img_gray.max() - img_gray.min() + 1e-5) * 255).astype(np.uint8)
    preview_gray = add_frame(Image.fromarray(img_gray))
    threshold = np.mean(img_gray)
    img_bin = (img_gray > threshold).astype(np.uint8)*255
    preview_bin = add_frame(Image.fromarray(img_bin))
    corners = [img_bin[0,0], img_bin[0,-1], img_bin[-1,0], img_bin[-1,-1]]
    img_inv = 255 - img_bin if np.mean(corners) > 127 else img_bin.copy()
    preview_inv = add_frame(Image.fromarray(img_inv))
    coords = np.column_stack(np.where(img_inv > 0))
    if len(coords) > 0:
        y_min, x_min = coords.min(axis=0)
        y_max, x_max = coords.max(axis=0)
        img_crop = img_inv[y_min:y_max+1, x_min:x_max+1]
    else:
        img_crop = img_inv
    h, w = img_crop.shape
    size_max = max(h,w)
    padded = np.zeros((size_max,size_max), dtype=np.uint8)
    y_offset = (size_max-h)//2
    x_offset = (size_max-w)//2
    padded[y_offset:y_offset+h, x_offset:x_offset+w] = img_crop
    preview_final = add_frame(Image.fromarray(padded).resize((28,28), Image.Resampling.BILINEAR))

    return [preview_orig, preview_gray, preview_bin, preview_inv, preview_final]

def preprocess_preview(data):
    return preprocess_image(data)

# =====================================================
# PREDICTION
# =====================================================
def predict_and_plot(data):
    img = data.get("composite") if isinstance(data, dict) else data
    if img is None:
        return "No image provided", None, ""
    
    start_time = time.time()
    img_pil = preprocess_image(img)
    if img_pil is None or not isinstance(img_pil, Image.Image):
        return "No se pudo procesar la imagen", None, ""
    if img_pil.mode != "L":
        img_pil = img_pil.convert("L")

    img_model = PILImageBW.create(img_pil)
    pred_class, pred_idx, probs = learn.predict(img_model)
    probs_percent = (probs.numpy()*100).tolist()
    confidence = probs[pred_idx]*100

    label_text = "No estoy seguro del dígito" if confidence < 60 else f"Dígito: {pred_class}"
    output_text = (
        f"{label_text}\n"
        f"Confianza: {confidence:.2f}%\n"
        f"Inference time: {round((time.time()-start_time)*1000,2)} ms"
    )

    top2_idx = np.argsort(probs.numpy())[::-1][:2]
    explanation = (
        f"Predicción principal: {pred_class} ({probs_percent[top2_idx[0]]:.1f}%)\n"
        f"Segunda opción: {top2_idx[1]} ({probs_percent[top2_idx[1]]:.1f}%)\n"
        f"Interpretación: la forma del dígito se asemeja más a {pred_class}, "
        f"pero podría confundirse con {top2_idx[1]}."
    )

    digits = list(range(10))
    fig, ax = plt.subplots()
    ax.bar(digits, probs_percent, color='skyblue')
    ax.set_xlabel("Dígito")
    ax.set_ylabel("Probabilidad (%)")
    ax.set_xticks(digits)
    ax.set_ylim(0, 100)

    return output_text, fig, explanation

# =====================================================
# ACTIVATION VISUALIZATION
# =====================================================
def visualize_activations(data):
    img = data.get("composite") if isinstance(data, dict) else data
    if img is None:
        return [None]*8

    img_pil = preprocess_image(img)
    img_tensor = torch.tensor(np.array(img_pil)/255.0, dtype=torch.float32).unsqueeze(0)  # [1,H,W]
    img_tensor = img_tensor.repeat(1,3,1,1)  # Convertir a 3 canales RGB

    activations = []

    def hook_fn(module, input, output):
        nonlocal activations
        activations = output.detach().squeeze(0)

    handle = learn.model.conv1.register_forward_hook(hook_fn)
    with torch.no_grad():
        _ = learn.model(img_tensor)
    handle.remove()

    act_imgs = []
    for i in range(min(8, activations.shape[0])):
        act_img = activations[i].numpy()
        act_img = ((act_img - act_img.min()) / (act_img.max() - act_img.min() + 1e-5) * 255).astype(np.uint8)
        act_imgs.append(add_frame(Image.fromarray(act_img)))

    while len(act_imgs) < 8:
        act_imgs.append(None)

    return act_imgs

# =====================================================
# GRADIO UI
# =====================================================
with gr.Blocks(title="Senior ML Portfolio - MNIST System") as demo:
    gr.Markdown("# 🧠 End-to-End ML System: MNIST Digit Classifier")
    gr.Markdown(
        "Proyecto completo de Machine Learning: dibuja o sube cualquier dígito, "
        "y el sistema lo preprocesa y clasifica automáticamente."
    )

    # ----- Demo -----
    with gr.Tab("🏠 Demo"):
        with gr.Row():
            with gr.Column(scale=1):
                gr.Markdown("### 📌 Paso 0: Dibuja o sube un dígito")
                img_input = gr.ImageEditor(
                    type="numpy",
                    label="Dibuja o sube un número",
                    image_mode="L",
                    height=200,
                    width=200
                )
            with gr.Column(scale=1):
                gr.Markdown("### 🖼 Previsualización del preprocesado")
                preview_img = gr.Image(label="Preprocessed Image (28x28)", interactive=False, type="pil")

        with gr.Row():
            with gr.Column(scale=1):
                gr.Markdown("### ⚡ Pasos a seguir")
                btn_preprocess = gr.Button("1️⃣ Preview 28x28")
                gr.Markdown("Ver cómo se procesa tu imagen")
                btn_predict = gr.Button("2️⃣ Run Inference")
                gr.Markdown("Obtener predicción y probabilidades")
            with gr.Column(scale=1):
                gr.Markdown("### 📊 Resultados de Predicción")
                output_text = gr.Textbox(label="Prediction Output")
                bar_chart = gr.Plot(label="Probabilidades por dígito (%)")
                explanation_text = gr.Textbox(label="Model Explanation")

        btn_preprocess.click(preprocess_preview, inputs=img_input, outputs=preview_img)
        btn_predict.click(predict_and_plot, inputs=img_input, outputs=[output_text, bar_chart, explanation_text])

    # ----- Data & Preprocessing -----
    with gr.Tab("📊 Data & Preprocessing"):
        gr.Markdown("## Pipeline de Preprocesamiento Paso a Paso")
        img_input_pipeline = img_input
        btn_preview_pipeline = gr.Button("Mostrar Preprocesamiento")
        with gr.Row():
            preview_outputs = []
            for i, title in enumerate([
                "1️⃣ Original", "2️⃣ Grayscale", "3️⃣ Binarizado", "4️⃣ Fondo invertido", "5️⃣ Centrado + Resize"
            ]):
                with gr.Column():
                    preview_outputs.append(gr.Image(label=title))
        btn_preview_pipeline.click(preprocessing_steps_preview, inputs=img_input_pipeline, outputs=preview_outputs)

    # ----- Model -----
    with gr.Tab("🧠 Model"):
        gr.Markdown("### Visualización de Activaciones (primeras 8)")
        btn_activations = gr.Button("Mostrar Activaciones")
        act_outputs = []
        with gr.Row():
            for i in range(8):
                with gr.Column():
                    act_outputs.append(gr.Image(label=f"Act {i+1}"))
        btn_activations.click(visualize_activations, inputs=img_input, outputs=act_outputs)

    # ----- Deployment -----
    with gr.Tab("⚙️ Deployment"):
        gr.Markdown("""
        - Separación training / inference
        - Carga de pesos (.pth)
        - Deploy en Hugging Face Spaces
        """)

    # ----- Engineering -----
    with gr.Tab("🏗 Engineering"):
        gr.Markdown("""
        - Pipeline consistente
        - Evitado load_learner
        - Preprocesado robusto
        - Control de confianza y explicabilidad
        """)

    # ----- Confusion Matrix -----
    with gr.Tab("📉 Confusion Matrix"):
        gr.Image(value="confusion_matrix.png", label="Confusion Matrix")

demo.launch()
