import time
import numpy as np
from PIL import Image, ImageOps
import torch
from fastai.vision.all import *
import gradio as gr
import matplotlib.pyplot as plt

# =====================================================
# INICIALIZACIÓN DEL MODELO
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
# FUNCIONES DE PREPROCESAMIENTO
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
# PREDICCIÓN
# =====================================================
def predict_and_plot(data):
    img = data.get("composite") if isinstance(data, dict) else data
    if img is None:
        return "No se ha proporcionado ninguna imagen", None, ""
    
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
        f"Tiempo de inferencia: {round((time.time()-start_time)*1000,2)} ms"
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
# VISUALIZACIÓN DE ACTIVACIONES
# =====================================================
def visualize_activations(data):
    img = data.get("composite") if isinstance(data, dict) else data
    if img is None:
        return [None]*8

    img_pil = preprocess_image(img)
    img_array = np.array(img_pil) / 255.0
    img_tensor = torch.tensor(img_array, dtype=torch.float32).unsqueeze(0).unsqueeze(0)
    img_tensor = img_tensor.repeat(1,3,1,1)
    device = next(learn.model.parameters()).device
    img_tensor = img_tensor.to(device)

    activations = None
    def hook_fn(module, input, output):
        nonlocal activations
        activations = output.detach().cpu()

    layer = learn.model[0][0]
    handle = layer.register_forward_hook(hook_fn)
    with torch.no_grad():
        _ = learn.model(img_tensor)
    handle.remove()

    if activations is None:
        return [None]*8

    activations = activations.squeeze(0)
    act_imgs = []

    for i in range(min(8, activations.shape[0])):
        act_img = activations[i].numpy()
        if act_img.max() == act_img.min():
            act_img = np.zeros_like(act_img)
        else:
            act_img = (act_img - act_img.min()) / (act_img.max() - act_img.min())
        act_img = (act_img * 255).astype(np.uint8)
        act_imgs.append(add_frame(Image.fromarray(act_img)))

    while len(act_imgs) < 8:
        act_imgs.append(None)

    return act_imgs

# =====================================================
# SWIMLANE / CRONOGRAMA
# =====================================================
def plot_swimlane():
    etapas = [
        "Definición del problema",
        "Análisis del dataset",
        "Preprocesamiento",
        "Entrenamiento del modelo",
        "Evaluación y ajuste",
        "Visualización y explicabilidad",
        "Despliegue y testing",
        "Documentación y presentación"
    ]
    
    inicio = [0, 1, 2, 4, 7, 9, 11, 13]
    duracion = [1, 1, 2, 3, 2, 2, 2, 1]
    colores = ['skyblue', 'lightgreen', 'orange', 'violet', 'salmon', 'cyan', 'gold', 'lightgray']
    
    fig, ax = plt.subplots(figsize=(10,5))
    for i, (etapa, start, dur, color) in enumerate(zip(etapas, inicio, duracion, colores)):
        ax.barh(i, dur, left=start, color=color, edgecolor='black')
        ax.text(start + dur/2, i, etapa, va='center', ha='center', fontsize=9, color='black', fontweight='bold')
    
    ax.set_yticks(range(len(etapas)))
    ax.set_yticklabels([])
    ax.set_xlabel("Días")
    ax.set_title("📊 Cronograma del Proyecto - Swimlane")
    ax.set_xlim(0, max([inicio[i]+duracion[i] for i in range(len(etapas))]) + 1)
    ax.invert_yaxis()
    ax.grid(axis='x', linestyle='--', alpha=0.5)
    
    return fig

# =====================================================
# INTERFAZ GRADIO COMPLETA
# =====================================================
with gr.Blocks(title="Portfolio ML: Clasificador de Dígitos MNIST") as demo:
    gr.Markdown("# 🧠 Sistema Completo de ML: Clasificador de Dígitos MNIST")
    gr.Markdown("Dibuja o sube un dígito y el sistema lo preprocesa y clasifica automáticamente.")

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
                preview_img = gr.Image(label="Imagen Preprocesada (28x28)", interactive=False, type="pil")

        with gr.Row():
            with gr.Column(scale=1):
                gr.Markdown("### ⚡ Pasos a seguir")
                btn_preprocess = gr.Button("1️⃣ Previsualizar 28x28")
                gr.Markdown("Ver cómo se procesa tu imagen")
                btn_predict = gr.Button("2️⃣ Ejecutar Predicción")
                gr.Markdown("Obtener predicción y probabilidades")
            with gr.Column(scale=1):
                gr.Markdown("### 📊 Resultados de Predicción")
                output_text = gr.Textbox(label="Salida de Predicción")
                bar_chart = gr.Plot(label="Probabilidades por dígito (%)")
                explanation_text = gr.Textbox(label="Explicación del Modelo")

        btn_preprocess.click(preprocess_preview, inputs=img_input, outputs=preview_img)
        btn_predict.click(predict_and_plot, inputs=img_input, outputs=[output_text, bar_chart, explanation_text])

    # ----- Datos & Preprocesamiento -----
    with gr.Tab("📊 Datos & Preprocesamiento"):
        gr.Markdown("## Pipeline de Preprocesamiento Paso a Paso")
        gr.Markdown("""
        Cada paso del preprocesamiento ayuda a que el modelo aprenda mejor:
        - **Grayscale**: convertimos a escala de grises para simplificar.
        - **Binarizado**: diferenciamos el dígito del fondo.
        - **Fondo invertido**: asegurar consistencia del fondo y dígito.
        - **Centrado + Redimensionado**: todos los dígitos a 28x28 píxeles y centrados.
        """)
        btn_preview_pipeline = gr.Button("Mostrar Preprocesamiento")
        preview_outputs = []
        with gr.Row():
            for i, title in enumerate([
                "1️⃣ Original", "2️⃣ Escala de Grises", "3️⃣ Binarizado", "4️⃣ Fondo Invertido", "5️⃣ Centrado + Redimensionado"
            ]):
                with gr.Column():
                    preview_outputs.append(gr.Image(label=title))
        btn_preview_pipeline.click(preprocessing_steps_preview, inputs=img_input, outputs=preview_outputs)

    # ----- Dataset -----
    # ----- Dataset MNIST -----
    # ----- Dataset ----- 
    with gr.Tab("📚 Dataset MNIST"):
        gr.Markdown("## Historia y Descripción del Dataset MNIST")
        gr.Markdown("""
            El dataset **MNIST (Modified National Institute of Standards and Technology)** es uno de los conjuntos de datos más icónicos en la historia del Machine Learning y la visión por computadora. Fue creado por **Yann LeCun** y colaboradores a partir del dataset original NIST. Contiene **70,000 imágenes de dígitos manuscritos** (0-9), de tamaño 28x28 píxeles en escala de grises, divididas en:
            
            - **60,000 imágenes de entrenamiento**  
            - **10,000 imágenes de prueba**  
            
            El dataset se diseñó para estandarizar el estudio y comparación de modelos de clasificación de dígitos. Cada imagen está centrada y normalizada, eliminando variaciones de escala y posición, lo que permitió a los investigadores enfocarse en el aprendizaje de la forma del dígito.
            
            ### Historia de MNIST y LeCun
            Yann LeCun, junto con Corinna Cortes y Christopher J.C. Burges, popularizó MNIST a fines de los años 90 como un benchmark para redes neuronales y algoritmos de reconocimiento de patrones. MNIST permitió medir el rendimiento de nuevas arquitecturas de redes neuronales de manera consistente y reproducible, lo que ayudó a consolidar el campo del Deep Learning inicial.
            
            ### Importancia en ML e IA
            MNIST es un dataset fundamental para el aprendizaje supervisado y clasificación de imágenes. Su relevancia incluye:
            
            - Evaluar y comparar nuevos algoritmos y arquitecturas de redes neuronales.  
            - Servir como banco de pruebas para técnicas de preprocesamiento, regularización y optimización.  
            - Proporcionar un entorno simple para enseñar conceptos de visión por computadora e Inteligencia Artificial.  
            - Establecer benchmarks históricos que muestran la evolución de la IA desde perceptrones simples hasta redes profundas.
            
            ### Historia de ResNet y Transfer Learning
            ResNet (Residual Networks) fue introducida por Kaiming He y colaboradores en 2015 para resolver el problema de degradación en redes muy profundas, donde añadir más capas empeoraba la precisión.  
            **ResNet18** y **ResNet50** son variantes de ResNet: ResNet18 tiene 18 capas y ResNet50 tiene 50 capas con bloques residuales que permiten el flujo de gradientes más eficiente.  
            **Transfer Learning** consiste en usar una red preentrenada en un gran dataset (como ImageNet) y ajustarla a un nuevo problema, como MNIST. Esto acelera el aprendizaje y mejora la precisión al reutilizar características generales ya aprendidas.
            
            ### Acceso al Dataset
            El dataset MNIST está disponible públicamente y se puede descargar desde varias fuentes oficiales:
            
            - Página oficial de Yann LeCun – MNIST  
            - Librerías como PyTorch o TensorFlow  
            - **Para este proyecto en concreto**, se utilizó la versión descargada desde Kaggle:  
            [Kaggle – MNIST Dataset](https://www.kaggle.com/datasets/mannu1170/minst-dataset)
                """)

    # ----- Activaciones -----
    with gr.Tab("🧠 Modelo"):
        gr.Markdown("### Visualización de Activaciones (primeras 8)")
        gr.Markdown("Cada activación muestra cómo los primeros filtros de la red responden a diferentes partes del dígito.")
        btn_activations = gr.Button("Mostrar Activaciones")
        act_outputs = []
        with gr.Row():
            for i in range(8):
                with gr.Column():
                    act_outputs.append(gr.Image(label=f"Act {i+1}"))
        btn_activations.click(visualize_activations, inputs=img_input, outputs=act_outputs)

    # ----- Deployment -----
    with gr.Tab("⚙️ Despliegue"):
        gr.Markdown("""
        - Separación de entrenamiento / inferencia
        - Carga de pesos (.pth)
        - Deploy en Hugging Face Spaces
        """)

    # ----- Ingeniería -----
    with gr.Tab("🏗 Ingeniería"):
        gr.Markdown("""
        ## 📌 Ingeniería del Proyecto
    
        En esta sección mostramos aspectos clave de la implementación, arquitectura y decisiones técnicas del proyecto MNIST:
        - Pipeline consistente y reproducible
        - Evitación de `load_learner` para mantener control sobre el preprocesado
        - Preprocesado robusto que maneja diferentes formatos de entrada
        - Control de confianza y explicabilidad de las predicciones
        """)
        
        # 🔹 Transfer Learning con ResNet18
        gr.Markdown("### 🔹 Transfer Learning con ResNet18")
        gr.Markdown(
            "Se utilizó la arquitectura ResNet18 preentrenada como extractor de características, "
            "lo que permite un aprendizaje más rápido y preciso comparado con una CNN desde cero."
        )
        gr.Image(
            value="https://huggingface.co/spaces/Santonla/mnist/resolve/main/imagenes/TransferlearningwithResNet18.png",
            label="Transfer Learning ResNet18",
            type="pil",
            interactive=False
        )
    
        # 🔹 Comparación CNN tradicional vs ResNet18
        gr.Markdown("### 🔹 Comparación CNN tradicional vs ResNet18")
        gr.Markdown(
            "ResNet18 logra mejor extracción de características profundas, evitando el problema de vanishing gradients "
            "y permitiendo capas más profundas que mejoran la precisión de clasificación."
        )
        gr.Image(
            value="https://huggingface.co/spaces/Santonla/mnist/resolve/main/imagenes/CNNvsResNet18.png",
            label="CNN vs ResNet18",
            type="pil",
            interactive=False
        )
    
        # 🔹 Fastai vs PyTorch
        gr.Markdown("### 🔹 Fastai vs PyTorch puro")
        gr.Markdown(
            "Fastai proporciona abstracciones y utilidades que simplifican la construcción del pipeline completo, "
            "mientras que PyTorch puro ofrece control más granular. Esta comparación nos ayudó a optimizar "
            "velocidad y robustez del proyecto."
        )
        gr.Image(
            value="https://huggingface.co/spaces/Santonla/mnist/resolve/main/imagenes/fastfaiv%20pytorch.png",
            label="Fastai vs PyTorch",
            type="pil",
            interactive=False
        )
    
        # 🔹 Problemas y soluciones
        gr.Markdown("### ⚠️ Problemas encontrados y cómo se resolvieron")
        gr.Markdown("""
        - Diferentes formatos de imagen de entrada: resuelto con un preprocesado robusto y centrado.
        - Activaciones que no se visualizaban correctamente: ajustando el hook a la capa correcta y manejando tensores de 3 canales.
        - Confusiones entre dígitos similares: control de confianza y explicación del top2.
        - Tiempo de inferencia y eficiencia: uso de ResNet18 y transfer learning.
        - Mantener reproducibilidad: pipeline consistente y documentación paso a paso.
        """)
        
        # 🔹 Ideas adicionales
        gr.Markdown("### 💡 Buenas prácticas destacadas")
        gr.Markdown("""
        - Mantener un pipeline modular facilita la integración de nuevas mejoras.
        - Documentar cada etapa ayuda en colaboración y reproducibilidad.
        - Visualizar activaciones y resultados intermedios aumenta la interpretabilidad.
        - Guardar pesos y resultados parciales permite volver atrás y probar ajustes sin repetir todo.
        """)
    # ----- Matriz de Confusión -----
    with gr.Tab("📉 Matriz de Confusión"):
        gr.Markdown("La matriz de confusión muestra qué dígitos son confundidos con otros por el modelo.")
        gr.Image(value="confusion_matrix.png", label="Matriz de Confusión")

    # ----- Cronograma / Swimlane -----
    with gr.Tab("⏱ Cronograma / Swimlane"):
        gr.Markdown("## 🗓 Visualización del proyecto por etapas (Swimlane)")
        gr.Markdown("Cada barra representa una etapa del proyecto, su inicio y duración aproximada.")
        swimlane_fig = gr.Plot(value=plot_swimlane(), label="Cronograma Swimlane")

demo.launch()
