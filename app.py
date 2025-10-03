# optimized_app_with_download_and_preview.py
import os
import io
import cv2
import tempfile
import gradio as gr
import numpy as np
from PIL import Image
import torch
import torchvision.models as models

# SAM
from segment_anything import sam_model_registry, SamPredictor, SamAutomaticMaskGenerator

# CLIP (for text-to-region similarity)
from transformers import CLIPProcessor, CLIPModel

# -----------------------------
# Config / Tuning
# -----------------------------
SAM_CHECKPOINT = "sam_vit_b_01ec64.pth"  # Put the file in the same folder
SAM_MODEL_TYPE = "vit_b"
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

PRELOAD_MODELS_AT_STARTUP = True   # Preload models at startup (reduces first-interaction lag)
FAST_MASK_SETTINGS = False        # Faster (coarser) SAM masks

CLIP_BATCH_SIZE = 64
USE_SMALL_CLASSIFIER = True  # faster classifier for name prediction

# -----------------------------
# Utilities
# -----------------------------
def pil_to_np_rgb(img_pil: Image.Image) -> np.ndarray:
    return cv2.cvtColor(np.array(img_pil.convert("RGB")), cv2.COLOR_RGB2BGR)

def np_to_pil_rgb(img_np_bgr: np.ndarray) -> Image.Image:
    return Image.fromarray(cv2.cvtColor(img_np_bgr, cv2.COLOR_BGR2RGB))

def overlay_mask_on_image(image_bgr, mask, alpha=0.45):
    """Overlay mask and draw mask contours."""
    color = np.random.randint(0, 256, size=(3,), dtype=np.uint8)
    overlay = image_bgr.copy()
    overlay[mask > 0] = (overlay[mask > 0] * (1 - alpha) + color * alpha).astype(np.uint8)
    cnts, _ = cv2.findContours(mask.astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    cv2.drawContours(overlay, cnts, -1, (255, 255, 255), 2)
    return overlay

def crop_by_mask(image_bgr, mask, pad=6):
    ys, xs = np.where(mask > 0)
    if len(xs) == 0 or len(ys) == 0:
        return None
    x0, x1 = max(xs.min()-pad, 0), min(xs.max()+pad, image_bgr.shape[1]-1)
    y0, y1 = max(ys.min()-pad, 0), min(ys.max()+pad, image_bgr.shape[0]-1)
    return image_bgr[y0:y1+1, x0:x1+1], (x0, y0, x1, y1)

def resize_point_to_model_space(point_xy, orig_w, orig_h, img_w, img_h):
    return np.array(point_xy, dtype=np.float32)

# -----------------------------
# Model loaders (same as earlier optimized version)
# -----------------------------
def load_sam_and_helpers(fast: bool = True):
    assert os.path.exists(SAM_CHECKPOINT), f"SAM checkpoint not found: {SAM_CHECKPOINT}"
    sam = sam_model_registry[SAM_MODEL_TYPE](checkpoint=SAM_CHECKPOINT)
    sam.to(device=DEVICE)
    predictor = SamPredictor(sam)
    if fast:
        mask_generator = SamAutomaticMaskGenerator(
            model=sam,
            points_per_side=16,
            pred_iou_thresh=0.78,
            stability_score_thresh=0.78,
            crop_n_layers=0,
            crop_n_points_downscale_factor=4,
            min_mask_region_area=800
        )
    else:
        mask_generator = SamAutomaticMaskGenerator(
            model=sam,
            points_per_side=32,
            pred_iou_thresh=0.86,
            stability_score_thresh=0.92,
            crop_n_layers=1,
            crop_n_points_downscale_factor=2,
            min_mask_region_area=200
        )
    return predictor, mask_generator

def load_clip():
    model = CLIPModel.from_pretrained("openai/clip-vit-base-patch32").to(DEVICE)
    proc = CLIPProcessor.from_pretrained("openai/clip-vit-base-patch32")
    return model, proc

def load_small_classifier():
    weights = models.MobileNet_V3_Small_Weights.IMAGENET1K_V1
    model = models.mobilenet_v3_small(weights=weights).to(DEVICE)
    model.eval()
    transforms = weights.transforms()
    return model, transforms

def load_resnet():
    weights = models.ResNet50_Weights.IMAGENET1K_V2
    model = models.resnet50(weights=weights).to(DEVICE)
    model.eval()
    transforms = weights.transforms()
    return model, transforms

def load_imagenet_labels_from_weights(weights):
    return weights.meta["categories"]

# -----------------------------
# Globals and optional preload
# -----------------------------
_PREDICTOR, _MASK_GEN = None, None
_CLIP_MODEL, _CLIP_PROC = None, None
_CLASSIFIER, _CLS_TRANS = None, None
_IMAGENET_LABELS = None

if PRELOAD_MODELS_AT_STARTUP:
    print("Preloading models (may take time)...")
    _PREDICTOR, _MASK_GEN = load_sam_and_helpers(fast=FAST_MASK_SETTINGS)
    _CLIP_MODEL, _CLIP_PROC = load_clip()
    if USE_SMALL_CLASSIFIER:
        _CLASSIFIER, _CLS_TRANS = load_small_classifier()
        _IMAGENET_LABELS = load_imagenet_labels_from_weights(models.MobileNet_V3_Small_Weights.IMAGENET1K_V1)
    else:
        _CLASSIFIER, _CLS_TRANS = load_resnet()
        _IMAGENET_LABELS = load_imagenet_labels_from_weights(models.ResNet50_Weights.IMAGENET1K_V2)
    print("Preloading finished.")

# -----------------------------
# Segmentation functions
# -----------------------------
def segment_point(image_pil, click_xy):
    image_bgr = pil_to_np_rgb(image_pil)
    h, w = image_bgr.shape[:2]
    predictor = _PREDICTOR
    predictor.set_image(cv2.cvtColor(image_bgr, cv2.COLOR_BGR2RGB))
    pt = resize_point_to_model_space(click_xy, w, h, w, h).reshape(1, 2)
    labels = np.array([1])
    masks, scores, _ = predictor.predict(point_coords=pt, point_labels=labels, multimask_output=True)
    best_idx = int(np.argmax(scores))
    mask = masks[best_idx].astype(np.uint8)
    # overlay and also draw the point marker
    over = overlay_mask_on_image(image_bgr, mask)
    cx, cy = int(click_xy[0]), int(click_xy[1])
    cv2.circle(over, (cx, cy), max(6, round(min(w,h)*0.01)), (255,255,255), 2)
    return np_to_pil_rgb(over), mask

def segment_box(image_pil, tl_xy, br_xy):
    image_bgr = pil_to_np_rgb(image_pil)
    predictor = _PREDICTOR
    predictor.set_image(cv2.cvtColor(image_bgr, cv2.COLOR_BGR2RGB))
    x0, y0 = tl_xy
    x1, y1 = br_xy
    x0, x1 = int(min(x0, x1)), int(max(x0, x1))
    y0, y1 = int(min(y0, y1)), int(max(y0, y1))
    box = np.array([x0, y0, x1, y1])
    masks, scores, _ = predictor.predict(point_coords=None, point_labels=None, box=box[None, :], multimask_output=True)
    best_idx = int(np.argmax(scores))
    mask = masks[best_idx].astype(np.uint8)
    over = overlay_mask_on_image(image_bgr, mask)
    # draw the bounding rectangle for clarity
    cv2.rectangle(over, (x0, y0), (x1, y1), (255,255,255), 2)
    return np_to_pil_rgb(over), mask

def segment_text(image_pil, text_query):
    image_bgr = pil_to_np_rgb(image_pil)
    masks_info = _MASK_GEN.generate(cv2.cvtColor(image_bgr, cv2.COLOR_BGR2RGB))
    if len(masks_info) == 0:
        return image_pil, None

    global _CLIP_MODEL, _CLIP_PROC
    if _CLIP_MODEL is None or _CLIP_PROC is None:
        _CLIP_MODEL, _CLIP_PROC = load_clip()
    clip_model, clip_proc = _CLIP_MODEL, _CLIP_PROC

    crops = []
    masks = []
    for mi in masks_info:
        mask = mi["segmentation"].astype(np.uint8)
        crop_res = crop_by_mask(image_bgr, mask)
        if crop_res is None:
            continue
        crop, _ = crop_res
        pil_crop = np_to_pil_rgb(crop).resize((224,224))
        crops.append(pil_crop)
        masks.append(mask)

    if len(crops) == 0:
        return image_pil, None

    best_score = -1e9
    best_mask = None
    text_input = clip_proc(text=[text_query], return_tensors="pt", padding=True).to(DEVICE)
    with torch.no_grad():
        text_emb = clip_model.get_text_features(**text_input)
        text_emb = text_emb / text_emb.norm(dim=-1, keepdim=True)
        for i in range(0, len(crops), CLIP_BATCH_SIZE):
            batch = crops[i:i+CLIP_BATCH_SIZE]
            inputs = clip_proc(images=batch, return_tensors="pt").to(DEVICE)
            img_embs = clip_model.get_image_features(**inputs)
            img_embs = img_embs / img_embs.norm(dim=-1, keepdim=True)
            sims = (img_embs @ text_emb.T).squeeze(-1).cpu().numpy()
            for k, s in enumerate(sims):
                if s > best_score:
                    best_score = float(s)
                    best_mask = masks[i + k]

    if best_mask is None:
        return image_pil, None

    over = overlay_mask_on_image(image_bgr, best_mask)
    return np_to_pil_rgb(over), best_mask

# -----------------------------
# Naming (classification)
# -----------------------------
def name_mask(image_pil, mask):
    if mask is None:
        return "—", None
    image_bgr = pil_to_np_rgb(image_pil)
    crop_res = crop_by_mask(image_bgr, mask)
    if crop_res is None:
        return "—", None
    crop, _bbox = crop_res
    global _CLASSIFIER, _CLS_TRANS, _IMAGENET_LABELS
    if _CLASSIFIER is None:
        if USE_SMALL_CLASSIFIER:
            _CLASSIFIER, _CLS_TRANS = load_small_classifier()
            _IMAGENET_LABELS = load_imagenet_labels_from_weights(models.MobileNet_V3_Small_Weights.IMAGENET1K_V1)
        else:
            _CLASSIFIER, _CLS_TRANS = load_resnet()
            _IMAGENET_LABELS = load_imagenet_labels_from_weights(models.ResNet50_Weights.IMAGENET1K_V2)
    pil_crop = np_to_pil_rgb(crop).resize((224,224))
    with torch.no_grad():
        inp = _CLS_TRANS(pil_crop).unsqueeze(0).to(DEVICE)
        logits = _CLASSIFIER(inp)
        probs = torch.softmax(logits, dim=1)
        top5 = torch.topk(probs, k=5, dim=1)
    idxs = top5.indices.squeeze(0).tolist()
    scores = top5.values.squeeze(0).tolist()
    labels = _IMAGENET_LABELS
    preds = [(labels[i], float(scores[k])) for k,i in enumerate(idxs)]
    best = f"{preds[0][0]} ({preds[0][1]*100:.1f}%)"
    return best, preds

# -----------------------------
# Helper: produce files for download
# -----------------------------
def save_mask_and_object_files(image_pil, mask, download_mode):
    """
    download_mode: "mask" (mask file),
                   "object_on_black" (object colored on black background),
                   "object_cropped_transparent" (cropped RGBA)
    Returns (mask_path, object_path, mask_preview, object_preview)
    """
    image_bgr = pil_to_np_rgb(image_pil)
    
    # Create mask preview and file
    mask_img = Image.fromarray((mask * 255).astype(np.uint8))
    # Convert to RGB for better preview (white mask on black background)
    mask_preview = mask_img.convert("RGB")
    
    tmp_mask = tempfile.NamedTemporaryFile(suffix=".png", delete=False)
    mask_img.save(tmp_mask, format="PNG")
    tmp_mask.flush(); tmp_mask.close()
    mask_path = tmp_mask.name

    # Create object preview and file
    object_path = None
    object_preview = None
    
    if download_mode == "object_on_black":
        out = np.zeros_like(image_bgr)
        out[mask > 0] = image_bgr[mask > 0]
        pil_obj = np_to_pil_rgb(out)
        object_preview = pil_obj
        
        tmp_obj = tempfile.NamedTemporaryFile(suffix=".png", delete=False)
        pil_obj.save(tmp_obj, format="PNG")
        tmp_obj.flush(); tmp_obj.close()
        object_path = tmp_obj.name

    elif download_mode == "object_cropped_transparent":
        crop_res = crop_by_mask(image_bgr, mask)
        if crop_res is not None:
            crop, (x0, y0, x1, y1) = crop_res
            h, w = crop.shape[:2]
            # alpha channel from mask crop
            mask_crop = mask[y0:y1+1, x0:x1+1]
            rgba = cv2.cvtColor(crop, cv2.COLOR_BGR2RGBA)
            rgba[..., 3] = (mask_crop * 255).astype(np.uint8)
            pil_rgba = Image.fromarray(rgba)
            object_preview = pil_rgba
            
            tmp_obj = tempfile.NamedTemporaryFile(suffix=".png", delete=False)
            pil_rgba.save(tmp_obj, format="PNG")
            tmp_obj.flush(); tmp_obj.close()
            object_path = tmp_obj.name

    return mask_path, object_path, mask_preview, object_preview

# -----------------------------
# Preview function: draws the current selection (point/box) so user sees it BEFORE segmentation
# -----------------------------
def preview_selection(image, clicks, mode):
    if image is None:
        return None
    img_bgr = pil_to_np_rgb(image)
    out = img_bgr.copy()
    clicks = clicks or []
    if mode == "Point":
        # show last point if exists
        if len(clicks) > 0:
            cx, cy = int(clicks[-1][0]), int(clicks[-1][1])
            cv2.circle(out, (cx, cy), max(6, round(min(out.shape[:2])*0.01)), (0,255,0), 3)
    elif mode == "Box":
        if len(clicks) == 1:
            cx, cy = int(clicks[0][0]), int(clicks[0][1])
            cv2.circle(out, (cx, cy), 6, (0,255,0), 3)
        elif len(clicks) >= 2:
            x0, y0 = clicks[0]
            x1, y1 = clicks[1]
            x0, x1 = int(min(x0,x1)), int(max(x0,x1))
            y0, y1 = int(min(y0,y1)), int(max(y0,y1))
            cv2.rectangle(out, (x0, y0), (x1, y1), (0,255,0), 3)
    return np_to_pil_rgb(out)

# -----------------------------
# Gradio UI
# -----------------------------
# Custom CSS for better styling and animations
custom_css = """
.gradio-container {
    max-width: 1200px !important;
    margin: auto !important;
    background: linear-gradient(135deg, #f5f7fa 0%, #c3cfe2 100%) !important;
    color: black !important;
}

.main-header {
    text-align: center;
    background: white;
    padding: 2rem;
    border-radius: 15px;
    box-shadow: 0 4px 15px rgba(0,0,0,0.1);
    margin-bottom: 2rem;
    color: black !important;
}

.control-panel {
    background: white;
    padding: 1.5rem;
    border-radius: 15px;
    box-shadow: 0 4px 15px rgba(0,0,0,0.1);
    margin: 1rem 0;
    color: black !important;
}

.image-container {
    background: white;
    padding: 1rem;
    border-radius: 15px;
    box-shadow: 0 4px 15px rgba(0,0,0,0.1);
    color: black !important;
}

.button-container {
    display: flex;
    gap: 10px;
    flex-wrap: wrap;
    justify-content: center;
    margin: 1rem 0;
}

.primary-btn {
    background: linear-gradient(45deg, #667eea 0%, #764ba2 100%) !important;
    border: none !important;
    border-radius: 25px !important;
    padding: 12px 30px !important;
    color: white !important;
    font-weight: 600 !important;
    transition: all 0.3s ease !important;
    box-shadow: 0 4px 15px rgba(102, 126, 234, 0.4) !important;
}

.primary-btn:hover {
    transform: translateY(-2px) !important;
    box-shadow: 0 6px 20px rgba(102, 126, 234, 0.6) !important;
}

.secondary-btn {
    background: white !important;
    border: 2px solid #667eea !important;
    border-radius: 25px !important;
    padding: 10px 25px !important;
    color: #667eea !important;
    font-weight: 500 !important;
    transition: all 0.3s ease !important;
}

.secondary-btn:hover {
    background: #667eea !important;
    color: white !important;
    transform: translateY(-2px) !important;
}

.result-container {
    background: white;
    padding: 1.5rem;
    border-radius: 15px;
    box-shadow: 0 4px 15px rgba(0,0,0,0.1);
    margin: 1rem 0;
    color: black !important;
}

.download-header {
    text-align: center;
    background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
    color: white !important;
    padding: 1.5rem;
    border-radius: 15px;
    margin: 2rem 0 1rem 0;
    box-shadow: 0 4px 15px rgba(102, 126, 234, 0.3);
}

.download-section {
    background: transparent;
    padding: 0;
    margin: 1rem 0;
    gap: 2rem;
}

.download-card {
    background: white;
    padding: 2rem;
    border-radius: 20px;
    box-shadow: 0 8px 25px rgba(0,0,0,0.15);
    border: 1px solid #e0e7ff;
    transition: all 0.3s ease;
    margin: 0.5rem;
}

.download-card:hover {
    transform: translateY(-5px);
    box-shadow: 0 12px 35px rgba(0,0,0,0.2);
}

.card-title {
    text-align: center;
    color: #1f2937 !important;
    font-weight: 700;
    margin-bottom: 1rem;
    padding-bottom: 0.5rem;
    border-bottom: 2px solid #e0e7ff;
}

.preview-image {
    border-radius: 15px;
    border: 3px solid #e0e7ff;
    margin: 1rem 0;
    box-shadow: 0 4px 15px rgba(0,0,0,0.1);
    background: #f8f9ff;
}

.download-button-row {
    margin-top: 1.5rem;
    justify-content: center;
}

.download-btn {
    background: linear-gradient(45deg, #10b981 0%, #059669 100%) !important;
    border: none !important;
    border-radius: 15px !important;
    padding: 1rem 2rem !important;
    color: white !important;
    font-weight: 600 !important;
    transition: all 0.3s ease !important;
    box-shadow: 0 4px 15px rgba(16, 185, 129, 0.4) !important;
    text-align: center;
    width: 100%;
}

.download-btn:hover {
    transform: translateY(-2px) !important;
    box-shadow: 0 6px 20px rgba(16, 185, 129, 0.6) !important;
    background: linear-gradient(45deg, #059669 0%, #047857 100%) !important;
}

.fade-in {
    animation: fadeIn 0.5s ease-in;
}

@keyframes fadeIn {
    from { opacity: 0; transform: translateY(20px); }
    to { opacity: 1; transform: translateY(0); }
}

.pulse {
    animation: pulse 2s infinite;
}

@keyframes pulse {
    0% { transform: scale(1); }
    50% { transform: scale(1.05); }
    100% { transform: scale(1); }
}

/* Radio and checkbox styling */
.radio-group, .checkbox-group {
    background: #f8f9ff;
    padding: 1rem;
    border-radius: 12px;
    border: 2px solid #e0e7ff;
    color: black !important;
}

/* File upload styling */
.file-upload {
    border: 2px dashed #667eea;
    border-radius: 12px;
    background: #f8f9ff;
    transition: all 0.3s ease;
    color: black !important;
}

.file-upload:hover {
    border-color: #4f46e5;
    background: #eef2ff;
}

/* Global text color override */
* {
    color: black !important;
}

/* Ensure all text elements are black */
h1, h2, h3, h4, h5, h6, p, span, div, label {
    color: black !important;
}

/* Markdown text */
.markdown {
    color: black !important;
}
"""

with gr.Blocks(
    title="🎨 SAM Image Segmentation Studio", 
    css=custom_css,
    theme=gr.themes.Soft(
        primary_hue="blue",
        secondary_hue="slate",
        neutral_hue="slate",
        font=gr.themes.GoogleFont("Inter")
    )
) as demo:
    
    # Header
    with gr.Row(elem_classes="main-header"):
        gr.Markdown("""
        # 🎨 SAM Image Segmentation Studio
        ### Advanced AI-powered object segmentation with point and box selection
        Upload an image, select your preferred segmentation method, and let AI do the magic! ✨
        """)
    
    # Main content area
    with gr.Row():
        # Left column - Image input
        with gr.Column(scale=2, elem_classes="image-container"):
            img_in = gr.Image(
                type="pil", 
                label="📤 Upload Your Image", 
                interactive=True,
                elem_classes="file-upload"
            )
            
        # Right column - Controls
        with gr.Column(scale=1, elem_classes="control-panel"):
            gr.Markdown("### 🛠️ Control Panel")
            
            mode = gr.Radio(
                ["Point", "Box"], 
                value="Point", 
                label="🎯 Selection Method",
                elem_classes="radio-group"
            )
            
            do_name = gr.Checkbox(
                value=True, 
                label="🏷️ Enable object name prediction",
                elem_classes="checkbox-group"
            )
            
            with gr.Row(elem_classes="button-container"):
                seg_btn = gr.Button(
                    "🚀 Segment Object", 
                    variant="primary",
                    elem_classes="primary-btn pulse"
                )
            
            with gr.Row(elem_classes="button-container"):    
                preview_btn = gr.Button(
                    "👁️ Preview Selection", 
                    elem_classes="secondary-btn"
                )
                reset_btn = gr.Button(
                    "🔄 Reset", 
                    elem_classes="secondary-btn"
                )

    # Results section
    with gr.Row(elem_classes="result-container fade-in"):
        img_out = gr.Image(
            type="pil", 
            label="✨ Segmentation Result",
            elem_classes="fade-in"
        )
    
    # Prediction results
    with gr.Row(elem_classes="result-container"):
        with gr.Column():
            gr.Markdown("### 🤖 AI Prediction Results")
            name_out = gr.Markdown(
                "Ready to analyze your selection...",
                elem_classes="fade-in"
            )
    
    # Download section - Enhanced design
    with gr.Row():
        gr.Markdown("## 📥 Download Your Results", elem_classes="download-header")
    
    with gr.Row(elem_classes="download-section"):
        # Mask preview and download
        with gr.Column(elem_classes="download-card"):
            with gr.Row():
                gr.Markdown("### 🎭 **Segmentation Mask**", elem_classes="card-title")
            
            mask_preview = gr.Image(
                type="pil",
                label="",
                interactive=False,
                elem_classes="preview-image fade-in",
                height=300,
                show_label=False
            )
            
            with gr.Row(elem_classes="download-button-row"):
                mask_file = gr.File(
                    label="📥 Download Mask File", 
                    elem_classes="download-btn fade-in"
                )
        
        # Spacer
        with gr.Column(scale=0.1):
            gr.HTML("<div style='width: 2rem;'></div>")
        
        # Object preview and download
        with gr.Column(elem_classes="download-card"):
            with gr.Row():
                gr.Markdown("### 🖼️ **Extracted Object**", elem_classes="card-title")
            
            obj_preview = gr.Image(
                type="pil",
                label="",
                interactive=False,
                elem_classes="preview-image fade-in",
                height=300,
                show_label=False
            )
            
            with gr.Row(elem_classes="download-button-row"):
                obj_file = gr.File(
                    label="📥 Download Object File", 
                    elem_classes="download-btn fade-in"
                )

    clicks_state = gr.State([])

    # update clicks_state when user clicks the image (select event gives coordinates as evt.index)
    def on_click(evt: gr.SelectData, clicks):
        if evt is None or evt.index is None:
            return clicks
        x, y = evt.index
        clicks = clicks or []
        # For box mode we store up to 2 clicks; for point mode we keep last click only
        if len(clicks) >= 2:
            clicks = []
        clicks.append((float(x), float(y)))
        return clicks

    def on_reset():
        return []

    # Preview button shows the selection marker(s)
    preview_btn.click(preview_selection, inputs=[img_in, clicks_state, mode], outputs=[img_out])

    # Main segment button: returns overlay, name text, mask file path, object file path
    def run_segment(image, mode, want_name, clicks):
        download_mode = "object_on_black"  # Default download mode
        text_query = ""  # Default empty text query
        global _PREDICTOR, _MASK_GEN, _CLIP_MODEL, _CLIP_PROC
        if image is None:
            return None, "Upload an image first.", None, None, None, None

        if _PREDICTOR is None or _MASK_GEN is None:
            _PREDICTOR, _MASK_GEN = load_sam_and_helpers(fast=FAST_MASK_SETTINGS)
        if _CLIP_MODEL is None or _CLIP_PROC is None:
            _CLIP_MODEL, _CLIP_PROC = load_clip()

        mask = None
        overlay = None
        try:
            if mode == "Point":
                if not clicks:
                    return None, "Point mode: click once on the object, then press Segment.", None, None, None, None
                overlay, mask = segment_point(image, clicks[-1])

            elif mode == "Box":
                if not clicks or len(clicks) < 2:
                    return None, "Box mode: click two corners, then press Segment.", None, None, None, None
                overlay, mask = segment_box(image, clicks[0], clicks[1])

            elif mode == "Text":
                if not text_query or text_query.strip() == "":
                    return None, "Text mode: enter a word/phrase and press Segment.", None, None, None, None
                overlay, mask = segment_text(image, text_query.strip())

            name_md, mask_path, obj_path, mask_preview, obj_preview = "—", None, None, None, None
            if mask is not None:
                if want_name:
                    best, preds = name_mask(image, mask)
                    name_md = f"*Top-1:* {best}\n\n<details><summary>Top-5</summary>\n\n" + \
                              "\n".join([f"- {lbl}: {p*100:.1f}%" for lbl, p in preds]) + "\n\n</details>"

                mask_path, obj_path, mask_preview, obj_preview = save_mask_and_object_files(image, mask, download_mode)

            return overlay, name_md, mask_path, obj_path, mask_preview, obj_preview

        except Exception as e:
            return None, f"Error during segmentation: {e}", None, None, None, None

    # wiring
    img_in.select(on_click, inputs=[clicks_state], outputs=[clicks_state])
    reset_btn.click(on_reset, outputs=[clicks_state])
    seg_btn.click(run_segment, inputs=[img_in, mode, do_name, clicks_state],
                  outputs=[img_out, name_out, mask_file, obj_file, mask_preview, obj_preview])

if __name__ == "__main__":
    demo.launch()
