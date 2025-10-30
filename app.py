import streamlit as st
from transformers import VisionEncoderDecoderModel, ViTImageProcessor, AutoTokenizer
from PIL import Image
import torch
import torchvision
import torchvision.transforms as T
import numpy as np

# =================== PAGE CONFIG ===================
st.set_page_config(
    page_title="AI Image Captioning & Segmentation",
    page_icon="🖼️",
    layout="wide"
)

# =================== CUSTOM STYLE ===================
st.markdown("""
    <style>
        .caption-box {
            background-color: #fff8e1;
            padding: 15px;
            border-radius: 10px;
            border-left: 5px solid #ffb300;
            font-size: 18px;
            color: #5d4037;
            font-weight: bold;
        }
        .seg-title {
            background-color: #f1f8e9;
            padding: 8px;
            border-radius: 5px;
            color: #33691e;
            font-weight: bold;
            text-align: center;
        }
    </style>
""", unsafe_allow_html=True)

# =================== LOAD MODELS ===================
@st.cache_resource
def load_models():
    # Lightweight & Streamlit-friendly captioning model
    caption_model = VisionEncoderDecoderModel.from_pretrained("nlpconnect/vit-gpt2-image-captioning")
    feature_extractor = ViTImageProcessor.from_pretrained("nlpconnect/vit-gpt2-image-captioning")
    tokenizer = AutoTokenizer.from_pretrained("nlpconnect/vit-gpt2-image-captioning")

    # Pretrained segmentation model (COCO)
    seg_model = torchvision.models.detection.maskrcnn_resnet50_fpn(pretrained=True)
    seg_model.eval()

    return caption_model, feature_extractor, tokenizer, seg_model

caption_model, feature_extractor, tokenizer, seg_model = load_models()
transform = T.Compose([T.ToTensor()])

# =================== FUNCTIONS ===================
def generate_caption(image):
    pixel_values = feature_extractor(images=image, return_tensors="pt").pixel_values
    # ✅ FIX: Removed beam search (NotImplementedError issue)
    output_ids = caption_model.generate(pixel_values, max_length=16)
    caption = tokenizer.decode(output_ids[0], skip_special_tokens=True).strip()
    return caption

def get_segmented_image(image, threshold=0.5):
    img_tensor = transform(image)
    with torch.no_grad():
        prediction = seg_model([img_tensor])
    masks = prediction[0]['masks']
    scores = prediction[0]['scores']
    img_np = np.array(image)

    for i in range(len(masks)):
        if scores[i] > threshold:
            mask = masks[i, 0].mul(255).byte().cpu().numpy()
            color = np.random.randint(0, 255, (3,), dtype=np.uint8)
            img_np[mask > 128] = img_np[mask > 128] * 0.5 + color * 0.5
    return img_np

# =================== STREAMLIT UI ===================
st.title("🖼️ AI Image Captioning & Segmentation")
st.write("Upload an image & let AI generate a caption and highlight objects!")

uploaded_file = st.file_uploader("📤 Upload an image", type=["jpg", "jpeg", "png"])

if uploaded_file:
    image = Image.open(uploaded_file).convert("RGB")
    st.image(image, caption="Uploaded Image", use_column_width=True)

    if st.button("✨ Generate"):
        with st.spinner("⏳ Processing..."):
            caption = generate_caption(image)
            seg_image = get_segmented_image(image)

        st.markdown(f'<p class="caption-box">📝 Caption: {caption}</p>', unsafe_allow_html=True)
        st.image(seg_image, caption="🎨 Segmented Objects", use_column_width=True)
