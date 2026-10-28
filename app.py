
import json
import re
import html
import numpy as np
import streamlit as st
import tensorflow as tf
from PIL import Image

# --------------------------------------------------
# 1. PAGE CONFIGURATION
# --------------------------------------------------
st.set_page_config(
    page_title="Plant Disease Intelligence",
    page_icon="🌿",
    layout="wide"
)

# --------------------------------------------------
# 2. CUSTOM STYLING
# --------------------------------------------------
st.markdown("""
<style>
.block-container {
    padding-top: 2rem;
    padding-bottom: 3rem;
    max-width: 1200px;
}

.hero {
    padding: 28px;
    border-radius: 18px;
    background: linear-gradient(120deg, #123524, #237447);
    color: white;
    margin-bottom: 25px;
}

.hero h1 {
    color: white;
    margin-bottom: 8px;
}

.result-card {
    padding: 24px;
    border-radius: 16px;
    border: 1px solid #cce5d3;
    background: #f0f8f2;
    color: #163b26;
    margin: 12px 0;
}

.result-card h2, .result-card h3 {
    color: #163b26;
}

.metric-card {
    padding: 18px;
    border-radius: 14px;
    background: #f4f7f5;
    border: 1px solid #e1e8e3;
}

.section-heading {
    margin-top: 24px;
}

div.stButton > button[kind="primary"] {
    background-color: #237447;
    border-color: #237447;
}
</style>
""", unsafe_allow_html=True)

# --------------------------------------------------
# 3. LOAD MODEL AND CLASS NAMES
# --------------------------------------------------
@st.cache_resource
def load_model():
    return tf.keras.models.load_model("plant_disease_model.keras")


@st.cache_data
def load_class_names():
    with open("class_names.json", "r", encoding="utf-8") as f:
        return json.load(f)


st.markdown("""
<div class="hero">
    <h1>🌿 Plant Disease Intelligence</h1>
    <p>AI-powered plant leaf classification for smarter agriculture</p>
</div>
""", unsafe_allow_html=True)

try:
    model = load_model()
    class_names = load_class_names()
except Exception as e:
    st.error(f"Unable to load model or class names: {e}")
    st.info(
        "Ensure app.py, plant_disease_model.keras, and "
        "class_names.json are in the same project folder."
    )
    st.stop()

if len(class_names) != model.output_shape[-1]:
    st.error(
        f"Class mismatch: the model has {model.output_shape[-1]} outputs, "
        f"but class_names.json contains {len(class_names)} classes."
    )
    st.stop()

# --------------------------------------------------
# 4. IMAGE UPLOAD AND CROP SELECTION
# --------------------------------------------------
st.subheader("📷 Upload a leaf image")

uploaded_file = st.file_uploader(
    "Choose an image from your computer",
    type=["jpg", "jpeg", "png"],
    help="Use a clear image with the leaf visible."
)

crop_options = [
    "Auto-detect (all crops)",
    "Apple",
    "Blueberry",
    "Cherry",
    "Corn (maize)",
    "Grape",
    "Orange",
    "Peach",
    "Pepper",
    "Potato",
    "Raspberry",
    "Soybean",
    "Squash",
    "Strawberry",
    "Tomato"
]

selected_crop = st.selectbox(
    "Which crop is shown in the image?",
    crop_options,
    help=(
        "Choose the known crop to flag predictions for a different crop. "
        "This selection does not force the model to predict that crop."
    )
)

# --------------------------------------------------
# 5. IMAGE DISPLAY
# --------------------------------------------------
if uploaded_file is not None:

    try:
        image = Image.open(uploaded_file).convert("RGB")
    except Exception:
        st.error("This image could not be opened. Please upload a valid JPG or PNG.")
        st.stop()

    left, right = st.columns([1, 1.25], gap="large")

    with left:
        st.markdown("### Uploaded Image")
        st.image(image, use_container_width=True)
        st.caption(f"File: {uploaded_file.name}")

    with right:
        st.markdown("### 🔍 AI Analysis")
        st.write(
            "Run the model to identify the most likely class "
            "among the 38 classes it was trained on."
        )

        analyze = st.button(
            "🌱 Analyze Leaf",
            type="primary",
            use_container_width=True
        )

    # --------------------------------------------------
    # 6. PREDICTION
    # --------------------------------------------------
    if analyze:

        with st.spinner("Analyzing image..."):
            resized_image = image.resize((128, 128))

            image_array = np.asarray(
                resized_image,
                dtype=np.float32
            )

            image_array = np.expand_dims(image_array, axis=0)

            probabilities = model.predict(
                image_array,
                verbose=0
            )[0]

            top_indices = np.argsort(probabilities)[::-1][:3]

            best_index = int(top_indices[0])
            best_label = class_names[best_index]
            confidence = float(probabilities[best_index]) * 100

        # Split dataset label into crop and disease
        if "___" in best_label:
            crop_raw, disease_raw = best_label.split("___", 1)
        else:
            crop_raw = best_label
            disease_raw = "Unknown"

        predicted_crop = crop_raw.replace("_", " ").strip()
        disease_name = disease_raw.replace("_", " ").strip()

        if "healthy" in disease_raw.lower():
            status_text = "Healthy class predicted"
        else:
            status_text = "Possible disease class predicted"

        # --------------------------------------------------
        # 7. RESULTS DASHBOARD
        # --------------------------------------------------
        st.divider()
        st.markdown("## 🌿 Prediction Results")

        result_left, result_right = st.columns([1.5, 1])

        with result_left:
            safe_crop = html.escape(predicted_crop)
            safe_disease = html.escape(disease_name)

            st.markdown(
                f"""
                <div class="result-card">
                    <div style="font-size:13px; font-weight:600;">
                        TOP PREDICTION
                    </div>
                    <h2 style="margin:10px 0 5px 0;">
                        {safe_crop}
                    </h2>
                    <h3 style="margin:0 0 12px 0;">
                        {safe_disease}
                    </h3>
                    <p style="margin:0;">{status_text}</p>
                </div>
                """,
                unsafe_allow_html=True
            )

        with result_right:
            st.markdown(
                '<div class="metric-card">',
                unsafe_allow_html=True
            )
            st.metric("Model confidence", f"{confidence:.2f}%")
            st.progress(
                min(max(confidence / 100, 0.0), 1.0)
            )
            st.caption(
                "This is the model's predicted probability, "
                "not a guarantee of diagnostic accuracy."
            )
            st.markdown("</div>", unsafe_allow_html=True)

        # --------------------------------------------------
        # 8. CROP MISMATCH WARNING
        # --------------------------------------------------
        if selected_crop != "Auto-detect (all crops)":

            predicted_normalized = re.sub(
                r"[^a-z0-9]+", "", predicted_crop.lower()
            )

            selected_normalized = re.sub(
                r"[^a-z0-9]+", "", selected_crop.lower()
            )

            # Dataset crop name aliases
            aliases = {
                "cherry": ["cherryincludingsour"],
                "cornmaize": ["cornmaize"],
                "pepper": ["pepperbell"],
            }

            accepted_names = aliases.get(
                selected_normalized,
                [selected_normalized]
            )

            crop_matches = (
                predicted_normalized in accepted_names
                or selected_normalized in predicted_normalized
                or predicted_normalized in selected_normalized
            )

            if not crop_matches:
                st.error(
                    f"⚠️ Crop mismatch: you selected **{selected_crop}**, "
                    f"but the model predicted **{predicted_crop}**. "
                    "Check that the correct image was uploaded. "
                    "The model may have misclassified the image."
                )
            else:
                st.success(
                    f"The predicted crop matches your selection: "
                    f"**{selected_crop}**."
                )

        # --------------------------------------------------
        # 9. TOP 3 PREDICTIONS
        # --------------------------------------------------
        st.markdown(
            '<h3 class="section-heading">📊 Top 3 Predictions</h3>',
            unsafe_allow_html=True
        )

        for rank, index in enumerate(top_indices, start=1):
            label = class_names[int(index)]
            probability = float(probabilities[index]) * 100

            if "___" in label:
                label_crop, label_disease = label.split("___", 1)
                readable_label = (
                    label_crop.replace("_", " ")
                    + " — "
                    + label_disease.replace("_", " ")
                )
            else:
                readable_label = label.replace("_", " ")

            st.write(
                f"**{rank}. {readable_label}** — "
                f"{probability:.2f}%"
            )
            st.progress(
                min(max(probability / 100, 0.0), 1.0)
            )

        # --------------------------------------------------
        # 10. SUGGESTED NEXT STEPS
        # --------------------------------------------------
        st.markdown("### 🌱 Suggested Next Steps")

        if "healthy" in disease_raw.lower():
            st.success(
                "The model classified the image as healthy. "
                "Continue routine crop monitoring."
            )
        else:
            st.warning(
                "The model detected a possible disease class. "
                "Inspect the plant and compare visible symptoms. "
                "Seek local agricultural advice before applying treatment."
            )

        if confidence < 60:
            st.info(
                "The top prediction has relatively low confidence. "
                "Try a clearer image with good lighting and a visible leaf."
            )

        st.caption(
            "AI screening only: this model always chooses from its trained "
            "classes and may misclassify unfamiliar images, fruits, or "
            "non-leaf objects. Verify the result before taking action."
        )

else:
    st.info(
        "👆 Upload a leaf image, select the crop if known, "
        "and click **Analyze Leaf**."
    )

# --------------------------------------------------
# 11. FOOTER
# --------------------------------------------------
st.divider()
st.caption(
    "Plant Disease Intelligence | CNN-based academic project"
)
