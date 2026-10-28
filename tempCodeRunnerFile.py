def load_class_names():
    with open("class_names.json", "r") as file:
        return json.load(file)

st.title("🌿 Plant Disease Detection")
st.write("Upload a leaf image to predict its disease.")

try:
    model = load_model()
    class_names = load_class_names()
except Exception as e:
    st.error(f"Could not load model or class names: {e}")
    st.stop()

uploaded_file = st.file_uploader(
    "Choose a leaf image",
    type=["jpg", "jpeg", "png"]
)

if uploaded_file is not None:
    image = Image.open(uploaded_file).convert("RGB")
    st.image(image, caption="Uploaded leaf", width=350)

    if st.button("Predict Disease"):
        with st.spinner("Analyzing leaf image..."):
            image = image.resize((128, 128))
            image_array = np.array(image, dtype=np.float32)
            image_array = np.expand_dims(image_array, axis=0)

            predictions = model.predict(image_array, verbose=0)[0]
            predicted_index = int(np.argmax(predictions))
            confidence = float(predictions[predicted_index]) * 100

        st.subheader("Prediction")
        st.write(f"**Class:** {class_names[predicted_index]}")
        st.write(f"**Model confidence:** {confidence:.2f}%")

        if confidence < 60:
            st.warning(
                "The prediction is uncertain. Try a clearer leaf image."
            )

        st.caption(
            "This is an AI prediction, not a confirmed agricultural diagnosis."
        )
