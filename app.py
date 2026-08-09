import streamlit as st
import tensorflow as tf
import numpy as np
import cv2
from PIL import Image



# PAGE CONFIGURATION

st.set_page_config(
    page_title="Plant Disease Detection",
    page_icon="🌿",
    layout="centered"
)



# TITLE
st.title("Plant Disease Detection")

st.write(
    "AI-powered plant disease classification using "
    "Transfer Learning with EfficientNetB0"
)



# LOAD MODEL
@st.cache_resource
def load_model():

    model = tf.keras.models.load_model(
        "plant_disease_detection.keras"
    )

    return model


model = load_model()

st.success("Model loaded successfully!")



# LOAD CLASS NAMES

@st.cache_data
def load_class_names():

    with open("labels.txt", "r") as f:

        class_names = [
            line.strip()
            for line in f.readlines()
        ]

    return class_names


class_names = load_class_names()

st.write(
    f"Number of classes: {len(class_names)}"
)



# GRAD-CAM FUNCTION

def make_gradcam_heatmap(image_array, model):

    
    # 1. Get EfficientNetB0

    base_model = model.get_layer(
        "efficientnetb0"
    )
    
    # 2. Get the final convolutional layer
    last_conv_layer = base_model.get_layer(
        "top_conv"
    )

    # 3. Create model for Grad-CAM

    grad_model = tf.keras.models.Model(

        inputs=base_model.input,

        outputs=[
            last_conv_layer.output,
            base_model.output
        ]
    )


    # 4. Pass image through augmentation layer

    x = image_array

    preprocessing_layer = model.get_layer(
        "sequential"
    )

    x = preprocessing_layer(
        x,
        training=False
    )

    # 5. Gradient Tape

    with tf.GradientTape() as tape:

        # EfficientNet forward pass

        conv_outputs, base_output = grad_model(
            x,
            training=False
        )


        # Classification head

        prediction = base_output


        # Layers after EfficientNetB0:
        #
        # 3 → GlobalAveragePooling2D
        # 4 → Dropout
        # 5 → Dense

        for layer in model.layers[3:]:

            prediction = layer(
                prediction,
                training=False
            )


        # Predicted class

        predicted_class = tf.argmax(
            prediction[0]
        )

   
        # Probability of predicted class

        class_channel = prediction[
            :,
            predicted_class
        ]


    # 6. Calculate gradients

    grads = tape.gradient(
        class_channel,
        conv_outputs
    )


    if grads is None:

        raise ValueError(
            "Gradients could not be calculated."
        )

    # 7. Average gradients

    pooled_grads = tf.reduce_mean(
        grads,
        axis=(0, 1, 2)
    )


    # 8. Remove batch dimension

    conv_outputs = conv_outputs[0]


    # 9. Weight feature map

    heatmap = tf.reduce_sum(
        conv_outputs * pooled_grads,
        axis=-1
    )


    
    # 10. ReLU
    

    heatmap = tf.maximum(
        heatmap,
        0
    )


    # 11. Normalize heatmap

    max_value = tf.reduce_max(
        heatmap
    )

    heatmap = heatmap / (
        max_value + 1e-8
    )


    return heatmap.numpy()


# CREATE GRAD-CAM OVERLAY
def create_gradcam_overlay(
    original_image,
    heatmap
):

    # Convert PIL → NumPy

    image = np.array(
        original_image
    )


    # RGB → BGR

    image_bgr = cv2.cvtColor(
        image,
        cv2.COLOR_RGB2BGR
    )


    # Resize heatmap

    heatmap = cv2.resize(
        heatmap,
        (
            image_bgr.shape[1],
            image_bgr.shape[0]
        )
    )


    # Convert heatmap to 0–255

    heatmap = np.uint8(
        255 * heatmap
    )


    # Apply JET colormap

    heatmap_color = cv2.applyColorMap(
        heatmap,
        cv2.COLORMAP_JET
    )


    # Overlay heatmap on original image

    overlay = cv2.addWeighted(
        image_bgr,
        0.6,
        heatmap_color,
        0.4,
        0
    )


    # BGR → RGB

    overlay = cv2.cvtColor(
        overlay,
        cv2.COLOR_BGR2RGB
    )


    return overlay



# UPLOAD SECTION


st.divider()

st.subheader(" Upload a Plant Leaf")

uploaded_file = st.file_uploader(
    "Choose a leaf image",
    type=[
        "jpg",
        "jpeg",
        "png"
    ]
)



# WHEN IMAGE IS UPLOADED

if uploaded_file is not None:

    # Open image

    image = Image.open(
        uploaded_file
    ).convert("RGB")


    # Display original image

    st.image(
        image,
        caption="Uploaded Leaf",
        use_container_width=True
    )


 
    # Resize image

    img = image.resize(
        (224, 224)
    )

  
    # Convert to NumPy

    img_array = np.array(
        img
    ).astype("float32")

    # Add batch dimension

    img_array = np.expand_dims(
        img_array,
        axis=0
    )


    # PREDICTION

    prediction = model.predict(
        img_array,
        verbose=0
    )


    # Highest probability index

    predicted_class = np.argmax(
        prediction[0]
    )

    # Confidence

    confidence = (
        prediction[0][predicted_class]
        * 100
    )

    # Class name

    predicted_label = class_names[
        predicted_class
    ]


    # DISPLAY PREDICTION

    st.divider()

    st.subheader("🔍 Prediction")


    st.success(
        f"🌿 Disease: {predicted_label}"
    )


    st.metric(
        "Confidence",
        f"{confidence:.2f}%"
    )


    # TOP 3 PREDICTIONS

    st.subheader(
        "Top 3 Predictions"
    )


    top_3_indices = np.argsort(
        prediction[0]
    )[-3:][::-1]


    for index in top_3_indices:

        label = class_names[index]

        probability = (
            prediction[0][index]
            * 100
        )

        st.write(
            f"**{label}** — "
            f"{probability:.2f}%"
        )


    # GRAD-CAM

    st.divider()

    st.subheader(
        "Grad-CAM Explanation"
    )


    try:

        # Generate heatmap

        heatmap = make_gradcam_heatmap(
            img_array,
            model
        )


        # Create overlay

        gradcam_image = create_gradcam_overlay(
            image,
            heatmap
        )


        # Display Grad-CAM

        st.image(
            gradcam_image,
            caption=(
                "Grad-CAM — Areas influencing "
                "the prediction"
            ),
            use_container_width=True
        )


        st.success(
            "Grad-CAM generated successfully!"
        )


    except Exception as e:

        st.error(
            "Grad-CAM could not be generated."
        )

        st.exception(e)