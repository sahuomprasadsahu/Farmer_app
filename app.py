#==========
#Farmer_app
#==========
import cv2
import numpy as np
from PIL import Image
from flask import Flask, render_template, request
import tensorflow as tf
import os
import json
import base64
from io import BytesIO

app = Flask(__name__)
app.config['MAX_CONTENT_LENGTH'] = 30 * 1024 * 1024

# ===============================
# LOAD MODELS
# ===============================
BASE_DIR = os.path.dirname(os.path.abspath(__file__))

plant_model = tf.keras.models.load_model(os.path.join(BASE_DIR, "models/plant_image_classifier.keras"))
final_model = tf.keras.models.load_model(os.path.join(BASE_DIR, "models/final_model.keras"))

# ===============================
# LOAD CLASS FILES
# ===============================
with open("models/plant_classes.json") as f:
    plant_classes = json.load(f)

with open("models/class_names.json") as f:
    class_names = json.load(f)

# ===============================
# UI TRANSLATIONS (Plantix Style)
# ===============================
translations = {
    "not_leaf": {"en": "Not a Leaf", "od": "ପତ୍ର ନୁହେଁ"},
    "healthy": {"en": "Healthy Plant", "od": "ସୁସ୍ଥ ଗଛ"},
    "disease": {"en": "Disease Detected", "od": "ରୋଗ ଚିହ୍ନଟ"},
    "low_conf": {"en": "Low Confidence", "od": "ନିମ୍ନ ବିଶ୍ୱାସ"},
}

def t(key, lang):
    return translations.get(key, {}).get(lang, key)

# ===============================
# DISEASE INFO (EN only, OD optional)
# ===============================
disease_info = {

# ================= APPLE =================
"Apple___Apple_scab": {
"description_en": "Olive-green to dark spots on leaves and fruits.",
"solution_en": "Apply fungicide like captan or sulfur.",
"description_od": "ପତ୍ର ଓ ଫଳରେ କଳା-ହରିତ ଦାଗ ହୁଏ।",
"solution_od": "କ୍ୟାପଟାନ୍ କିମ୍ବା ସଲଫର ଔଷଧ ଛିଟନ୍ତୁ।"
},

"Apple___Black_rot": {
"description_en": "Black rotten spots on fruits and leaves.",
"solution_en": "Remove infected branches and spray fungicide.",
"description_od": "ଫଳ ଓ ପତ୍ରରେ କଳା ପଚା ଦାଗ ହୁଏ।",
"solution_od": "ଖରାପ ଡାଳ କାଟି ଔଷଧ ଛିଟନ୍ତୁ।"
},

"Apple___Cedar_apple_rust": {
"description_en": "Yellow-orange spots on leaves.",
"solution_en": "Remove nearby infected plants and spray fungicide.",
"description_od": "ପତ୍ରରେ ହଳଦିଆ-କେସରି ଦାଗ ଦେଖାଯାଏ।",
"solution_od": "ପାଖରେ ଥିବା ଖରାପ ଗଛ ହଟାଇ ଔଷଧ ଛିଟନ୍ତୁ।"
},

"Apple___healthy": {
"description_en": "Plant is healthy.",
"solution_en": "Maintain proper care.",
"description_od": "ଗଛ ସୁସ୍ଥ ଅଛି।",
"solution_od": "ଠିକ୍ ଭାବେ ପାଣି ଓ ସାର ଦିଅନ୍ତୁ।"
},

# ================= CHERRY =================
"Cherry_(including_sour)___Powdery_mildew": {
"description_en": "White powdery layer on leaves.",
"solution_en": "Use sulfur spray and keep air flow.",
"description_od": "ପତ୍ରରେ ସ୍ୱେତ ଗୁଡ଼ିଆ ଭଳି ଲେୟର ହୁଏ।",
"solution_od": "ସଲଫର ଔଷଧ ଛିଟନ୍ତୁ ଓ ବାତାସ ଚଳାଚଳ ରଖନ୍ତୁ।"
},

"Cherry_(including_sour)___healthy": {
"description_en": "Plant is healthy.",
"solution_en": "Continue normal care.",
"description_od": "ଗଛ ସୁସ୍ଥ ଅଛି।",
"solution_od": "ସାଧାରଣ ଯତ୍ନ ଚାଲୁ ରଖନ୍ତୁ।"
},

# ================= CORN =================
"Corn_(maize)___Cercospora_leaf_spot Gray_leaf_spot": {
"description_en": "Gray rectangular spots on leaves.",
"solution_en": "Use resistant seeds and spray fungicide.",
"description_od": "ପତ୍ରରେ ଧୂସର ଚକୋର ଦାଗ ହୁଏ।",
"solution_od": "ଭଲ ବିଆ ବ୍ୟବହାର କରନ୍ତୁ ଓ ଔଷଧ ଛିଟନ୍ତୁ।"
},

"Corn_(maize)___Common_rust_": {
"description_en": "Reddish-brown spots on leaves.",
"solution_en": "Use fungicide spray.",
"description_od": "ପତ୍ରରେ ଲାଲ-ବୁଣା ଦାଗ ହୁଏ।",
"solution_od": "ଫଙ୍ଗିସାଇଡ୍ ଛିଟନ୍ତୁ।"
},

"Corn_(maize)___Northern_Leaf_Blight": {
"description_en": "Long gray lesions on leaves.",
"solution_en": "Rotate crops and spray fungicide.",
"description_od": "ପତ୍ରରେ ଲମ୍ବା ଧୂସର ଦାଗ ହୁଏ।",
"solution_od": "ଫସଲ ବଦଳାନ୍ତୁ ଓ ଔଷଧ ଛିଟନ୍ତୁ।"
},

"Corn_(maize)___healthy": {
"description_en": "Plant is healthy.",
"solution_en": "Maintain soil and water.",
"description_od": "ଗଛ ସୁସ୍ଥ ଅଛି।",
"solution_od": "ମାଟି ଓ ପାଣି ଠିକ୍ ରଖନ୍ତୁ।"
},

# ================= GRAPE =================
"Grape___Black_rot": {
"description_en": "Brown spots and black shriveled fruits.",
"solution_en": "Remove infected parts and spray fungicide.",
"description_od": "ପତ୍ରରେ ବୁଣା ଦାଗ ଓ ଫଳ କଳା ହୋଇଯାଏ।",
"solution_od": "ଖରାପ ଅଂଶ କାଟି ଔଷଧ ଛିଟନ୍ତୁ।"
},

"Grape___Esca_(Black_Measles)": {
"description_en": "Leaf discoloration and streaks.",
"solution_en": "Remove infected vines.",
"description_od": "ପତ୍ରର ରଙ୍ଗ ବଦଳିଯାଏ।",
"solution_od": "ଖରାପ ଲତା କାଟିଦିଅନ୍ତୁ।"
},

"Grape___Leaf_blight_(Isariopsis_Leaf_Spot)": {
"description_en": "Brown spots with yellow edges.",
"solution_en": "Improve airflow and spray fungicide.",
"description_od": "ପତ୍ରରେ ବୁଣା ଦାଗ ଓ ପାଖରେ ହଳଦିଆ ରଙ୍ଗ।",
"solution_od": "ବାତାସ ଚଳାଚଳ ରଖନ୍ତୁ ଓ ଔଷଧ ଛିଟନ୍ତୁ।"
},

"Grape___healthy": {
"description_en": "Plant is healthy.",
"solution_en": "Maintain care.",
"description_od": "ଗଛ ସୁସ୍ଥ ଅଛି।",
"solution_od": "ଠିକ୍ ଯତ୍ନ ରଖନ୍ତୁ।"
},

# ================= PEACH =================
"Peach___Bacterial_spot": {
"description_en": "Dark water-soaked spots.",
"solution_en": "Use copper spray.",
"description_od": "ପତ୍ରରେ କଳା ଭିଜା ଦାଗ ହୁଏ।",
"solution_od": "କପର ଔଷଧ ଛିଟନ୍ତୁ।"
},

"Peach___healthy": {
"description_en": "Plant is healthy.",
"solution_en": "Maintain nutrients.",
"description_od": "ଗଛ ସୁସ୍ଥ ଅଛି।",
"solution_od": "ସାର ଠିକ୍ ଦିଅନ୍ତୁ।"
},

# ================= PEPPER =================
"Pepper,_bell___Bacterial_spot": {
"description_en": "Small brown spots.",
"solution_en": "Avoid overwatering and spray medicine.",
"description_od": "ଛୋଟ ବୁଣା ଦାଗ ହୁଏ।",
"solution_od": "ଅଧିକ ପାଣି ଦିଅନ୍ତୁ ନାହିଁ ଓ ଔଷଧ ଛିଟନ୍ତୁ।"
},

"Pepper,_bell___healthy": {
"description_en": "Plant is healthy.",
"solution_en": "Maintain sunlight.",
"description_od": "ଗଛ ସୁସ୍ଥ ଅଛି।",
"solution_od": "ପର୍ଯ୍ୟାପ୍ତ ଧୂପ ଦିଅନ୍ତୁ।"
},

# ================= POTATO =================
"Potato___Early_blight": {
"description_en": "Brown ring spots.",
"solution_en": "Remove infected leaves.",
"description_od": "ପତ୍ରରେ ବୁଣା ଗୋଲା ଦାଗ ହୁଏ।",
"solution_od": "ଖରାପ ପତ୍ର କାଟନ୍ତୁ।"
},

"Potato___Late_blight": {
"description_en": "Fast spreading dark spots.",
"solution_en": "Immediate fungicide spray.",
"description_od": "କଳା ଦାଗ ଶୀଘ୍ର ଛଡାଏ।",
"solution_od": "ତୁରନ୍ତ ଔଷଧ ଛିଟନ୍ତୁ।"
},

"Potato___healthy": {
"description_en": "Plant is healthy.",
"solution_en": "Maintain soil moisture.",
"description_od": "ଗଛ ସୁସ୍ଥ ଅଛି।",
"solution_od": "ମାଟିରେ ଆର୍ଦ୍ରତା ରଖନ୍ତୁ।"
},

# ================= STRAWBERRY =================
"Strawberry___Leaf_scorch": {
"description_en": "Purple-brown spots drying leaves.",
"solution_en": "Remove infected leaves.",
"description_od": "ପତ୍ରରେ ବୁଣା-ଜାମୁନି ଦାଗ ହୁଏ।",
"solution_od": "ଖରାପ ପତ୍ର କାଟନ୍ତୁ।"
},

"Strawberry___healthy": {
"description_en": "Plant is healthy.",
"description_od": "ଗଛ ସୁସ୍ଥ ଅଛି।",
"solution_en": "Maintain spacing.",
"solution_od": "ଗଛ ମଧ୍ୟରେ ଦୂରତା ରଖନ୍ତୁ।"
},

# ================= TOMATO =================
"Tomato___Bacterial_spot": {
"description_en": "Small dark spots.",
"solution_en": "Use copper spray.",
"description_od": "ଛୋଟ କଳା ଦାଗ ହୁଏ।",
"solution_od": "କପର ଔଷଧ ଛିଟନ୍ତୁ।"
},

"Tomato___Leaf_Mold": {
"description_en": "Yellow spots with mold under leaf.",
"solution_en": "Reduce humidity.",
"description_od": "ପତ୍ରରେ ହଳଦିଆ ଦାଗ ଓ ତଳେ ଫଂଗସ୍।",
"solution_od": "ଆର୍ଦ୍ରତା କମ କରନ୍ତୁ।"
},

"Tomato___Septoria_leaf_spot": {
"description_en": "Small round spots.",
"solution_en": "Remove infected leaves.",
"description_od": "ଛୋଟ ଗୋଲା ଦାଗ ହୁଏ।",
"solution_od": "ଖରାପ ପତ୍ର କାଟନ୍ତୁ।"
},

"Tomato___Spider_mites Two-spotted_spider_mite": {
"description_en": "Tiny yellow spots and webs.",
"solution_en": "Spray water or pesticide.",
"description_od": "ଛୋଟ ହଳଦିଆ ଦାଗ ଓ ଜାଲ ହୁଏ।",
"solution_od": "ପାଣି ଛିଟନ୍ତୁ କିମ୍ବା ଔଷଧ ବ୍ୟବହାର କରନ୍ତୁ।"
},

"Tomato___Target_Spot": {
"description_en": "Target-like brown spots.",
"description_od": "ପତ୍ରରେ ଟାର୍ଗେଟ ଭଳି ଦାଗ ହୁଏ।",
"solution_en": "Improve airflow.",
"solution_od": "ବାତାସ ଚଳାଚଳ ରଖନ୍ତୁ।"
},

"Tomato___Tomato_mosaic_virus": {
"description_en": "Light and dark green patches.",
"solution_en": "Remove infected plants.",
"description_od": "ପତ୍ରରେ ହଳଦିଆ-ହରିତ ଦାଗ ହୁଏ।",
"solution_od": "ଖରାପ ଗଛ ହଟାନ୍ତୁ।"
},

"Tomato___Tomato_Yellow_Leaf_Curl_Virus": {
"description_en": "Leaves turn yellow and curl.",
"solution_en": "Control insects.",
"description_od": "ପତ୍ର ହଳଦିଆ ହୋଇ ବାକୁଡ଼ିଯାଏ।",
"solution_od": "ପୋକା ନିୟନ୍ତ୍ରଣ କରନ୍ତୁ।"
},

# ================= RICE =================
"Rice___Brown_spot": {
"description_en": "Brown spots on leaves.",
"solution_en": "Use Mancozeb.",
"description_od": "ଧାନ ପତ୍ରରେ ବୁଣା ଦାଗ ହୁଏ।",
"solution_od": "ମାନକୋଜେବ୍ ଔଷଧ ଛିଟନ୍ତୁ।"
},

"Rice___Leafblast": {
"description_en": "Diamond-shaped spots.",
"solution_en": "Use Tricyclazole.",
"description_od": "ପତ୍ରରେ ହୀରା ଆକାର ଦାଗ ହୁଏ।",
"solution_od": "ଟ୍ରାଇସାଇକ୍ଲାଜୋଲ୍ ଛିଟନ୍ତୁ।"
},

"Rice___Hispa": {
"description_en": "Insect damages leaves.",
"solution_en": "Use insecticide.",
"description_od": "ପୋକା ପତ୍ର ଖାଇ ଦିଏ।",
"solution_od": "ପୋକା ମାରିବା ଔଷଧ ଛିଟନ୍ତୁ।"
},

"Rice___Healthy": {
"description_en": "Plant is healthy.",
"solution_en": "Maintain water.",
"description_od": "ଧାନ ଗଛ ସୁସ୍ଥ ଅଛି।",
"solution_od": "ପାଣି ଠିକ୍ ରଖନ୍ତୁ।"
},

# ================= NON LEAF =================
"Non___Leaf": {
"description_en": "Not a leaf.",
"solution_en": "Upload correct image.",
"description_od": "ଏହା ପତ୍ର ନୁହେଁ।",
"solution_od": "ଠିକ୍ ପତ୍ର ଫଟୋ ଦିଅନ୍ତୁ।"
}

}
# ===============================
# HOME
# ===============================
@app.route("/")
def home():
    return render_template("index.html")


# ===============================
# PREDICT
# ===============================
@app.route("/predict", methods=["POST"])
def predict():

    lang = request.form.get("lang", "en")  # 🔥 language select

    image = None
    image_data = request.form.get("image_data")

    # ===============================
    # GET IMAGE
    # ===============================
    if image_data:
        try:
            image_data = image_data.split(",")[1]
            image_bytes = base64.b64decode(image_data)
            image = Image.open(BytesIO(image_bytes)).convert("RGB")
        except:
            return "Camera error"

    elif "file" in request.files:
        file = request.files["file"]
        if file.filename != "":
            image = Image.open(file).convert("RGB")

    if image is None:
        return "No image"

    # ===============================
    # PLANT MODEL
    # ===============================
    plant_img = image.resize((128, 128))
    plant_array = np.array(plant_img) / 255.0
    plant_array = np.expand_dims(plant_array, axis=0)

    plant_pred = plant_model.predict(plant_array, verbose=0)
    plant_name = plant_classes[np.argmax(plant_pred[0])]

    # ===============================
    # FINAL MODEL
    # ===============================
    final_img = image.resize((224, 224))
    final_array = np.array(final_img) / 255.0
    final_array = np.expand_dims(final_array, axis=0)

    final_pred = final_model.predict(final_array, verbose=0)
    predicted_class = class_names[np.argmax(final_pred[0])].strip()
    confidence = float(np.max(final_pred[0])) * 100

    # ===============================
    # LOGIC
    # ===============================
    if predicted_class.lower() == "non___leaf":

        status_text = t("not_leaf", lang)
        status_color = "orange"

        info = disease_info.get("Non___Leaf")

    else:

        if "healthy" in predicted_class.lower():
            status_text = t("healthy", lang)
            status_color = "green"
        else:
            status_text = t("disease", lang)
            status_color = "red"

        info = disease_info.get(predicted_class)

    # ===============================
    # LANGUAGE OUTPUT
    # ===============================
    if info:
        if lang == "od":
            description = info.get("description_od", info.get("description_en"))
            solution = info.get("solution_od", info.get("solution_en"))
        else:
            description = info.get("description_en")
            solution = info.get("solution_en")
    else:
        description = "No info"
        solution = "Consult expert"

    # ===============================
    # IMAGE DISPLAY
    # ===============================
    buffered = BytesIO()
    image.save(buffered, format="PNG")
    img_str = base64.b64encode(buffered.getvalue()).decode()

    return render_template(
        "index.html",
        prediction=predicted_class,
        description=description,
        solution=solution,
        confidence=round(confidence, 2),
        img_data=img_str,
        status_color=status_color,
        status_text=status_text,
        lang=lang
    )


# ===============================
# RUN
# ===============================
if __name__ == "__main__":
    app.run(host="0.0.0.0",port=5000,debug=True)