
import cv2
import numpy as np
from PIL import Image
from flask import Flask, render_template, request
import tensorflow as tf
import os
import json
import base64
from io import BytesIO
import requests
from sklearn.linear_model import LinearRegression
import pickle
from dotenv import load_dotenv

app = Flask(__name__)
app.config['MAX_CONTENT_LENGTH'] = 30 * 1024 * 1024

# ===============================
# MARKET MODEL (Simple Regression)
# ===============================
def train_market_model():
    import random
    import pandas as pd

    data = []

    for i in range(500):
        price = random.randint(15, 30)
        quantity = random.randint(50, 200)
        transport = random.randint(100, 500)
        waste = random.uniform(0.05, 0.2)

        profit = (price * quantity) - transport - (waste * price * quantity)

        data.append([price, quantity, transport, waste, profit])

    df = pd.DataFrame(data, columns=["price","quantity","transport","waste","profit"])

    X = df[["price","quantity","transport","waste"]]
    y = df["profit"]

    model = LinearRegression()
    model.fit(X, y)

    return model

market_model = train_market_model()

# ===============================
# LOAD MODELS
# ===============================
BASE_DIR = os.path.dirname(os.path.abspath(__file__))

plant_model = tf.keras.models.load_model(os.path.join(BASE_DIR, "models/plant_image_classifier.keras"))
#plant_model.summary() #used to know model parameters
final_model = tf.keras.models.load_model(os.path.join(BASE_DIR, "models/final_model.keras"))
#final_model.summary()

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

def t(key, lang):
    return translations.get(key,{}).get(lang, key)

#================================
#Wheather API key
#================================

load_dotenv()

API_KEY = os.getenv("API_KEY")
#===============================
#Get-Wheather 
#===============================
def get_current_weather(city=None, lat=None, lon=None):
    try:

        if city:
            url = f"http://api.openweathermap.org/data/2.5/weather?q={city}&appid={API_KEY}&units=metric"
        else:
            url = f"http://api.openweathermap.org/data/2.5/weather?lat={lat}&lon={lon}&appid={API_KEY}&units=metric"

        res = requests.get(url).json()

        return {
            "temp": res["main"]["temp"],
            "humidity": res["main"]["humidity"],
            "desc": res["weather"][0]["description"],
            "city": res["name"]
        }
    except:
        return None

# ===============================
# 7 DAY FORECAST
# ===============================
def get_7day_weather(lat, lon):

    try:
        url = f"https://api.openweathermap.org/data/2.5/forecast?lat={lat}&lon={lon}&appid={API_KEY}&units=metric"
        res = requests.get(url).json()

        forecast = []
        for i in range(0, len(res["list"]), 8):  # every 24 hours
            day = res["list"][i]
            forecast.append({
                "temp": day["main"]["temp"],
                "desc": day["weather"][0]["description"]
            })

        return forecast[:7]
    except:
        return []



# ===============================
# HOME
# ===============================
@app.route("/", methods=["GET", "POST"])
def home():

    weather = None

    if request.method == "POST":
        city = request.form.get("city")

        if city:
            weather = get_current_weather(city=city)
        else:
            lat = request.form.get("lat")
            lon = request.form.get("lon")
            weather = get_current_weather(lat=lat, lon=lon)

    return render_template("index.html", weather=weather)
# ===============================
# MARKET PAGE
# ===============================
@app.route("/market")
def market_page():
    return render_template("market.html")


# ===============================
# PREDICT MARKET
# ===============================
@app.route("/predict_market", methods=["POST"])
def predict_market():

    try:
        # ✅ SAFE INPUT (avoid NoneType error)
        num_markets = int(request.form.get("num_markets") or 0)
        quantity = float(request.form.get("quantity") or 0)

        # ❌ validation
        if num_markets < 1 or quantity <= 0:
            return render_template("market.html", error="Invalid input")

        markets = []

        # ===============================
        # COLLECT MARKET DATA
        # ===============================
        for i in range(1, num_markets + 1):

            name = request.form.get(f"name{i}")
            price = request.form.get(f"price{i}")
            transport = request.form.get(f"transport{i}")
            waste = request.form.get(f"waste{i}")

            # ❌ skip incomplete markets
            if not name or not price or not transport or not waste:
                continue

            try:
                price = float(price)
                transport = float(transport)
                waste = float(waste) / 100
            except:
                continue  # skip invalid values

            markets.append({
                "name": name,
                "price": price,
                "quantity": quantity,
                "transport": transport,
                "waste": waste
            })

        # ❌ if no valid markets
        if len(markets) == 0:
            return render_template("market.html", error="No valid market data")

        best_market = None
        max_profit = float('-inf')
        results = []

        # ===============================
        # PREDICTION
        # ===============================
        for m in markets:

            X = [[m["price"], m["quantity"], m["transport"], m["waste"]]]

            try:
                profit = market_model.predict(X)[0]
            except:
                profit = (m["price"] * m["quantity"]) - m["transport"] - (m["waste"] * m["price"] * m["quantity"])
                #to avaoid crash (correct with fall back)

            results.append({
                "name": m["name"],
                "profit": round(profit, 2)
            })

            if profit > max_profit:
                max_profit = profit
                best_market = m["name"]

        # ===============================
        # RETURN RESULT
        # ===============================
        return render_template(
            "market.html",
            best_market=best_market,
            max_profit=round(max_profit, 2),
            results=results
        )

    except Exception as e:
        # 🔥 debug error safely
        print("ERROR:", e)
        return render_template("market.html", error="Something went wrong")
#===============================
#Get-Forecast api
#===============================
@app.route("/forecast", methods=["POST"])
def forecast():

    lat = request.form.get("lat")
    lon = request.form.get("lon")

    data = get_7day_weather(lat, lon)

    return json.dumps(data)


# ===============================
# PREDICT
# ===============================
@app.route("/predict", methods=["POST"])
def predict():
    try:
        lang = request.form.get("lang", "en")

        image = None
        image_data = request.form.get("image_data")

        # CAMERA
        if image_data:
            image_data = image_data.split(",")[1]
            image_bytes = base64.b64decode(image_data)
            image = Image.open(BytesIO(image_bytes)).convert("RGB")

        # UPLOAD
        elif request.files.get("file"):
            file = request.files["file"]
            if file and file.filename != "":
                image = Image.open(file).convert("RGB")

        if image is None:
            return "No image"

        # Resize
        image = image.resize((512,512))

        img = image.resize((224,224))
        arr = np.array(img)/255.0
        arr = np.expand_dims(arr,0)

        # Prediction
        pred = final_model.predict(arr, verbose=0)
        predicted_class = class_names[np.argmax(pred[0])]
        confidence = float(np.max(pred[0])) * 100

        # ===== FIX LOGIC =====
        if predicted_class == "Non___Leaf":
            status_text = t("not_leaf", lang)
            status_color = "orange"
            info = disease_info.get("Non___Leaf", {})

        elif "healthy" in predicted_class.lower():
            status_text = t("healthy", lang)
            status_color = "green"
            info = disease_info.get(predicted_class, {})

        else:
            status_text = t("disease", lang)
            status_color = "red"
            info = disease_info.get(predicted_class, {})

        # ===== DESCRIPTION FIX =====
        description = info.get(f"description_{lang}", info.get("description_en", "No description"))
        solution = info.get(f"solution_{lang}", info.get("solution_en", "No solution"))

        # Image compress
        buf = BytesIO()
        image.save(buf, format="JPEG", quality=60)
        img_str = base64.b64encode(buf.getvalue()).decode()

        return render_template(
            "index.html",
            prediction=predicted_class,
            confidence=round(confidence,2),
            description=description,
            solution=solution,
            img_data=img_str,
            status_color=status_color,
            status_text=status_text,
            lang=lang
        )

    except Exception as e:
        print("ERROR:", e)
        return "Error"
# ===============================
# RUN
# ===============================
import os

if __name__ == "__main__":
    port = int(os.environ.get("PORT", 5000))
    app.run(host="0.0.0.0", port=port)
