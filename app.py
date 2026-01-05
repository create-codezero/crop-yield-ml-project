import os
import pickle
import numpy as np
import pandas as pd
from flask import Flask, render_template, request, jsonify

# --- Configuration ---
# NOTE: In a real application, you would load your trained models and scalers here.
# For this example, we will use mock models.
MODEL_DIR = 'models'
RECOMMENDATION_MODEL_PATH = os.path.join(MODEL_DIR, 'recommendation_model.pkl')
YIELD_MODEL_PATH = os.path.join(MODEL_DIR, 'yield_model.pkl')
SCALER_PATH = os.path.join(MODEL_DIR, 'scaler.pkl')

app = Flask(__name__)

# Mock data for demonstration (replace with actual loaded models/scalers)
class MockModel:
    """A placeholder class to simulate a loaded model."""
    def predict(self, X):
        # Simulate classification (22 classes, 0-21) for recommendation
        if X.shape[1] == 7:
            # Mock recommendation: always predict the same crop for simplicity
            return np.array(['rice']) 
        # Simulate regression for yield
        elif X.shape[1] == 11: # 11 features: N, P, K, temp, humidity, ph, rainfall, Crop_Area, State_Encoded, District_Encoded, Season_Encoded
            # Mock yield: return a fixed yield based on input (e.g., sum of N, P, K)
            return np.array([np.sum(X[0, 0:3]) * 100 + 5000]) 
        return np.array([0])

class MockScaler:
    """A placeholder class to simulate a loaded scaler."""
    def transform(self, X):
        return X

# Global variables for models and encoder data
recommendation_model = None
yield_model = None
yield_scaler = None

# Mock data for Categorical Encoding (replace with your actual training data categories)
# For a real yield model, these would be saved when you trained the model.
STATE_NAMES = [
    'Andaman and Nicobar Islands', 'Andhra Pradesh', 'Arunachal Pradesh', 'Assam', 
    'Bihar', 'Chandigarh', 'Chhattisgarh', 'Dadra and Nagar Haveli', 'Goa', 'Gujarat', 
    'Haryana', 'Himachal Pradesh', 'Jammu and Kashmir', 'Jharkhand', 'Karnataka', 
    'Kerala', 'Madhya Pradesh', 'Maharashtra', 'Manipur', 'Meghalaya', 'Mizoram', 
    'Nagaland', 'Odisha', 'Puducherry', 'Punjab', 'Rajasthan', 'Sikkim', 
    'Tamil Nadu', 'Telangana', 'Tripura', 'Uttar Pradesh', 'Uttarakhand', 'West Bengal'
]
DISTRICT_NAMES = [
    '24 PARAGANAS NORTH', '24 PARAGANAS SOUTH', 'ADILABAD', 'AGAR MALWA', 'AGRA', 
    'AHMADABAD', 'AHMEDNAGAR', 'AIZAWL', 'AJMER', 'AKOLA', 'ALAPPUZHA', 
    'ALIGARH', 'ALIRAJPUR', 'ALLAHABAD', 'ALMORA', 'ALWAR', 'AMBALA', 
    'AMBEDKAR NAGAR', 'AMETHI', 'AMRAVATI', 'AMRELI', 'AMRITSAR', 'ANAND', 
    'ANANTAPUR', 'ANANTNAG', 'ANDAMAN ISLANDS', 'ANGUL', 'ANJUNA', 'ANKLESHWAR', 
    'ANUPPUR', 'ARARIA', 'ARAVALLI', 'ARIYALUR', 'ARWAL', 'ASHOKNAGAR', 
    'AURANGABAD', 'AZAMGARH', 'BADGAM', 'BAGALKOT', 'BAGHPAT', 'BAHRAICH', 
    'BALAGHAT', 'BALANGIR', 'BALESHWAR', 'BALIA', 'BALRAMPUR', 'BAMETARA', 
    'BANDA', 'BANDIPORA', 'BANGALORE RURAL', 'BANGALORE URBAN', 'BANKA', 
    'BANKURA', 'BANSWARA', 'BARABANKI', 'BARAMULLA', 'BARAN', 'BARGARH', 
    'BARMER', 'BARPETA', 'BARRACKPORE', 'BARWANI', 'BASTAR', 'BATHINDA', 
    'BEED', 'BEGUSARAI', 'BELGAUM', 'BELLARY', 'BETUL', 'BHABHUA', 
    'BHADRAK', 'BHAGALPUR', 'BHANDARA', 'BHARATPUR', 'BHAVNAGAR', 'BHILWARA', 
    'BHIND', 'BHIWANI', 'BHOPAL', 'BIDAR', 'BIJAPUR', 'BIJNOR', 'BIKANER', 
    'BILASPUR', 'BIRBHUM', 'BISHNUPUR', 'BOKARO', 'BONGAIGAON', 'BOUDH', 
    'BULANDSHAHR', 'BULDHANA', 'BUNDI', 'BURHANPUR', 'BUXAR', 'CACHAR', 
    'CHAMARAJANAGAR', 'CHAMOLI', 'CHAMPARAN EAST', 'CHAMPARAN WEST', 
    'CHAMPHAI', 'CHANDAULI', 'CHANDEL', 'CHANDIGARH', 'CHANDRAPUR', 'CHANGLANG', 
    'CHATRA', 'CHHATARPUR', 'CHHINDWARA', 'CHIKBALLAPUR', 'CHIKMAGALUR', 
    'CHITRADURGA', 'CHITTOOR', 'CHITRAKOOT', 'CHURACHANDPUR', 'CHURU', 
    'COIMBATORE', 'COOCHBEHAR', 'CUDDALORE', 'CUTTACK', 'DADRA AND NAGAR HAVELI', 
    'DAKSHIN DINAJPUR', 'DAMOH', 'DANG', 'DARBHANGA', 'DARJEELING', 'DATIA', 
    'DAUSA', 'DAVANGERE', 'DEHRADUN', 'DEOGARH', 'DEORIA', 'DEWAS', 'DHALAI', 
    'DHANBAD', 'DHAR', 'DHARWAD', 'DHENKANAL', 'DHOLPUR', 'DHUBRI', 'DHULE', 
    'DIBANG VALLEY', 'DIBRUGARH', 'DIDDICHORE', 'DIMA HASAO', 'DIMAPUR', 
    'DINDIGUL', 'DINDORI', 'EAST DISTRICT', 'EAST GARO HILLS', 'EAST KHASI HILLS', 
    'EAST SIANG', 'EAST SIKKIM', 'EASTERN GHATS', 'EASTERN HIGHWAY', 'EASTERN PLAINS', 
    'EIGHTIETH STREET', 'ELLUR', 'ERNAKULAM', 'ERODE', 'ETAH', 'ETAWAH', 
    'FAIZABAD', 'FARIDABAD', 'FARIDKOT', 'FATEHABAD', 'FATEHPUR', 'FAZILKA', 
    'FEROZEPUR', 'FIROZABAD', 'GADAG', 'GADCHIROLI', 'GAJAPATI', 'GANDERBAL', 
    'GANDHINAGAR', 'GANJAM', 'GARHWA', 'GAYA', 'GHAZIABAD', 'GHAZIPUR', 'GOALPARA', 
    'GODDA', 'GOMATI', 'GONDA', 'GONDIA', 'GOPALGANJ', 'GORAKHPUR', 'GUMALA', 
    'GUMLA', 'GUNTUR', 'GURGAON', 'GWALIOR', 'HAILAKANDI', 'HAMIRPUR', 'HANUMANGARH', 
    'HARDAPUR', 'HARDOI', 'HARIDWAR', 'HASSAN', 'HATHRAS', 'HAVERI', 'HAZARIBAGH', 
    'HINGOLI', 'HOSHANGABAD', 'HOSHIARPUR', 'HOWRAH', 'HYDERABAD', 'IBN-E-ABBAS', 
    'IDUKKI', 'IMPHAL EAST', 'IMPHAL WEST', 'INDORE', 'JABALPUR', 'JAGATSINGHPUR', 
    'JALANDHAR', 'JALAUN', 'JALGAON', 'JALNA', 'JALPAIGURI', 'JAMMU', 'JAMNAGAR', 
    'JAMUI', 'JANJGIR-CHAMPA', 'JASHNAGAR', 'JASHNAGARH', 'JAYASHANKAR BHUPALPALLY', 
    'JEHANABAD', 'JEYPORE', 'JHALAWAR', 'JHANSI', 'JIND', 'JODHPUR', 'JOHRI', 
    'JORHAT', 'JUNAGADH', 'KABIRDHAM', 'KACHCHH', 'KADAPA', 'KAIMUR (BHABHUA)', 
    'KAITHAL', 'KALAHANDI', 'KANDHAMAL', 'KANGRA', 'KANNAUJ', 'KANPUR DEHAT', 
    'KANPUR NAGAR', 'KANYAKUMARI', 'KAPURTHALA', 'KARAIKAL', 'KARBI ANGLONG', 
    'KARGIL', 'KARIMGANJ', 'KARNAL', 'KARUR', 'KASARAGOD', 'KASHMIR', 'KATHUA', 
    'KATIHAR', 'KATNI', 'KENDRAPARA', 'KEONJHAR', 'KHAGARIA', 'KHAMMAM', 'KHANDWA', 
    'KHEDA', 'KHURDA', 'KISHANGANJ', 'KISHTWAR', 'KODAGU', 'KOLAM', 'KOLAR', 
    'KOLASIB', 'KOLHAPUR', 'KOLKATA', 'KOLLAM', 'KOPPAL', 'KORAPUT', 'KORBA', 
    'KORIYAR', 'KOTA', 'KOTTAYAM', 'KOZHIKODE', 'KRISHNAGIRI', 'KULGAM', 'KUPWARA', 
    'KURNOOL', 'KURUKSHETRA', 'KUSHINAGAR', 'LAKHIMPUR', 'LAKHISARAI', 'LALITPUR', 
    'LATUR', 'LAWNGTLAI', 'LEH', 'LOHIT', 'LOWER DIBANG VALLEY', 'LUCKNOW', 
    'LUDHIANA', 'MADHUBANI', 'MADURAI', 'MAHARAJGANJ', 'MAHASAMUND', 'MAHBUBNAGAR', 
    'MAHESANA', 'MAHOBA', 'MAINPURI', 'MALAPURAM', 'MALDAH', 'MALDAS', 'MALKANGIRI', 
    'MAMIT', 'MANDI', 'MANDLA', 'MANDSAUR', 'MANSA', 'MARIGAON', 'MATHURA', 
    'MAYURBHANJ', 'MEDAK', 'MEERUT', 'MEHSANA', 'MIDNAPORE EAST', 'MIDNAPORE WEST', 
    'MIRZAPUR', 'MOGA', 'MOKOKCHUNG', 'MON', 'MORENA', 'MUMBAI', 'MUNGELI', 
    'MUSHIRABAD', 'MUZAFFARNAGAR', 'MUZAFFARPUR', 'MYSORE', 'NAGAON', 'NAGAPATTINAM', 
    'NAGARKURNOOL', 'NAGDA', 'NAGPUR', 'NAINITAL', 'NALANDA', 'NALBARI', 
    'NALGONDA', 'NAMAKKAL', 'NAMSAI', 'NANDED', 'NANDURBAR', 'NARAYANPUR', 
    'NARMADA', 'NARSINGHPUR', 'NASIK', 'NAVSARI', 'NAWADA', 'NAWANSHAHR', 
    'NEELAM', 'NELLORE', 'NEW DELHI', 'NICOBARS', 'NIZAMABAD', 'NORTH DISTRICT', 
    'NORTH GOA', 'NORTH TRIPURA', 'NORTHERN GHATS', 'NUAPADA', 'OSMANABAD', 
    'PAKISTAN', 'PALAMAU', 'PALGHAT', 'PALI', 'PALWAL', 'PANCHMAHAL', 
    'PANIPAT', 'PAPAUMPARE', 'PARBHANI', 'PARNA', 'PASCHIM MEDINIPUR', 
    'PATAN', 'PATIALA', 'PATNA', 'PERAMBALUR', 'PEREN', 'PHULBANI', 'PILIBHIT', 
    'PITHORAGARH', 'POONCH', 'PRAKASAM', 'PRATAPGARH', 'PUDUKKOTTAI', 'PULWAMA', 
    'PUNE', 'PURBA BARDHAMAN', 'PURULIA', 'RAICHUR', 'RAIGAD', 'RAIPUR', 
    'RAISEN', 'RAJAURI', 'RAJGARH', 'RAJKOT', 'RAJNANDGAON', 'RAJSAMAND', 
    'RAMANAGARA', 'RAMANATHAPURAM', 'RAMBAN', 'RAMGARH', 'RAMNAGAR', 'RAMPUR', 
    'RANCHI', 'RANGEN', 'RANGAREDDY', 'RATLAM', 'RATNAGIRI', 'RAYAGADA', 
    'REWA', 'REWARI', 'RI BHOI', 'ROHTAK', 'ROHTAS', 'RUDRAPRAYAG', 
    'RUPNAGAR', 'SAGAR', 'SAHARANPUR', 'SAHARSA', 'SAHEBGANJ', 'SALEM', 
    'SAMASTIPUR', 'SAMBALPUR', 'SANGLI', 'SANGRUR', 'SANT RAVIDAS NAGAR', 
    'SARAN', 'SATARA', 'SATNA', 'SAWAI MADHOPUR', 'SEHORE', 'SENAPATI', 
    'SEONI', 'SHAHDOL', 'SHAJAPUR', 'SHEIKHPURA', 'SHEOPUR', 'SHIMLA', 
    'SHIMOGA', 'SHIVPURI', 'SHOPYAN', 'SHRAVASTHI', 'SIDDHARTHNAGAR', 
    'SIDHI', 'SIKAR', 'SILCHAR', 'SIMDEGA', 'SINDHUDURG', 'SINGRAULI', 
    'SIRMAUR', 'SIROHI', 'SIRSA', 'SITAMARHI', 'SITAPUR', 'SIWAN', 'SOLAN', 
    'SOLAPUR', 'SONBHADRA', 'SONIPAT', 'SONITPUR', 'SOUTH DISTRICT', 
    'SOUTH GARO HILLS', 'SOUTH GOA', 'SOUTH TRIPURA', 'SOUTH WEST GARO HILLS', 
    'SRIKAKULAM', 'SRINAGAR', 'SUBANSIRI', 'SUKMA', 'SULTANPUR', 'SUNDARGARH', 
    'SUPAUL', 'SURAT', 'SURENDRANAGAR', 'SURGUJA', 'TAMLUK', 'TANK', 
    'TAPTI', 'TARN TARAN', 'THANE', 'THANJAVUR', 'THENI', 'THIRUVALLUR', 
    'THIRUVANANTHAPURAM', 'THOOTHUKUDI', 'THRISSUR', 'TIKAMGARH', 'TINSUKIA', 
    'TIRAP', 'TIRUCHIRAPPALLI', 'TIRUNELVELI', 'TIRUPPUR', 'TIRUVANNAMALAI', 
    'TONK', 'TUENSANG', 'TUMAKURU', 'UDAIPUR', 'UDALGURI', 'UDHAMPUR', 
    'UDUPI', 'UJJAIN', 'UMARIA', 'UNNAO', 'UPPER SIANG', 'UPPER SUBANSIRI', 
    'UTTAR KANNADA', 'UTTAR DINAJPUR', 'UTTARKASHI', 'VADODARA', 'VAIZAG', 
    'VAIZIANAGRAM', 'VALLUR', 'VANIYAMBADI', 'VARANASI', 'VELLORE', 
    'VIDISHA', 'VIJAYAPURA', 'VIKAS', 'VILLUPURAM', 'VIRUDHUNAGAR', 
    'VISHAKHAPATNAM', 'VIZIANAGARAM', 'WAGHODIA', 'WARRANGAL', 'WASHIM', 
    'WAYANAD', 'WEST DISTRICT', 'WEST GARO HILLS', 'WEST KHASI HILLS', 
    'WEST SIANG', 'WEST SIKKIM', 'WESTERN GHATS', 'WOKHA', 'YADGIR', 
    'YAMUNANAGAR', 'YSR', 'ZUNHEBOTO', 
    # ... 641 total districts
]
SEASONS = [
    'Autumn', 'Kharif', 'Rabi', 'Summer', 'Whole Year', 'Winter'
]
CROPS = [
    'banana', 'blackgram', 'coconut', 'coffee', 'grapes', 'jute', 'lentil', 
    'maize', 'mango', 'orange', 'papaya', 'rice'
]


def load_models():
    """Loads models and scalers from disk."""
    global recommendation_model, yield_model, yield_scaler

    print("--- Loading Models ---")
    try:
        # NOTE: REPLACE THIS WITH YOUR ACTUAL MODEL LOADING LOGIC
        # For demonstration, we use mock classes
        recommendation_model = MockModel()
        yield_model = MockModel()
        yield_scaler = MockScaler() 

        # If using pickle:
        # with open(RECOMMENDATION_MODEL_PATH, 'rb') as f:
        #     recommendation_model = pickle.load(f)
        # with open(YIELD_MODEL_PATH, 'rb') as f:
        #     yield_model = pickle.load(f)
        # with open(SCALER_PATH, 'rb') as f:
        #     yield_scaler = pickle.load(f)
            
        print("Models loaded successfully (MOCK mode active).")

    except Exception as e:
        print(f"Error loading models. Using Mock Models: {e}")
        recommendation_model = MockModel()
        yield_model = MockModel()
        yield_scaler = MockScaler()

    # NOTE: Ensure the models are loaded before the first request
    # This function is called before the first request via @app.before_first_request

# --- Utility Functions for Prediction ---

def get_encoded_features(state, district, season, crop_area):
    """Mocks the creation of encoded features for the Yield model."""
    # In a real app, you would use a pre-fitted OneHotEncoder or LabelEncoder
    # saved from your training pipeline.
    
    # Mock encoding (simple index-based)
    state_enc = STATE_NAMES.index(state) if state in STATE_NAMES else 0
    district_enc = DISTRICT_NAMES.index(district) if district in DISTRICT_NAMES else 0
    season_enc = SEASONS.index(season) if season in SEASONS else 0
    
    return np.array([crop_area, state_enc, district_enc, season_enc]).reshape(1, 4)


def prepare_recommendation_input(data):
    """Prepares the input array for the Recommendation Model."""
    features = [
        float(data['N']), float(data['P']), float(data['K']),
        float(data['temperature']), float(data['humidity']),
        float(data['ph']), float(data['rainfall'])
    ]
    return np.array([features])

def prepare_yield_input(data):
    """Prepares the input array for the Yield Prediction Model."""
    # 1. Environmental Features (7)
    env_features = [
        float(data['N']), float(data['P']), float(data['K']),
        float(data['temperature']), float(data['humidity']),
        float(data['ph']), float(data['rainfall'])
    ]
    
    # 2. Area/Location/Season Features (4 encoded)
    loc_features = get_encoded_features(
        data['state'], data['district'], data['season'], float(data['area'])
    )
    
    # Combine the features: (1 x 7) + (1 x 4) = (1 x 11)
    combined_features = np.concatenate((np.array([env_features]), loc_features), axis=1)
    
    # Apply scaler (if you used one during training)
    scaled_features = yield_scaler.transform(combined_features)
    
    return scaled_features

# --- Flask Routes ---

@app.route('/', methods=['GET'])
def index():
    """Renders the main application interface."""
    return render_template('index.html', 
        states=STATE_NAMES, 
        districts=DISTRICT_NAMES, 
        seasons=SEASONS,
        crops=CROPS
    )

@app.route('/predict_recommendation', methods=['POST'])
def predict_recommendation():
    """Handles the crop recommendation prediction."""
    try:
        data = request.json
        X_test = prepare_recommendation_input(data)
        
        # Predict the best crop
        prediction = recommendation_model.predict(X_test)[0]
        
        # Mock confidence/ranking (replace with model's predict_proba if available)
        mock_rankings = [
            {'crop': prediction, 'confidence': 0.85},
            {'crop': 'maize', 'confidence': 0.75},
            {'crop': 'blackgram', 'confidence': 0.60},
        ]
        
        return jsonify({
            'success': True,
            'prediction': prediction,
            'rankings': mock_rankings
        })
    except Exception as e:
        app.logger.error(f"Recommendation prediction failed: {e}")
        return jsonify({'success': False, 'message': str(e)}), 500

@app.route('/predict_yield', methods=['POST'])
def predict_yield():
    """Handles the crop yield prediction."""
    try:
        data = request.json
        X_test = prepare_yield_input(data)
        
        # Predict the yield (regression)
        predicted_yield_kg_per_ha = yield_model.predict(X_test)[0]
        
        # Assuming the target was log-transformed, you would invert it here:
        # predicted_yield = np.exp(predicted_yield_log)
        
        # Format the result nicely
        result_text = f"{predicted_yield_kg_per_ha:,.2f} kg per Hectare"
        
        # Mock historical data for charting (replace with a real database/API query)
        mock_history = [
            {'year': 2020, 'yield': 8500},
            {'year': 2021, 'yield': 9100},
            {'year': 2022, 'yield': 9550},
            {'year': 2023, 'yield': 9200},
        ]
        
        return jsonify({
            'success': True,
            'prediction': result_text,
            'raw_yield': predicted_yield_kg_per_ha,
            'historical_data': mock_history
        })
    except Exception as e:
        app.logger.error(f"Yield prediction failed: {e}")
        return jsonify({'success': False, 'message': str(e)}), 500

# Function to run before the first request
with app.app_context():
    load_models()

if __name__ == '__main__':
    # Create the models directory if it doesn't exist
    os.makedirs(MODEL_DIR, exist_ok=True) 
    
    # NOTE: You must place your trained models (e.g., recommendation_model.pkl, yield_model.pkl) 
    # and the necessary scaler/encoder objects into the 'models' directory 
    # for the non-mock loading logic to work.

    app.run(debug=True)