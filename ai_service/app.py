import uvicorn
from fastapi import FastAPI, UploadFile, File, Form, HTTPException, Request
from fastapi.responses import JSONResponse
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
import random
from typing import Optional
import io
import os
import sys
import json
import pandas as pd
import torch
import numpy as np
import logging
from pathlib import Path
import hashlib


# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

try:
    from PIL import Image
except Exception:
    Image = None

ROOT_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if ROOT_DIR not in sys.path:
    sys.path.append(ROOT_DIR)


def get_transforms():
    try:
        from torchvision import transforms
        return transforms
    except Exception:
        return None


def get_image_inference_transform():
    transforms = get_transforms()
    if transforms is None:
        return None
    return transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(
            mean=[0.485, 0.456, 0.406],
            std=[0.229, 0.224, 0.225]
        )
    ])


def load_image_checkpoint(candidates):
    for candidate in candidates:
        checkpoint = load_image_model(candidate)
        if checkpoint is not None:
            logger.info(f"Loaded image checkpoint: {candidate}")
            return checkpoint, candidate
    logger.warning(f"No image checkpoint could be loaded from candidates: {candidates}")
    return None, None


def analyze_leaf_image(image):
    pixels = np.asarray(image.convert('RGB'), dtype=np.float32) / 255.0
    red = pixels[:, :, 0]
    green = pixels[:, :, 1]
    blue = pixels[:, :, 2]
    brightness = pixels.mean(axis=2)

    green_ratio = float(((green > red * 1.05) & (green > blue * 1.05) & (green > 0.2)).mean())
    dark_ratio = float((brightness < 0.35).mean())
    brown_ratio = float(((red > green * 1.08) & (red > 0.25) & (green > blue * 0.9)).mean())

    return {
        'green_ratio': green_ratio,
        'dark_ratio': dark_ratio,
        'brown_ratio': brown_ratio,
    }


def get_plant_family(label):
    if not label:
        return None
    if '___' in label:
        return label.split('___', 1)[0]
    return label.split('_', 1)[0].title()


def choose_plant_heuristic_label(image_stats, model_label=None):
    return None


def format_plant_display_name(label):
    if not label:
        return 'Unknown Plant Condition'
    if '___' not in label:
        return str(label).replace('_', ' ').title()

    crop_name, disease_name = label.split('___', 1)
    crop_name = crop_name.replace('_', ' ').title()
    disease_name = disease_name.replace('_', ' ').title()
    if disease_name.lower() == 'healthy':
        return f'{crop_name} Healthy Leaf'
    return f'{crop_name} {disease_name}'

# model loader in the service
try:
    from .model_loader import (
        load_crop_recommender,
        load_crop_metadata,
        load_yield_predictor,
        load_yield_metadata,
        load_price_predictor,
        load_price_metadata,
        load_fertilizer_recommender,
        load_fertilizer_metadata,
        load_image_model,
    )
except ImportError:
    from ai_service.model_loader import (
        load_crop_recommender,
        load_crop_metadata,
        load_yield_predictor,
        load_yield_metadata,
        load_price_predictor,
        load_price_metadata,
        load_fertilizer_recommender,
        load_fertilizer_metadata,
        load_image_model,
    )

# attempt to import ResNet9 for inference construction
try:
    from ai.models.resnet9 import ResNet9
except Exception:
    ResNet9 = None

PLANT_CHECKPOINT = None
ANIMAL_CHECKPOINT = None
_IMAGE_MODEL_CACHE = {}

def get_cached_resnet_model(model_key, classes, model_state):
    if model_key in _IMAGE_MODEL_CACHE:
        return _IMAGE_MODEL_CACHE[model_key]
    if ResNet9 is None:
        raise RuntimeError("ResNet9 model class is not available")
    model = ResNet9(in_channels=3, num_classes=len(classes))
    model.load_state_dict(model_state)
    model.eval()
    _IMAGE_MODEL_CACHE[model_key] = model
    return model


def verify_and_load_model_startup(candidates, model_name):
    import time
    if isinstance(candidates, str):
        candidates = [candidates]
        
    resolved_path = None
    for cand in candidates:
        p = Path(os.path.join(ROOT_DIR, cand)).resolve()
        if p.exists():
            resolved_path = p
            break
            
    print(f"=== Model Loading Verification for {model_name} ===", flush=True)
    print(f"Model path: {resolved_path}", flush=True)
    exists = resolved_path is not None and resolved_path.exists()
    print(f"Whether the file exists: {exists}", flush=True)
    if not exists:
        print(f"ERROR: Model file for {model_name} not found in candidates {candidates}!", flush=True)
        sys.exit(1)
        
    size = os.path.getsize(resolved_path)
    print(f"Model size: {size} bytes", flush=True)
    mtime = os.path.getmtime(resolved_path)
    print(f"Last modified date: {time.ctime(mtime)}", flush=True)
    print(f"Framework version: PyTorch {torch.__version__}", flush=True)
    
    try:
        device = 'cuda' if torch.cuda.is_available() else 'cpu'
        checkpoint = torch.load(resolved_path, map_location=device, weights_only=False)
        if not isinstance(checkpoint, dict):
            raise ValueError(f"Loaded checkpoint is not a dictionary, got {type(checkpoint)}")
            
        classes = checkpoint.get('classes')
        model_state = checkpoint.get('model_state', checkpoint)
        
        if classes is None:
            # Attempt to resolve from json
            for lbl in ['plant_labels.json', 'animal_labels.json', 'label_classes.json']:
                lbl_path = resolved_path.parent / lbl
                if not lbl_path.exists():
                    lbl_path = Path(ROOT_DIR) / lbl
                if lbl_path.exists():
                    with open(lbl_path, 'r', encoding='utf-8') as f:
                        data = json.load(f)
                        classes = data.get('classes', data) if isinstance(data, dict) else data
                        checkpoint['classes'] = classes
                        break
                        
        if classes is None or model_state is None:
            raise ValueError("Checkpoint does not contain 'classes' or 'model_state'")
        
        print(f"Input shape: [1, 3, 224, 224]", flush=True)
        print(f"Output shape: [1, {len(classes)}]", flush=True)
        print(f"Number of classes: {len(classes)}", flush=True)
        
        # Pre-cache model in memory
        get_cached_resnet_model(str(resolved_path), classes, model_state)
        checkpoint['_resolved_path'] = str(resolved_path)
        return checkpoint
    except Exception as e:
        print(f"ERROR: Failed to load {model_name}: {e}", flush=True)
        sys.exit(1)


def process_and_debug_image(contents, file, model_name):
    # Step 1: Verify the API
    file_name = file.filename
    file_size = len(contents)
    mime_type = file.content_type
    
    img = Image.open(io.BytesIO(contents)).convert('RGB')
    width, height = img.size
    img_hash = hashlib.sha256(contents).hexdigest()
    
    print(f"=== API Verification for {model_name} ===", flush=True)
    print(f"Uploaded file name: {file_name}", flush=True)
    print(f"File size: {file_size} bytes", flush=True)
    print(f"MIME type: {mime_type}", flush=True)
    print(f"Image width: {width}", flush=True)
    print(f"Image height: {height}", flush=True)
    print(f"SHA256 hash of the uploaded image: {img_hash}", flush=True)
    
    # Step 2: Verify Image Loading
    os.makedirs("debug", exist_ok=True)
    img.save("debug/original.jpg")
    
    transform = get_image_inference_transform()
    if transform is None:
        raise RuntimeError('torchvision is unavailable')
    tensor = transform(img)
    tensor_batch = torch.unsqueeze(tensor, 0)
    
    # Save preprocessed
    transforms_mod = get_transforms()
    std = torch.tensor([0.229, 0.224, 0.225]).view(3, 1, 1)
    mean = torch.tensor([0.485, 0.456, 0.406]).view(3, 1, 1)
    unnorm_tensor = tensor * std + mean
    unnorm_tensor = torch.clamp(unnorm_tensor, 0, 1)
    unnorm_img = transforms_mod.ToPILImage()(unnorm_tensor)
    unnorm_img.save("debug/preprocessed.png")
    
    print(f"=== Image Preprocessing Verification for {model_name} ===", flush=True)
    print(f"shape: {tensor_batch.shape}", flush=True)
    print(f"dtype: {tensor_batch.dtype}", flush=True)
    print(f"min pixel value: {tensor.min().item():.4f}", flush=True)
    print(f"max pixel value: {tensor.max().item():.4f}", flush=True)
    print(f"mean pixel value: {tensor.mean().item():.4f}", flush=True)
    
    return img, tensor_batch


app = FastAPI(
    title="AGRI-DIAGNOSE AI Microservice",
    description="Python FastAPI endpoints for Plant Pathology, Livestock Health, Crop & Yield recommendations",
    version="1.0.0"
)

# Enable CORS for cross-service calls
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

@app.on_event("startup")
async def startup_event():
    global PLANT_CHECKPOINT, ANIMAL_CHECKPOINT
    logger.info("Pre-loading and caching all models at startup...")
    try:
        load_crop_recommender()
        load_crop_metadata()
        load_yield_predictor()
        load_yield_metadata()
        load_price_predictor()
        load_price_metadata()
        load_fertilizer_recommender()
        load_fertilizer_metadata()
        
        # Verify and load Plant model
        PLANT_CHECKPOINT = verify_and_load_model_startup(
            [
                'plant_disease_model.pth',
                'ai/models/image_classification/plant_disease_model.pth',
                'ai/models/image_classification/plant_resnet9_best.pt',
                'ai/models/image_classification/plant_resnet9.pt'
            ],
            'Plant Disease Model'
        )
        
        # Verify and load Animal model
        ANIMAL_CHECKPOINT = verify_and_load_model_startup(
            [
                'animal_disease_model.pth',
                'ai/models/image_classification/animal_disease_model.pth',
                'ai/models/image_classification/animal_resnet9_best.pt',
                'ai/models/image_classification/animal_resnet9.pt'
            ],
            'Animal Disease Model'
        )
        logger.info("All AI models pre-loaded and cached successfully.")
    except Exception as e:
        logger.error(f"Error during startup model pre-loading: {str(e)}")
        sys.exit(1)

@app.exception_handler(HTTPException)
async def http_exception_handler(request: Request, exc: HTTPException):
    logger.error(f"HTTP error occurred: {exc.status_code} - {exc.detail}")
    return JSONResponse(
        status_code=exc.status_code,
        content={"detail": exc.detail}
    )

@app.exception_handler(Exception)
async def global_exception_handler(request: Request, exc: Exception):
    logger.error(f"Global unhandled error: {str(exc)}", exc_info=True)
    return JSONResponse(
        status_code=500,
        content={"detail": f"An unexpected error occurred: {str(exc)}"}
    )

# --- REQUEST SCHEMAS ---

class CropRecInputs(BaseModel):
    state: Optional[str] = None
    district: Optional[str] = None
    season: Optional[str] = None
    soilType: Optional[str] = None
    nitrogen: float
    phosphorus: float
    potassium: float
    temperature: float
    humidity: float
    rainfall: float

class YieldInputs(BaseModel):
    area: float
    rainfall: float
    fertilizer: float
    soil: str
    crop: str
    temperature: float
    humidity: float

class PriceInputs(BaseModel):
    feature1: Optional[float] = None
    feature2: Optional[float] = None
    feature3: Optional[float] = None
    crop: Optional[str] = None
    state: Optional[str] = None
    market: Optional[str] = None
    month: Optional[str] = None

class FertilizerInputs(BaseModel):
    nitrogen: float
    phosphorus: float
    potassium: float
    temperature: float
    humidity: float
    moisture: float
    crop: Optional[str] = None
    soilType: Optional[str] = None
    cropType: Optional[str] = None

# --- ENDPOINTS ---

ANIMAL_DISEASE_METADATA = {
    "cattle_foot_and_mouth": {
        "animalType": "Cow",
        "diseaseName": "Foot and Mouth Disease (FMD)",
        "description": "Foot-and-Mouth Disease (FMD) is a highly contagious, severe viral disease affecting cloven-hoofed animals including cattle, sheep, goats, and pigs. It is caused by an Aphthovirus of the Picornaviridae family and causes devastating losses in milk yield and productivity.",
        "severity": "High",
        "symptoms": "High fever, excessive drooling, sticky stringy saliva, painful vesicles/blisters on the mouth, tongue, dental pad, and interdigital space of hooves, severe lameness, and decreased milk yield.",
        "treatment": "Symptomatic and supportive treatment. Wash oral and hoof lesions with mild antiseptics (e.g. 1% potassium permanganate or 2% boric acid), apply topical soothing ointments on lesions, and provide soft, nutritious mash feed.",
        "isolationGuidance": "IMMEDIATE STRICT QUARANTINE. Isolate the infected animal at least 100 meters away from healthy livestock. Prohibit movement of vehicles and personnel between pens.",
        "emergencyAdvice": "Immediately notify the local state veterinary officer and village panchayat. Disinfect animal housing with 4% sodium carbonate or 2% caustic soda solution. Suspend milk collection and cattle trade from the premises.",
        "nearbyVet": "Dr. Rajesh Sharma, District Vet Clinic (+91 9888888888)"
    },
    "cattle_lumpy": {
        "animalType": "Cow",
        "diseaseName": "Lumpy Skin Disease (LSD)",
        "description": "Lumpy Skin Disease is a vector-borne poxviral disease caused by the Lumpy skin disease virus (Capripoxvirus genus). Characterized by nodular skin lesions across the body, high fever, and secondary bacterial infections.",
        "severity": "Medium",
        "symptoms": "Persistent high fever (40-41.5°C), eruption of firm round circumscribed skin nodules (2-5cm) over head, neck, udder, and limbs, enlarged superficial lymph nodes, edema of limbs and dewlap, and purulent nasal discharge.",
        "treatment": "Supportive care with broad-spectrum antibiotics to prevent secondary bacterial infections, non-steroidal anti-inflammatory drugs (NSAIDs) for fever and pain, and topical antiseptic sprays with fly repellents on open lesions.",
        "isolationGuidance": "Strict isolation in fly-proof and mosquito-screened sheds. Intensive vector control (spray sheds with deltamethrin or cypermethrin to eliminate stable flies, mosquitoes, and ticks).",
        "emergencyAdvice": "Report outbreak to animal husbandry authorities. Administer homologous goat pox / LSD vaccine to all healthy livestock in the 5 km radius.",
        "nearbyVet": "Dr. Rajesh Sharma, District Vet Clinic (+91 9888888888)"
    },
    "cattle_healthy": {
        "animalType": "Cow",
        "diseaseName": "Healthy Cattle",
        "description": "The cattle exhibits excellent physiological condition with clear eyes, moist muzzle, glossy coat, normal rumination, and absence of cutaneous or mucosal lesions.",
        "severity": "Low",
        "symptoms": "No abnormal symptoms. Normal body temperature (38.5-39.2°C), normal appetite, active cud chewing, smooth skin coat, and solid dung consistency.",
        "treatment": "Maintain balanced green and dry fodder, ad-libitum clean water, mineral mixture supplementation, and scheduled routine deworming and preventive vaccination.",
        "isolationGuidance": "None required. Safe to mingle with the herd and pasture.",
        "emergencyAdvice": "Maintain standard biosafety, periodic shed disinfection, and adhere to annual FMD, HS, and BQ vaccination schedules.",
        "nearbyVet": "Dr. Rajesh Sharma, District Vet Clinic (+91 9888888888)"
    },
    "dog_demodicosis": {
        "animalType": "Dog",
        "diseaseName": "Demodectic Mange",
        "description": "Demodicosis (Demodectic Mange or Red Mange) is an inflammatory parasitic skin disease in canines caused by overpopulation of Demodex canis mites within hair follicles and sebaceous glands, often associated with immature or compromised immune systems.",
        "severity": "Medium",
        "symptoms": "Patchy or generalized hair loss (alopecia), erythema (red skin), scaling, crusting, comedones, and hyperpigmentation, frequently around the periocular region, muzzle, and paws.",
        "treatment": "Isoxazoline therapeutics (fluralaner, sarolaner, or afoxolaner) as prescribed by a veterinarian, medicated benzoyl peroxide antibacterial baths, and treatment of secondary pyoderma.",
        "isolationGuidance": "Isolate from other young or immunocompromised dogs as a precaution. Thoroughly wash and sanitize dog bedding and brushes.",
        "emergencyAdvice": "Consult a registered veterinarian for skin scraping microscopy to confirm mite counts and rule out systemic endocrine conditions.",
        "nearbyVet": "Dr. Rajesh Sharma, District Vet Clinic (+91 9888888888)"
    },
    "dog_dermatitis": {
        "animalType": "Dog",
        "diseaseName": "Canine Atopic Dermatitis",
        "description": "Canine Atopic Dermatitis is a genetically predisposed inflammatory and pruritic allergic skin disease caused by hypersensitivity to environmental allergens like pollen, dust mites, and molds.",
        "severity": "Low",
        "symptoms": "Intense chronic itching (pruritus), frequent licking and chewing of paws, rubbing face and ears, red irritated skin around muzzle, ventral abdomen, and groin, secondary yeast odor.",
        "treatment": "Multimodal allergic management: anti-pruritic medications (Oclacitinib / Lokivetmab / Antihistamines), soothing colloidal oatmeal or chlorhexidine baths, essential fatty acid supplements, and hypoallergenic diet.",
        "isolationGuidance": "None required. Atopic dermatitis is an allergic condition and completely non-contagious to other animals or humans.",
        "emergencyAdvice": "Avoid known environmental allergens, maintain clean indoor air with HEPA filtration, and prevent self-trauma using an Elizabethan collar if severe.",
        "nearbyVet": "Dr. Rajesh Sharma, District Vet Clinic (+91 9888888888)"
    },
    "dog_fungal_infections": {
        "animalType": "Dog",
        "diseaseName": "Fungal Skin Infection",
        "description": "Fungal dermatosis is an opportunistic fungal/yeast infection (often Malassezia pachydermatis or Microsporum spp.) that colonizes canine epidermis, typically triggered by moisture, allergies, or underlying immune dysfunction.",
        "severity": "Medium",
        "symptoms": "Greasy, malodorous skin, dark hyperpigmentation, lichenification (thickened elephant skin), circular patches of alopecia, severe scratching, and brown ear discharge.",
        "treatment": "Topical antifungal and antibacterial shampoos (ketoconazole + chlorhexidine 2-3 times weekly), topical miconazole cream, and systemic oral antifungals (itraconazole/fluconazole) under vet supervision.",
        "isolationGuidance": "Isolate the pet from other animals until fungal lesions resolve. Wear protective gloves when bathing and wash hands thoroughly.",
        "emergencyAdvice": "Keep skin folds clean and dry. Wash pet grooming clippers and bedding in hot water with antifungal disinfectant.",
        "nearbyVet": "Dr. Rajesh Sharma, District Vet Clinic (+91 9888888888)"
    },
    "dog_healthy": {
        "animalType": "Dog",
        "diseaseName": "Healthy Dog",
        "description": "The canine demonstrates healthy dermatological and physical signs: lustrous coat, clear dermatological barrier with no visible erythema, alopecia, or parasitic infestation.",
        "severity": "Low",
        "symptoms": "Clean, smooth skin without scabs or redness, lustrous and intact hair coat, bright alert eyes, moist cool nose, and normal energy levels.",
        "treatment": "Routine balanced canine nutrition, daily exercise, regular monthly tick/flea prevention, and annual core vaccination boosters (DHPP + Rabies).",
        "isolationGuidance": "None required.",
        "emergencyAdvice": "Maintain regular grooming and veterinary wellness checks twice a year.",
        "nearbyVet": "Dr. Rajesh Sharma, District Vet Clinic (+91 9888888888)"
    },
    "dog_hypersensitivity": {
        "animalType": "Dog",
        "diseaseName": "Flea Allergy Dermatitis / Hypersensitivity",
        "description": "Flea Allergy Dermatitis (FAD) is a severe hypersensitivity reaction to antigenic proteins present in flea saliva injected during feeding. Even a single flea bite can trigger severe generalized pruritus.",
        "severity": "Medium",
        "symptoms": "Severe itching, hair loss on tail base and lower back, hot spots, scabs.",
        "treatment": "Strict flea control treatments. Corticosteroids for temporary itch relief.",
        "isolationGuidance": "Treat all household pets for fleas. Vacuum carpets regularly.",
        "emergencyAdvice": "Eradicate fleas from host and environment. Prevent secondary bacterial infection.",
        "nearbyVet": "Dr. Rajesh Sharma, District Vet Clinic (+91 9888888888)"
    },
    "dog_ringworm": {
        "animalType": "Dog",
        "diseaseName": "Ringworm Infection",
        "description": "Ringworm is a contagious superficial fungal infection of keratinized tissue (hair, skin, claws) caused by dermatophyte fungi, primarily Microsporum canis. It is a major zoonotic disease transmissible to humans.",
        "severity": "Medium",
        "symptoms": "Circular patches of hair loss with red raised edges and scaly centers. Brittle hair.",
        "treatment": "Topical miconazole/clotrimazole creams. Lime sulfur dips.",
        "isolationGuidance": "Strict isolation. Zoonotic pathogen: can infect humans. Use gloves when handling.",
        "emergencyAdvice": "Disinfect entire household environment. Keep children away from infected pet.",
        "nearbyVet": "Dr. Rajesh Sharma, District Vet Clinic (+91 9888888888)"
    },
    "goat_healthy": {
        "animalType": "Goat",
        "diseaseName": "Healthy Goat",
        "description": "The goat shows optimal physiological health, normal rumination, clear conjunctiva, alert carriage, clean fleece/hair, and steady herd movement.",
        "severity": "Low",
        "symptoms": "Normal herd behavior, healthy coat, bright eyes, solid fecal pellets, good rumination.",
        "treatment": "Standard mineral mixtures and clean drinking water.",
        "isolationGuidance": "None required.",
        "emergencyAdvice": "Follow standard deworming and PPR vaccination calendar.",
        "nearbyVet": "Dr. Amit Verma, Regional Outpost (+91 9444455555)"
    },
    "goat_unhealthy": {
        "animalType": "Goat",
        "diseaseName": "Unhealthy Goat (General Sickness)",
        "description": "The goat displays systemic signs of distress or illness (such as Contagious Ecthyma / Orf, Caprine Respiratory Infection, or Enterotoxemia). Prompt clinical intervention is required.",
        "severity": "Medium",
        "symptoms": "Lethargy, isolation from herd, poor appetite, dull eyes, rough coat, mild nasal discharge.",
        "treatment": "Symptomatic support. Deworming treatment. Keep in draft-free warm shelter.",
        "isolationGuidance": "Isolate in a separate pen as precaution until a vet examines.",
        "emergencyAdvice": "Observe for fever, respiratory distress, or severe diarrhea.",
        "nearbyVet": "Dr. Amit Verma, Regional Outpost (+91 9444455555)"
    }
}


def resolve_plant_disease_details(disease_name: str):
    dn = disease_name.lower()
    
    if "healthy" in dn:
        crop_prefix = disease_name.split("___")[0].replace("_", " ").title() if "___" in disease_name else "Crop"
        disease = f"{crop_prefix} Healthy Leaf"
        severity = 0.0
        desc = f"The {crop_prefix.lower()} foliage appears healthy with no visible signs of infectious disease."
        symptoms = "None. Green, normal leaf turgor and structure."
        actions = "Maintain regular monitoring, watering, and fertilization."
        chemical = "None."
        organic = "None needed. Keep utilizing organic compost."
        prevent = "Maintain good sanitation and weed control."
        treatment = "No treatment required."
    elif "potato" in dn:
        if "late" in dn or "blight" in dn:
            disease = "Potato Late Blight (Phytophthora infestans)"
            severity = 0.75
            desc = "Late blight is a devastating disease caused by the water mold Phytophthora infestans."
            symptoms = "Dark, water-soaked patches on leaves that enlarge rapidly, white velvet mold on leaf undersides, dry rot in tubers."
            actions = "Destroy infected foliage. Halt overhead sprinkling; apply drip irrigation instead."
            chemical = "Apply Mancozeb @ 2g/liter or Metalaxyl-M + Mancozeb formulation @ 2.5g/liter."
            organic = "Spray copper oxychloride suspensions. Ensure proper earthing-up."
            prevent = "Plant certified disease-free seed tubers. Keep a minimum 3-year crop rotation."
            treatment = "Apply late blight fungicides (Mancozeb or Metalaxyl-M) immediately. Stop overhead watering."
        else:
            disease = "Potato Early Blight (Alternaria solani)"
            severity = 0.35
            desc = "Early blight is caused by the fungus Alternaria solani. It primarily affects older foliage, causing defoliation."
            symptoms = "Dark spots with concentric target-like rings on older leaves, yellowing and leaf drops."
            actions = "Prune lower leaves that touch the ground. Apply straw mulch."
            chemical = "Spray Chlorothalonil @ 2g/liter or Copper fungicides at 7-10 day intervals."
            organic = "Apply compost tea or spray Bacillus subtilis formulations."
            prevent = "Rotate crops. Maintain proper soil fertilization. Avoid overhead irrigation."
            treatment = "Apply Early Blight fungicides (Chlorothalonil or Copper). Prune infected lower foliage."
    elif "tomato" in dn:
        disease = "Tomato Early Blight (Alternaria solani)"
        severity = 0.35
        desc = "Early blight is caused by the fungus Alternaria solani. It primarily affects older foliage, causing defoliation."
        symptoms = "Dark spots with concentric target-like rings on older leaves, yellowing and leaf drops."
        actions = "Prune lower leaves that touch the ground. Apply straw mulch."
        chemical = "Spray Chlorothalonil @ 2g/liter or Copper fungicides at 7-10 day intervals."
        organic = "Apply compost tea or spray Bacillus subtilis formulations."
        prevent = "Rotate crops. Maintain proper soil fertilization. Avoid overhead irrigation."
        treatment = "Apply Early Blight fungicides (Chlorothalonil or Copper). Prune infected lower foliage."
    elif "corn" in dn or "maize" in dn or "rust" in dn:
        disease = "Corn Common Rust (Puccinia sorghi)"
        severity = 0.30
        desc = "Common rust is caused by the fungus Puccinia sorghi. It is favored by high humidity."
        symptoms = "Golden-brown to reddish-orange powdery pustules on both upper and lower leaf surfaces."
        actions = "Tillage to bury infected crop residues. Harvest early if crop is mature."
        chemical = "Foliar application of Mancozeb or Pyraclostrobin if rust appears early."
        organic = "Dust with sulfur powder. Use neem oil extracts."
        prevent = "Plant rust-resistant hybrids. Balance nitrogen application."
        treatment = "Apply rust-specific fungicides if infestation is early. Prune heavily infested foliage."
    elif "rice" in dn:
        disease = "Rice Blast (Magnaporthe oryzae)"
        severity = 0.45
        desc = "Rice blast is caused by the fungus Magnaporthe oryzae. It is one of the most destructive diseases of rice worldwide."
        symptoms = "Spindle-shaped spots on leaves with gray or whitish centers and brown borders. Leaf collar rot and neck rot."
        actions = "Avoid excessive nitrogen application. Keep fields flooded. Remove infected crop residues."
        chemical = "Foliar spray of Tricyclazole 75% WP @ 120g/acre or Carbendazim 50% WP @ 200g/acre."
        organic = "Spray Pseudomonas fluorescens formulation @ 5g/liter or Neem oil @ 3%."
        prevent = "Use resistant rice varieties. Treat seeds before sowing. Maintain proper spacing."
        treatment = "Spray Tricyclazole or Carbendazim. Regulate nitrogen application and irrigation."
    else:
        if "___" in disease_name:
            crop_prefix, disease_suffix = disease_name.split("___", 1)
            crop_prefix = crop_prefix.replace("_", " ").title()
            disease_suffix = disease_suffix.replace("_", " ").title()
            disease = f"{crop_prefix} {disease_suffix}"
        else:
            disease = disease_name.replace("_", " ").title()
            parts = disease.split()
            crop_prefix = parts[0] if len(parts) > 0 else "Crop"
            disease_suffix = " ".join(parts[1:]) if len(parts) > 1 else "Infection"
            
        severity = 0.50
        desc = f"{disease} is an agricultural pathology affecting {crop_prefix} plants, which can disrupt photosynthesis and potentially impact crop yield."
        symptoms = f"Visual signs of {disease_suffix.lower()} (spots, lesions, mold, or discoloration) on the leaves of the {crop_prefix} plant."
        actions = f"Isolate affected {crop_prefix} plants, prune heavily infected foliage, and ensure proper field sanitation."
        chemical = f"Apply a crop-approved broad-spectrum fungicide or bactericide suitable for {crop_prefix} protection."
        organic = "Spray organic neem oil extract (3% solution) or copper oxychloride to suppress pathogen spread."
        prevent = f"Utilize certified disease-free seed/clones, practice crop rotation, and maintain balanced NPK nutrition."
        treatment = f"Apply appropriate treatment (chemical: {chemical} or organic: {organic}) and regulate field irrigation."

    return {
        "diseaseName": disease,
        "severityScore": severity,
        "description": desc,
        "symptoms": symptoms,
        "immediateActions": actions,
        "treatment": treatment,
        "recommendedMedicines": chemical,
        "organicTreatment": organic,
        "preventiveMeasures": prevent
    }

@app.post("/api/predictPlant")
async def predict_plant(file: UploadFile = File(...)):
    global PLANT_CHECKPOINT
    logger.info(f"🌱 Plant disease prediction requested for file: {file.filename}")
    
    try:
        contents = await file.read()
        img, tensor = process_and_debug_image(contents, file, 'Plant Disease Model')
        
        if PLANT_CHECKPOINT is None:
            raise HTTPException(status_code=500, detail="Plant classification model is not loaded")
            
        classes = PLANT_CHECKPOINT.get('classes')
        model_state = PLANT_CHECKPOINT.get('model_state')
        
        # Load model using get_cached_resnet_model
        model_key = PLANT_CHECKPOINT.get('_resolved_path', 'plant_disease_model.pth')
        model = get_cached_resnet_model(model_key, classes, model_state)
        
        with torch.no_grad():
            logits = model(tensor)
            probs = torch.softmax(logits, dim=1).cpu().numpy()[0]
            
        # Step 4: Print Raw Model Output
        print(f"=== Raw Model Output for Plant Disease Model ===", flush=True)
        print(probs.tolist(), flush=True)
        
        top_idx = int(np.argmax(probs))
        disease = classes[top_idx]
        confidence = float(probs[top_idx]*100.0)
        
        # Step 5: Verify Class Mapping
        print(f"=== Class Mapping Verification for Plant Disease Model ===", flush=True)
        print(f"Predicted class index: {top_idx}", flush=True)
        print(f"Predicted class label: {disease}", flush=True)
        print(f"Label loaded from checkpoint classes list: {classes[top_idx]}", flush=True)
        assert len(classes) == len(probs), f"Classes length ({len(classes)}) != probs length ({len(probs)})"
        
        # Step 6: Verify Disease Metadata
        meta = resolve_plant_disease_details(disease)
        print(f"=== Disease Metadata Verification for Plant Disease Model ===", flush=True)
        print(f"Metadata lookup key: {disease}", flush=True)
        print(f"Selected metadata: {meta}", flush=True)
        
        top3_indices = np.argsort(probs)[::-1][:3]
        top3_preds = [
            {
                "class": format_plant_display_name(classes[idx]),
                "confidence": float(probs[idx] * 100.0)
            }
            for idx in top3_indices if idx < len(classes)
        ]
        
        return {
            'diseaseName': format_plant_display_name(disease),
            'confidence': confidence,
            'severityScore': meta['severityScore'],
            'description': meta['description'],
            'symptoms': meta['symptoms'],
            'immediateActions': meta['immediateActions'],
            'treatment': meta['treatment'],
            'recommendedMedicines': meta['recommendedMedicines'],
            'organicTreatment': meta['organicTreatment'],
            'preventiveMeasures': meta['preventiveMeasures'],
            'source': 'model',
            'top3Predictions': top3_preds,
            'uncertain': confidence < 60.0
        }
    except Exception as e:
        logger.error(f"❌ Plant model-based inference failed: {str(e)}", exc_info=True)
        if isinstance(e, HTTPException):
            raise e
        raise HTTPException(status_code=500, detail=f"Model inference failed: {str(e)}")


@app.post("/api/predictAnimal")
async def predict_animal(file: UploadFile = File(...)):
    global ANIMAL_CHECKPOINT
    logger.info(f"🐄 Animal disease prediction requested for file: {file.filename}")
    
    try:
        contents = await file.read()
        img, tensor = process_and_debug_image(contents, file, 'Animal Disease Model')
        
        if ANIMAL_CHECKPOINT is None:
            raise HTTPException(status_code=500, detail="Animal classification model is not loaded")
            
        classes = ANIMAL_CHECKPOINT.get('classes')
        model_state = ANIMAL_CHECKPOINT.get('model_state')
        
        model_key = ANIMAL_CHECKPOINT.get('_resolved_path', 'animal_disease_model.pth')
        model = get_cached_resnet_model(model_key, classes, model_state)
        
        with torch.no_grad():
            logits = model(tensor)
            probs = torch.softmax(logits, dim=1).cpu().numpy()[0]
            
        # Step 4: Print Raw Model Output
        print(f"=== Raw Model Output for Animal Disease Model ===", flush=True)
        print(probs.tolist(), flush=True)
        
        top_idx = int(np.argmax(probs))
        disease = classes[top_idx]
        confidence = float(probs[top_idx]*100.0)
        
        # Step 5: Verify Class Mapping
        print(f"=== Class Mapping Verification for Animal Disease Model ===", flush=True)
        print(f"Predicted class index: {top_idx}", flush=True)
        print(f"Predicted class label: {disease}", flush=True)
        print(f"Label loaded from checkpoint classes list: {classes[top_idx]}", flush=True)
        assert len(classes) == len(probs), f"Classes length ({len(classes)}) != probs length ({len(probs)})"
        
        # Step 6: Verify Disease Metadata
        meta = ANIMAL_DISEASE_METADATA.get(disease, {
            "animalType": "Cow" if "cattle" in disease or "cow" in disease else ("Goat" if "goat" in disease or "sheep" in disease else "Dog"),
            "diseaseName": disease.replace("_", " ").title(),
            "description": f"Diagnostic analysis of {disease.replace('_', ' ')} in animal.",
            "severity": "Medium",
            "symptoms": "General symptoms observed. Please consult a veterinarian.",
            "treatment": "Supportive care. Provide clean water and nutrition.",
            "isolationGuidance": "Isolate the animal from the rest of the herd/pets as a precaution.",
            "emergencyAdvice": "Consult a local veterinarian for a complete diagnostic evaluation.",
            "nearbyVet": "Dr. Rajesh Sharma, District Vet Clinic (+91 9888888888)"
        })
        print(f"=== Disease Metadata Verification for Animal Disease Model ===", flush=True)
        print(f"Metadata lookup key: {disease}", flush=True)
        print(f"Selected metadata: {meta}", flush=True)
        
        top3_indices = np.argsort(probs)[::-1][:3]
        top3_preds = [
            {
                "class": classes[idx].replace("_", " ").title(),
                "confidence": float(probs[idx] * 100.0)
            }
            for idx in top3_indices if idx < len(classes)
        ]
        
        return {
            "animalType": meta["animalType"],
            "diseaseName": meta["diseaseName"],
            "description": meta.get("description", f"Dermatological and physical condition {meta['diseaseName']} in {meta['animalType']}."),
            "confidence": confidence,
            "severity": meta["severity"],
            "symptoms": meta["symptoms"],
            "treatment": meta["treatment"],
            "isolationGuidance": meta["isolationGuidance"],
            "emergencyAdvice": meta["emergencyAdvice"],
            "nearbyVet": meta["nearbyVet"],
            "source": "model",
            "top3Predictions": top3_preds,
            "uncertain": confidence < 60.0
        }
    except Exception as e:
        logger.error(f"❌ Animal model-based inference failed: {str(e)}", exc_info=True)
        if isinstance(e, HTTPException):
            raise e
        raise HTTPException(status_code=500, detail=f"Model inference failed: {str(e)}")



def _estimate_soil_ph(soil_type: Optional[str]) -> float:
    if soil_type is None:
        return 7.0
    value = soil_type.lower()
    if 'acid' in value:
        return 6.0
    if 'alkal' in value or 'saline' in value:
        return 8.0
    if 'clay' in value:
        return 6.8
    if 'sandy' in value:
        return 7.2
    if 'loam' in value:
        return 7.0
    return 7.0


@app.post("/api/cropRecommendation")
async def crop_recommendation(inputs: CropRecInputs):
    # Try to use a saved XGBoost model if available
    try:
        model = load_crop_recommender()
        metadata = load_crop_metadata()
        if model is not None:
            df = pd.DataFrame([{
                'N': inputs.nitrogen,
                'P': inputs.phosphorus,
                'K': inputs.potassium,
                'temperature': inputs.temperature,
                'humidity': inputs.humidity,
                'ph': _estimate_soil_ph(inputs.soilType),
                'rainfall': inputs.rainfall
            }])
            try:
                preds = model.predict(df)
                if hasattr(model, 'predict_proba'):
                    proba = model.predict_proba(df)
                    conf = float(np.max(proba) * 100.0)
                else:
                    conf = 90.0
                predicted_crop = _decode_label(preds[0], metadata.get('label_classes'))
                display_crop = _format_crop_display_name(predicted_crop)

                expected_yield = 4.5
                reason = f"Balanced NPK and moderate rainfall suited for {display_crop} growth."
                if "rice" in predicted_crop.lower():
                    expected_yield = 3.9
                    reason = "Paddy rice requires waterlogged soil profile and clayey conditions."
                elif "grapes" in predicted_crop.lower():
                    expected_yield = 12.0
                    reason = "Horticultural grapes require elevated Potassium (K) levels for sugar and skin development."
                elif "chickpea" in predicted_crop.lower():
                    expected_yield = 1.9
                    reason = "Chickpea is a legume fixing atmospheric nitrogen, requiring higher Phosphorus (P) for root nodulation."
                elif "mung" in predicted_crop.lower():
                    expected_yield = 1.3
                    reason = "Short-duration pulse highly suited for semi-arid and low moisture regions."

                # Print verification logs
                logger.info(f"Crop Recommendation Input (Raw): {inputs.dict()}")
                logger.info(f"Crop Recommendation Input (Scaled/Processed): {df.to_dict(orient='records')}")
                logger.info(f"Crop Recommendation Prediction (Raw): {preds[0]}, Decoded: {predicted_crop}")

                return {
                    'recommendedCrop': display_crop,
                    'confidence': conf,
                    'expectedYield': expected_yield,
                    'reason': reason,
                    'source': 'model'
                }
            except Exception as e:
                logger.error(f"Error predicting crop: {str(e)}", exc_info=True)
                raise HTTPException(status_code=500, detail=f"Model prediction failed: {str(e)}")
        else:
            raise HTTPException(status_code=500, detail="Crop recommendation model not loaded")
    except Exception as e:
        if isinstance(e, HTTPException):
            raise e
        logger.error(f"Error in crop recommendation: {str(e)}", exc_info=True)
        raise HTTPException(status_code=500, detail=f"Recommendation failed: {str(e)}")


@app.post("/api/yieldPrediction")
async def yield_prediction(inputs: YieldInputs):
    # Try to use saved RF regressor
    try:
        model = load_yield_predictor()
        metadata = load_yield_metadata()
        if model is not None:
            df = pd.DataFrame([{
                'area': inputs.area,
                'rainfall': inputs.rainfall,
                'fertilizer': inputs.fertilizer,
                'temperature': inputs.temperature,
                'humidity': inputs.humidity
            }])
            feature_columns = metadata.get('feature_columns', ['area', 'rainfall', 'fertilizer', 'temperature', 'humidity'])
            df = _align_features(df, feature_columns)
            try:
                pred = model.predict(df)
                predicted_yield = float(pred[0])

                # Print verification logs
                logger.info(f"Yield Prediction Input (Raw): {inputs.dict()}")
                logger.info(f"Yield Prediction Input (Scaled/Processed): {df.to_dict(orient='records')}")
                logger.info(f"Yield Prediction Prediction (Raw): {predicted_yield}")

                return {'predictedYield': predicted_yield, 'unit': 'Metric Tons', 'source': 'model'}
            except Exception as e:
                logger.error(f"Error predicting yield: {str(e)}", exc_info=True)
                raise HTTPException(status_code=500, detail=f"Model prediction failed: {str(e)}")
        else:
            raise HTTPException(status_code=500, detail="Yield prediction model not loaded")
    except Exception as e:
        if isinstance(e, HTTPException):
            raise e
        logger.error(f"Error in yield prediction: {str(e)}", exc_info=True)
        raise HTTPException(status_code=500, detail=f"Prediction failed: {str(e)}")


@app.post("/api/pricePrediction")
async def price_prediction(inputs: PriceInputs):
    # Try model-based RF price predictor
    try:
        model = load_price_predictor()
        metadata = load_price_metadata()
        if model is not None:
            if inputs.feature1 is None or inputs.feature2 is None or inputs.feature3 is None:
                raise HTTPException(status_code=400, detail='feature1, feature2, and feature3 are required for price prediction')
            df = pd.DataFrame([{
                'feature1': inputs.feature1,
                'feature2': inputs.feature2,
                'feature3': inputs.feature3
            }])
            feature_columns = metadata.get('feature_columns', ['feature1', 'feature2', 'feature3'])
            df = _align_features(df, feature_columns)
            try:
                pred = model.predict(df)
                expected_price = float(pred[0])

                # Print verification logs
                logger.info(f"Price Prediction Input (Raw): {inputs.dict()}")
                logger.info(f"Price Prediction Input (Scaled/Processed): {df.to_dict(orient='records')}")
                logger.info(f"Price Prediction Prediction (Raw): {expected_price}")

                return {'expectedPrice': expected_price, 'currency': 'INR per Quintal', 'source': 'model'}
            except Exception as e:
                logger.error(f"Error predicting price: {str(e)}", exc_info=True)
                raise HTTPException(status_code=500, detail=f"Model prediction failed: {str(e)}")
        else:
            raise HTTPException(status_code=500, detail="Price prediction model not loaded")
    except Exception as e:
        if isinstance(e, HTTPException):
            raise e
        logger.error(f"Error in price prediction: {str(e)}", exc_info=True)
        raise HTTPException(status_code=500, detail=f"Prediction failed: {str(e)}")


def _normalize_category(value):
    return str(value).strip().lower().replace(' ', '_')


def _decode_label(prediction, label_classes):
    if label_classes is None:
        return str(prediction)
    try:
        predicted_index = int(prediction)
    except (TypeError, ValueError):
        return str(prediction)
    if 0 <= predicted_index < len(label_classes):
        return str(label_classes[predicted_index])
    return str(prediction)


def _align_features(frame, expected_columns):
    aligned = frame.copy()
    for column in expected_columns:
        if column not in aligned.columns:
            aligned[column] = 0
    return aligned[expected_columns]


def _format_crop_display_name(crop_name):
    if crop_name is None:
        return 'Unknown Crop'
    return str(crop_name).replace('_', ' ').strip().title()


def map_crop_to_crop_type(crop: str) -> str:
    c = crop.lower()
    if 'rice' in c or 'paddy' in c:
        return 'Paddy'
    if 'maize' in c or 'corn' in c:
        return 'Maize'
    if 'wheat' in c:
        return 'Wheat'
    if 'barley' in c:
        return 'Barley'
    if 'cotton' in c:
        return 'Cotton'
    if 'groundnut' in c or 'ground nut' in c or 'peanut' in c:
        return 'Ground Nuts'
    if 'millet' in c:
        return 'Millets'
    if 'oil' in c:
        return 'Oil seeds'
    if 'pulse' in c or 'gram' in c or 'bean' in c or 'pea' in c:
        return 'Pulses'
    if 'sugarcane' in c:
        return 'Sugarcane'
    if 'tobacco' in c:
        return 'Tobacco'
    return 'Paddy' # Default fallback


def guess_soil_type_for_crop(crop_type: str) -> str:
    ct = crop_type.lower()
    if 'paddy' in ct:
        return 'Clayey'
    if 'wheat' in ct or 'barley' in ct:
        return 'Loamy'
    if 'cotton' in ct:
        return 'Black'
    if 'ground nuts' in ct:
        return 'Sandy'
    return 'Loamy' # Default fallback


@app.post("/api/fertilizerRecommendation")
async def fertilizer_recommendation(inputs: FertilizerInputs):
    # Try model-based prediction first
    try:
        model = load_fertilizer_recommender()
        metadata = load_fertilizer_metadata()
        if model is not None:
            # Map inputs soilType and cropType using smart fallback mapping if missing
            soil_value = inputs.soilType
            if soil_value is None:
                mapped_crop = map_crop_to_crop_type(inputs.crop or "unknown")
                soil_value = guess_soil_type_for_crop(mapped_crop)
            
            crop_value = inputs.cropType
            if crop_value is None:
                crop_value = map_crop_to_crop_type(inputs.crop or "unknown")

            df = pd.DataFrame([{
                'temperature': inputs.temperature,
                'humidity': inputs.humidity,
                'moisture': inputs.moisture,
                'nitrogen': inputs.nitrogen,
                'potassium': inputs.potassium,
                'phosphorous': inputs.phosphorus,
                'soil_type': soil_value,
                'crop_type': crop_value
            }])
            feature_columns = metadata.get('feature_columns', [
                'temperature', 'humidity', 'moisture', 'nitrogen', 'potassium', 'phosphorous', 'soil_type', 'crop_type'
            ])
            for col in feature_columns:
                if col not in df.columns:
                    df[col] = 0
            df = df[feature_columns]

            label_encoders = metadata.get('label_encoders', {})
            for col in ['soil_type', 'crop_type']:
                if col in df.columns and col in label_encoders:
                    categories = label_encoders[col]
                    category_map = {
                        _normalize_category(category): idx
                        for idx, category in enumerate(categories)
                    }
                    df[col] = df[col].astype(str).map(lambda value: category_map.get(_normalize_category(value), 0)).astype(float)

            df = df.astype(float)
            try:
                preds = model.predict(df)
                label_classes = metadata.get('label_classes')
                recommended = _decode_label(preds[0], label_classes)
                conf = 92.0
                qty = "120 kg per acre"
                organic = "Apply compost manure @ 5 tons/acre. Spray bio-extracts."
                if "urea" in recommended.lower():
                    qty = "100 kg per acre"
                    organic = "Grow green manure crops. Apply neem seed cake @ 150 kg/acre."
                elif "dap" in recommended.lower():
                    qty = "80 kg per acre"
                    organic = "Apply rock phosphate + Phosphate Solubilizing Bacteria (PSB) cultures."
                elif "mop" in recommended.lower() or "potash" in recommended.lower():
                    qty = "60 kg per acre"
                    organic = "Apply wood ash @ 200 kg per acre or banana peel compost."

                # Print verification logs
                logger.info(f"Fertilizer Recommendation Input (Raw): {inputs.dict()}")
                logger.info(f"Fertilizer Recommendation Input (Scaled/Processed): {df.to_dict(orient='records')}")
                logger.info(f"Fertilizer Recommendation Prediction (Raw): {preds[0]}, Decoded: {recommended}")

                return {
                    'recommendedFertilizer': recommended,
                    'applicationQuantity': qty,
                    'organicAlternatives': organic,
                    'confidence': conf,
                    'source': 'model'
                }
            except Exception as e:
                logger.error(f"Error predicting fertilizer: {str(e)}", exc_info=True)
                raise HTTPException(status_code=500, detail=f"Model prediction failed: {str(e)}")
        else:
            raise HTTPException(status_code=500, detail="Fertilizer recommendation model not loaded")
    except Exception as e:
        if isinstance(e, HTTPException):
            raise e
        logger.error(f"Error in fertilizer recommendation: {str(e)}", exc_info=True)
        raise HTTPException(status_code=500, detail=f"Recommendation failed: {str(e)}")


if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000)
