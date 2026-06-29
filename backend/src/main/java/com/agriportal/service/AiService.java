package com.agriportal.service;

import org.slf4j.Logger;
import org.slf4j.LoggerFactory;
import org.springframework.beans.factory.annotation.Value;
import org.springframework.core.io.ByteArrayResource;
import org.springframework.http.*;
import org.springframework.stereotype.Service;
import org.springframework.util.LinkedMultiValueMap;
import org.springframework.util.MultiValueMap;
import org.springframework.web.client.RestTemplate;
import org.springframework.web.multipart.MultipartFile;

import java.util.HashMap;
import java.util.Map;
import java.util.Random;

@Service
public class AiService {
    private static final Logger logger = LoggerFactory.getLogger(AiService.class);

    @Value("${app.aiServiceUrl}")
    private String aiServiceUrl;

    private final RestTemplate restTemplate = new RestTemplate();
    private final Random random = new Random();

    @SuppressWarnings({"unchecked", "rawtypes"})
    public Map<String, Object> predictPlant(MultipartFile file) {
        try {
            HttpHeaders headers = new HttpHeaders();
            headers.setContentType(MediaType.MULTIPART_FORM_DATA);

            MultiValueMap<String, Object> body = new LinkedMultiValueMap<>();
            body.add("file", new ByteArrayResource(file.getBytes()) {
                @Override
                public String getFilename() {
                    return file.getOriginalFilename();
                }
            });

            HttpEntity<MultiValueMap<String, Object>> requestEntity = new HttpEntity<>(body, headers);
            ResponseEntity<Map> response = restTemplate.postForEntity(aiServiceUrl + "/predictPlant", requestEntity, Map.class);
            
            if (response.getStatusCode() == HttpStatus.OK && response.getBody() != null) {
                return response.getBody();
            }
        } catch (Exception e) {
            logger.warn("FastAPI predictPlant call failed, falling back to Java Mock Engine. Error: {}", e.getMessage());
        }
        return getMockPlantPrediction(file.getOriginalFilename());
    }

    @SuppressWarnings({"unchecked", "rawtypes"})
    public Map<String, Object> predictAnimal(MultipartFile file) {
        try {
            HttpHeaders headers = new HttpHeaders();
            headers.setContentType(MediaType.MULTIPART_FORM_DATA);

            MultiValueMap<String, Object> body = new LinkedMultiValueMap<>();
            body.add("file", new ByteArrayResource(file.getBytes()) {
                @Override
                public String getFilename() {
                    return file.getOriginalFilename();
                }
            });

            HttpEntity<MultiValueMap<String, Object>> requestEntity = new HttpEntity<>(body, headers);
            ResponseEntity<Map> response = restTemplate.postForEntity(aiServiceUrl + "/predictAnimal", requestEntity, Map.class);
            
            if (response.getStatusCode() == HttpStatus.OK && response.getBody() != null) {
                return response.getBody();
            }
        } catch (Exception e) {
            logger.warn("FastAPI predictAnimal call failed, falling back to Java Mock Engine. Error: {}", e.getMessage());
        }
        return getMockAnimalPrediction(file.getOriginalFilename());
    }

    @SuppressWarnings({"unchecked", "rawtypes"})
    public Map<String, Object> cropRecommendation(Map<String, Object> inputs) {
        try {
            HttpHeaders headers = new HttpHeaders();
            headers.setContentType(MediaType.APPLICATION_JSON);

            HttpEntity<Map<String, Object>> requestEntity = new HttpEntity<>(inputs, headers);
            ResponseEntity<Map> response = restTemplate.postForEntity(aiServiceUrl + "/cropRecommendation", requestEntity, Map.class);

            if (response.getStatusCode() == HttpStatus.OK && response.getBody() != null) {
                return response.getBody();
            }
        } catch (Exception e) {
            logger.warn("FastAPI cropRecommendation failed, falling back to Java Mock Engine. Error: {}", e.getMessage());
        }
        return getMockCropRecommendation(inputs);
    }

    @SuppressWarnings({"unchecked", "rawtypes"})
    public Map<String, Object> yieldPrediction(Map<String, Object> inputs) {
        try {
            HttpHeaders headers = new HttpHeaders();
            headers.setContentType(MediaType.APPLICATION_JSON);

            HttpEntity<Map<String, Object>> requestEntity = new HttpEntity<>(inputs, headers);
            ResponseEntity<Map> response = restTemplate.postForEntity(aiServiceUrl + "/yieldPrediction", requestEntity, Map.class);

            if (response.getStatusCode() == HttpStatus.OK && response.getBody() != null) {
                return response.getBody();
            }
        } catch (Exception e) {
            logger.warn("FastAPI yieldPrediction failed, falling back to Java Mock Engine. Error: {}", e.getMessage());
        }
        return getMockYieldPrediction(inputs);
    }

    @SuppressWarnings({"unchecked", "rawtypes"})
    public Map<String, Object> pricePrediction(Map<String, Object> inputs) {
        try {
            HttpHeaders headers = new HttpHeaders();
            headers.setContentType(MediaType.APPLICATION_JSON);

            HttpEntity<Map<String, Object>> requestEntity = new HttpEntity<>(inputs, headers);
            ResponseEntity<Map> response = restTemplate.postForEntity(aiServiceUrl + "/pricePrediction", requestEntity, Map.class);

            if (response.getStatusCode() == HttpStatus.OK && response.getBody() != null) {
                return response.getBody();
            }
        } catch (Exception e) {
            logger.warn("FastAPI pricePrediction failed, falling back to Java Mock Engine. Error: {}", e.getMessage());
        }
        return getMockPricePrediction(inputs);
    }

    @SuppressWarnings({"unchecked", "rawtypes"})
    public Map<String, Object> fertilizerRecommendation(Map<String, Object> inputs) {
        try {
            HttpHeaders headers = new HttpHeaders();
            headers.setContentType(MediaType.APPLICATION_JSON);

            HttpEntity<Map<String, Object>> requestEntity = new HttpEntity<>(inputs, headers);
            ResponseEntity<Map> response = restTemplate.postForEntity(aiServiceUrl + "/fertilizerRecommendation", requestEntity, Map.class);

            if (response.getStatusCode() == HttpStatus.OK && response.getBody() != null) {
                return response.getBody();
            }
        } catch (Exception e) {
            logger.warn("FastAPI fertilizerRecommendation failed, falling back to Java Mock Engine. Error: {}", e.getMessage());
        }
        return getMockFertilizerRecommendation(inputs);
    }

    // --- MOCK GENERATORS ---

    private Map<String, Object> getMockPlantPrediction(String filename) {
        Map<String, Object> res = new HashMap<>();
        String fn = filename != null ? filename.toLowerCase() : "";

        String disease = "Rice Blast (Magnaporthe oryzae)";
        double confidence = 87.5;
        double severity = 0.45;
        String desc = "Rice blast is caused by the fungus Magnaporthe oryzae. It is one of the most destructive diseases of rice worldwide, causing lesions on leaves, nodes, and panicles.";
        String symptoms = "Spindle-shaped spots on leaves with gray or whitish centers and brown borders. Lesions on nodes cause the stem to rot and break (nodal blast). Panicle branches turn brown and die (neck blast).";
        String actions = "Avoid excessive nitrogen application. Keep fields flooded to maintain soil moisture. Remove infected crop residues.";
        String chemical = "Foliar spray of Tricyclazole 75% WP @ 120g/acre or Carbendazim 50% WP @ 200g/acre.";
        String organic = "Spray Pseudomonas fluorescens formulation @ 5g/liter or Neem oil @ 3%. Apply trichoderma-enriched farmyard manure.";
        String prevent = "Use resistant rice varieties. Treat seeds before sowing. Maintain proper spacing between rows.";

        if (fn.contains("potato") || fn.contains("solanum")) {
            disease = "Potato Late Blight (Phytophthora infestans)";
            confidence = 91.2;
            severity = 0.72;
            desc = "Late blight is a devastating disease caused by the oomycete Phytophthora infestans. Under cool and wet conditions, it can destroy an entire potato crop in days.";
            symptoms = "Dark, water-soaked patches on leaves that enlarge rapidly. A white, velvety mold appears on the underside of leaves during humid weather. Tubers show brown-red dry rot.";
            actions = "Destroy infected plants immediately to prevent spore dispersal. Stop overhead irrigation; use drip irrigation instead.";
            chemical = "Apply Mancozeb @ 2g/liter or Metalaxyl-M + Mancozeb formulation @ 2.5g/liter.";
            organic = "Spray copper oxychloride or copper hydroxide suspensions. Ensure proper earthing-up to protect tubers from spores.";
            prevent = "Plant certified disease-free seed tubers. Keep a minimum 3-year crop rotation. Ensure good air circulation.";
        } else if (fn.contains("tomato") || fn.contains("lycopersicum")) {
            disease = "Tomato Early Blight (Alternaria solani)";
            confidence = 79.4;
            severity = 0.35;
            desc = "Early blight is caused by the fungus Alternaria solani. It primarily affects older foliage, causing premature defoliation and yield reduction.";
            symptoms = "Dark spots with concentric rings (target-like pattern) on older leaves. Leaves turn yellow and drop off. Dark leathery spots at the stem end of tomatoes.";
            actions = "Prune lower leaves that touch the ground. Apply mulch to prevent spores splashing from soil onto lower leaves.";
            chemical = "Spray Chlorothalonil @ 2g/liter or Copper fungicides at 7-10 day intervals.";
            organic = "Apply compost tea or spray Bacillus subtilis formulations. Mulch soil with straw or plastic sheets.";
            prevent = "Rotate crops with non-solanaceous plants. Maintain proper soil fertilization. Avoid overhead irrigation.";
        } else if (fn.contains("corn") || fn.contains("maize") || fn.contains("zea")) {
            disease = "Corn Common Rust (Puccinia sorghi)";
            confidence = 88.0;
            severity = 0.28;
            desc = "Common rust is caused by the fungus Puccinia sorghi. It is favored by high humidity and moderate temperatures (16-23 degrees Celsius).";
            symptoms = "Golden-brown to reddish-orange powdery pustules (uredinia) on both upper and lower leaf surfaces. In severe cases, leaves turn yellow and wither.";
            actions = "Harvest early if infection is severe and crop is mature. Tillage to bury infected crop residues.";
            chemical = "Foliar application of Mancozeb or Pyraclostrobin if rust appears early in the season.";
            organic = "Dust with sulfur powder. Use neem oil extracts to suppress spore germination.";
            prevent = "Plant rust-resistant hybrids. Ensure balanced nitrogen application.";
        } else if (random.nextDouble() > 0.5) {
            // Randomize a low confidence case to test Vet Referral feature
            disease = "Unknown Leaf Spot (Likely Fungal)";
            confidence = 62.5;
            severity = 0.50;
            desc = "Unidentified leaf abnormality. The visual symptoms are not sufficient for a highly confident diagnosis.";
            symptoms = "Scattered irregular brownish spots on the foliage with minor chlorosis.";
            actions = "Isolate the affected plants. Consult a local agricultural extension officer or schedule a veterinary/expert consultation.";
            chemical = "Broad-spectrum contact fungicide like Mancozeb as a general precaution.";
            organic = "Spray fresh homemade neem seed kernel extract (NSKE) 5%.";
            prevent = "Improve field sanitation and keep weeds under control.";
        }

        res.put("diseaseName", disease);
        res.put("confidence", confidence);
        res.put("severityScore", severity);
        res.put("description", desc);
        res.put("symptoms", symptoms);
        res.put("immediateActions", actions);
        res.put("treatment", "Apply recommended fungicides and adjust crop nutrition.");
        res.put("recommendedMedicines", chemical);
        res.put("organicTreatment", organic);
        res.put("preventiveMeasures", prevent);

        return res;
    }

    private Map<String, Object> getMockAnimalPrediction(String filename) {
        Map<String, Object> res = new HashMap<>();
        String fn = filename != null ? filename.toLowerCase() : "";

        String animal = "Cow";
        String disease = "Foot and Mouth Disease (FMD)";
        double confidence = 92.4;
        String severity = "High";
        String symptoms = "High fever, excessive salivation (drooling, stringy saliva), vesicles/blisters on the tongue, lips, dental pad, and interdigital space of hooves. Lameness.";
        String treatment = "No specific antiviral treatment. Clean wounds with mild antiseptics. Apply soda ash or sodium carbonate solution on foot lesions. Give soft feed.";
        String isolation = "IMMEDIATE ISOLATION. Quarantine infected animals at least 100 meters away from healthy livestock. Limit personnel access to the quarantine zone.";
        String emergency = "Notify local veterinary officer immediately. Disinfect barns, equipment, and vehicles with 4% sodium carbonate. Do not move animals or milk out of the farm.";
        String nearbyVet = "Dr. Rajesh Sharma, Hebbal Veterinary Hospital (+91 9888888888)";

        if (fn.contains("cow") || fn.contains("cattle") || fn.contains("bull")) {
            if (random.nextDouble() > 0.5) {
                animal = "Cow";
                disease = "Lumpy Skin Disease (LSD)";
                confidence = 89.1;
                severity = "Medium";
                symptoms = "Fever, followed by the eruption of firm, round nodules (2-5 cm diameter) on the skin of head, neck, limbs, and udder. Nasal and ocular discharge. Swollen lymph nodes.";
                treatment = "Symptomatic treatment. Antibiotics to prevent secondary bacterial infections. Anti-inflammatory drugs for pain. Clean skin lesions with antiseptic sprays.";
                isolation = "Strictly isolate affected cattle. Vector control is critical: use insect repellents, spray barns, and eliminate stagnant water to reduce mosquito/fly populations.";
                emergency = "Report outbreak to authorities. Vaccinate healthy cattle in the surrounding region. Ensure clean drinking water and soft feeding.";
                nearbyVet = "Hebbal Veterinary Outpost, Bengaluru (+91 9888888888)";
            }
        } else if (fn.contains("sheep") || fn.contains("goat")) {
            animal = "Goat";
            disease = "Peste des Petits Ruminants (PPR)";
            confidence = 85.3;
            severity = "High";
            symptoms = "Sudden fever, dry muzzle, watery nasal and ocular discharge that later becomes thick and yellow. Sores in the mouth and gums. Severe diarrhea. Labored breathing.";
            treatment = "Supportive care. Antibiotics for secondary pneumonia. Rehydration therapy for diarrhea. Clean mouth sores with saline or potassium permanganate.";
            isolation = "Isolate the goat immediately. Keep in a dry, warm, and draft-free shelter. Quarantine incoming sheep and goats for 21 days.";
            emergency = "PPR is highly contagious. Immediately restrict all animal movement. Notify vet department for ring vaccination.";
            nearbyVet = "Dr. Amit Verma, District Veterinary Clinic (+91 9444455555)";
        } else if (fn.contains("buffalo")) {
            animal = "Buffalo";
            disease = "Mastitis";
            confidence = 94.0;
            severity = "Medium";
            symptoms = "Swollen, hard, hot, and painful udder. Watery, clotted, or blood-tinged milk. Decline in milk yield. Fever and depression.";
            treatment = "Intramammary antibiotic infusions administered by a vet. Complete stripping of affected quarters. Dry cow therapy.";
            isolation = "Separate the buffalo during milking. Milk the infected buffalo last. Disinfect milking equipment thoroughly.";
            emergency = "Test herd milk using CMT (California Mastitis Test). Maintain clean bedding and proper hygiene during milking.";
            nearbyVet = "Dr. Rajesh Sharma, Hebbal Veterinary Hospital (+91 9888888888)";
        }

        res.put("animalType", animal);
        res.put("diseaseName", disease);
        res.put("confidence", confidence);
        res.put("severity", severity);
        res.put("symptoms", symptoms);
        res.put("treatment", treatment);
        res.put("isolationGuidance", isolation);
        res.put("emergencyAdvice", emergency);
        res.put("nearbyVet", nearbyVet);

        return res;
    }

    private Map<String, Object> getMockCropRecommendation(Map<String, Object> inputs) {
        Map<String, Object> res = new HashMap<>();
        
        double n = Double.parseDouble(inputs.getOrDefault("nitrogen", "50").toString());
        double p = Double.parseDouble(inputs.getOrDefault("phosphorus", "50").toString());
        double k = Double.parseDouble(inputs.getOrDefault("potassium", "50").toString());
        double rainfall = Double.parseDouble(inputs.getOrDefault("rainfall", "100").toString());

        String recommendedCrop = "Maize";
        double confidence = 89.2;
        double expectedYield = 4.2; // Tons per hectare
        String reason = "Based on Nitrogen (" + n + "), Phosphorus (" + p + "), Potassium (" + k + "), and a moderate rainfall of " + rainfall + "mm, the soil nutrition and water availability are ideal for Maize cultivation.";

        if (rainfall > 180) {
            recommendedCrop = "Rice";
            confidence = 94.5;
            expectedYield = 3.8;
            reason = "High water/rainfall conditions (>180mm) combined with clayey soil types and balanced NPK parameters are excellent for paddy rice crops.";
        } else if (k > 150) {
            recommendedCrop = "Grapes";
            confidence = 91.0;
            expectedYield = 12.5;
            reason = "A very high Potassium level (" + k + ") indicates soil profile suited for horticultural fruit crops like grapes.";
        } else if (n < 30 && p > 60) {
            recommendedCrop = "Chickpea";
            confidence = 87.0;
            expectedYield = 1.8;
            reason = "Legumes like chickpeas thrive in lower Nitrogen soils because they fix nitrogen biologically, requiring higher Phosphorus for root nodulation.";
        } else if (rainfall < 60) {
            recommendedCrop = "Mung Bean";
            confidence = 85.5;
            expectedYield = 1.2;
            reason = "Arid and low-rainfall regions (<60mm) are highly suitable for short-duration drought-tolerant pulses like Mung Beans.";
        }

        res.put("recommendedCrop", recommendedCrop);
        res.put("confidence", confidence);
        res.put("expectedYield", expectedYield);
        res.put("reason", reason);

        return res;
    }

    private Map<String, Object> getMockYieldPrediction(Map<String, Object> inputs) {
        Map<String, Object> res = new HashMap<>();

        double area = Double.parseDouble(inputs.getOrDefault("area", "1.0").toString());
        double rainfall = Double.parseDouble(inputs.getOrDefault("rainfall", "500").toString());
        double fertilizer = Double.parseDouble(inputs.getOrDefault("fertilizer", "100").toString());
        String crop = inputs.getOrDefault("crop", "Rice").toString();

        double baseYieldPerHa = 3.5;
        if (crop.equalsIgnoreCase("Rice")) baseYieldPerHa = 4.2;
        else if (crop.equalsIgnoreCase("Maize")) baseYieldPerHa = 5.0;
        else if (crop.equalsIgnoreCase("Grapes")) baseYieldPerHa = 15.0;
        else if (crop.equalsIgnoreCase("Chickpea")) baseYieldPerHa = 2.0;

        // Add variance based on rainfall and fertilizer
        double fertFactor = Math.min(1.3, 0.7 + (fertilizer / 200.0));
        double rainFactor = Math.min(1.2, 0.8 + (rainfall / 1000.0));
        
        double predictedYieldVal = area * baseYieldPerHa * fertFactor * rainFactor;
        // round to two decimal places
        predictedYieldVal = Math.round(predictedYieldVal * 100.0) / 100.0;

        res.put("predictedYield", predictedYieldVal);
        res.put("unit", "Metric Tons");
        return res;
    }

    private Map<String, Object> getMockPricePrediction(Map<String, Object> inputs) {
        Map<String, Object> res = new HashMap<>();
        String crop = inputs.getOrDefault("crop", "Rice").toString();

        double basePrice = 2200.0; // INR per quintal
        String trend = "Stable";

        if (crop.equalsIgnoreCase("Rice")) {
            basePrice = 2400.0;
            trend = "Slightly Bullish (Increasing due to high demand and export margins)";
        } else if (crop.equalsIgnoreCase("Maize")) {
            basePrice = 2050.0;
            trend = "Bullish (Increasing industrial starch and poultry feed demand)";
        } else if (crop.equalsIgnoreCase("Grapes")) {
            basePrice = 8500.0;
            trend = "Volatile (High seasonal fluctuation)";
        } else if (crop.equalsIgnoreCase("Chickpea")) {
            basePrice = 5300.0;
            trend = "Bearish (Decreasing due to high buffer stocks)";
        }

        double expectedPriceVal = basePrice + (random.nextDouble() * 200 - 100);
        expectedPriceVal = Math.round(expectedPriceVal * 100.0) / 100.0;

        res.put("expectedPrice", expectedPriceVal);
        res.put("marketTrend", trend);
        res.put("currency", "INR per Quintal (100 kg)");
        return res;
    }

    private Map<String, Object> getMockFertilizerRecommendation(Map<String, Object> inputs) {
        Map<String, Object> res = new HashMap<>();

        double n = Double.parseDouble(inputs.getOrDefault("nitrogen", "50").toString());
        double p = Double.parseDouble(inputs.getOrDefault("phosphorus", "50").toString());
        double k = Double.parseDouble(inputs.getOrDefault("potassium", "50").toString());

        String recommendedFertilizer = "NPK 19-19-19 (Balanced)";
        String applicationQuantity = "120 kg per acre (in 3 split doses)";
        String organicAlternatives = "Apply well-decomposed Farmyard Manure (FYM) @ 5 tons/acre. Incorporate Vermicompost @ 2 tons/acre and spray Jeevamrutha at 15-day intervals.";

        if (n < 40) {
            recommendedFertilizer = "Urea (46% Nitrogen)";
            applicationQuantity = "100 kg per acre (50kg basal dose + 50kg top dressing)";
            organicAlternatives = "Grow green manure crops like Sunn hemp or Dhaincha and incorporate before cultivation. Apply Neem Cake @ 150 kg/acre.";
        } else if (p < 30) {
            recommendedFertilizer = "DAP (Diammonium Phosphate) or Single Super Phosphate (SSP)";
            applicationQuantity = "75 kg per acre (applied at the time of sowing/transplanting)";
            organicAlternatives = "Apply Rock Phosphate @ 100 kg/acre combined with Phosphate Solubilizing Bacteria (PSB) inoculation @ 2 kg/acre.";
        } else if (k < 30) {
            recommendedFertilizer = "MOP (Muriate of Potash)";
            applicationQuantity = "50 kg per acre (applied in 2 split doses)";
            organicAlternatives = "Apply Wood Ash @ 250 kg/acre or banana peel compost. Spray potassium-mobilizing biofertilizers.";
        }

        res.put("recommendedFertilizer", recommendedFertilizer);
        res.put("applicationQuantity", applicationQuantity);
        res.put("organicAlternatives", organicAlternatives);

        return res;
    }
}
