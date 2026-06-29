package com.agriportal.service;

import com.agriportal.entity.*;
import com.agriportal.repository.*;
import org.springframework.beans.factory.annotation.Autowired;
import org.springframework.stereotype.Service;
import org.springframework.transaction.annotation.Transactional;

import java.time.LocalDateTime;
import java.util.HashMap;
import java.util.List;
import java.util.Map;

@Service
@SuppressWarnings("null")
public class RecordService {

    @Autowired
    private PlantDiseaseRecordRepository plantRepo;

    @Autowired
    private LivestockDiseaseRecordRepository animalRepo;

    @Autowired
    private CropRecommendationRecordRepository cropRepo;

    @Autowired
    private YieldPredictionRecordRepository yieldRepo;

    @Autowired
    private PricePredictionRecordRepository priceRepo;

    @Autowired
    private FertilizerRecommendationRecordRepository fertilizerRepo;

    @Transactional
    public PlantDiseaseRecord savePlantRecord(User farmer, String diseaseName, Double confidence, Double severity,
                                             Map<String, Object> aiResponse, String imageUrl, Double lat, Double lon) {
        PlantDiseaseRecord record = new PlantDiseaseRecord();
        record.setFarmer(farmer);
        record.setDiseaseName(diseaseName);
        record.setConfidence(confidence);
        record.setSeverityScore(severity);
        record.setDescription((String) aiResponse.get("description"));
        record.setSymptoms((String) aiResponse.get("symptoms"));
        record.setImmediateActions((String) aiResponse.get("immediateActions"));
        record.setTreatment((String) aiResponse.get("treatment"));
        record.setRecommendedMedicines((String) aiResponse.get("recommendedMedicines"));
        record.setOrganicTreatment((String) aiResponse.get("organicTreatment"));
        record.setPreventiveMeasures((String) aiResponse.get("preventiveMeasures"));
        record.setImageUrl(imageUrl);
        record.setLatitude(lat);
        record.setLongitude(lon);
        record.setTimestamp(LocalDateTime.now());
        return plantRepo.save(record);
    }

    @Transactional
    public LivestockDiseaseRecord saveAnimalRecord(User farmer, String animalType, String diseaseName, Double confidence,
                                                 String severity, Map<String, Object> aiResponse, String imageUrl, Double lat, Double lon) {
        LivestockDiseaseRecord record = new LivestockDiseaseRecord();
        record.setFarmer(farmer);
        record.setAnimalType(animalType);
        record.setDiseaseName(diseaseName);
        record.setConfidence(confidence);
        record.setSeverity(severity);
        record.setSymptoms((String) aiResponse.get("symptoms"));
        record.setTreatment((String) aiResponse.get("treatment"));
        record.setIsolationGuidance((String) aiResponse.get("isolationGuidance"));
        record.setEmergencyAdvice((String) aiResponse.get("emergencyAdvice"));
        record.setNearbyVet((String) aiResponse.get("nearbyVet"));
        record.setImageUrl(imageUrl);
        record.setLatitude(lat);
        record.setLongitude(lon);
        record.setTimestamp(LocalDateTime.now());
        return animalRepo.save(record);
    }

    @Transactional
    public CropRecommendationRecord saveCropRecord(User farmer, String state, String district, String season, String soilType,
                                                  Double n, Double p, Double k, Double temp, Double hum, Double rain,
                                                  String recCrop, Double confidence, Double expectedYield, String reason) {
        CropRecommendationRecord record = new CropRecommendationRecord();
        record.setFarmer(farmer);
        record.setState(state);
        record.setDistrict(district);
        record.setSeason(season);
        record.setSoilType(soilType);
        record.setNitrogen(n);
        record.setPhosphorus(p);
        record.setPotassium(k);
        record.setTemperature(temp);
        record.setHumidity(hum);
        record.setRainfall(rain);
        record.setRecommendedCrop(recCrop);
        record.setConfidence(confidence);
        record.setExpectedYield(expectedYield);
        record.setReason(reason);
        return cropRepo.save(record);
    }

    @Transactional
    public YieldPredictionRecord saveYieldRecord(User farmer, Double area, Double rainfall, Double fertilizer,
                                                 String crop, Double temp, Double hum, String soil, Double predictedYield) {
        YieldPredictionRecord record = new YieldPredictionRecord();
        record.setFarmer(farmer);
        record.setArea(area);
        record.setRainfall(rainfall);
        record.setFertilizer(fertilizer);
        record.setCrop(crop);
        record.setTemperature(temp);
        record.setHumidity(hum);
        record.setSoil(soil);
        record.setPredictedYield(predictedYield);
        return yieldRepo.save(record);
    }

    @Transactional
    public PricePredictionRecord savePriceRecord(User farmer, String crop, String state, String market, String month,
                                                 Double expectedPrice, String trend) {
        PricePredictionRecord record = new PricePredictionRecord();
        record.setFarmer(farmer);
        record.setCrop(crop);
        record.setState(state);
        record.setMarket(market);
        record.setMonth(month);
        record.setExpectedPrice(expectedPrice);
        record.setMarketTrend(trend);
        return priceRepo.save(record);
    }

    @Transactional
    public FertilizerRecommendationRecord saveFertilizerRecord(User farmer, Double n, Double p, Double k, Double temp,
                                                               Double hum, Double moisture, String crop, String recFert,
                                                               String qty, String organicAlts) {
        FertilizerRecommendationRecord record = new FertilizerRecommendationRecord();
        record.setFarmer(farmer);
        record.setNitrogen(n);
        record.setPhosphorus(p);
        record.setPotassium(k);
        record.setTemperature(temp);
        record.setHumidity(hum);
        record.setMoisture(moisture);
        record.setCrop(crop);
        record.setRecommendedFertilizer(recFert);
        record.setApplicationQuantity(qty);
        record.setOrganicAlternatives(organicAlts);
        return fertilizerRepo.save(record);
    }

    public List<PlantDiseaseRecord> getPlantRecordsByFarmer(Long farmerId) {
        return plantRepo.findByFarmerIdOrderByTimestampDesc(farmerId);
    }

    public List<LivestockDiseaseRecord> getAnimalRecordsByFarmer(Long farmerId) {
        return animalRepo.findByFarmerIdOrderByTimestampDesc(farmerId);
    }

    public List<CropRecommendationRecord> getCropRecordsByFarmer(Long farmerId) {
        return cropRepo.findByFarmerIdOrderByTimestampDesc(farmerId);
    }

    public List<YieldPredictionRecord> getYieldRecordsByFarmer(Long farmerId) {
        return yieldRepo.findByFarmerIdOrderByTimestampDesc(farmerId);
    }

    public List<PricePredictionRecord> getPriceRecordsByFarmer(Long farmerId) {
        return priceRepo.findByFarmerIdOrderByTimestampDesc(farmerId);
    }

    public List<FertilizerRecommendationRecord> getFertilizerRecordsByFarmer(Long farmerId) {
        return fertilizerRepo.findByFarmerIdOrderByTimestampDesc(farmerId);
    }

    public List<PlantDiseaseRecord> getAllPlantRecords() {
        return plantRepo.findAllByOrderByTimestampDesc();
    }

    public List<LivestockDiseaseRecord> getAllAnimalRecords() {
        return animalRepo.findAllByOrderByTimestampDesc();
    }

    @Transactional
    public void deletePlantRecord(Long id) {
        plantRepo.deleteById(id);
    }

    @Transactional
    public void deleteAnimalRecord(Long id) {
        animalRepo.deleteById(id);
    }

    public Map<String, Object> getSystemStats() {
        Map<String, Object> stats = new HashMap<>();
        stats.put("plantRecordsCount", plantRepo.count());
        stats.put("animalRecordsCount", animalRepo.count());
        stats.put("cropPredictionsCount", cropRepo.count());
        stats.put("yieldPredictionsCount", yieldRepo.count());
        stats.put("pricePredictionsCount", priceRepo.count());
        stats.put("fertilizerPredictionsCount", fertilizerRepo.count());
        return stats;
    }
}
