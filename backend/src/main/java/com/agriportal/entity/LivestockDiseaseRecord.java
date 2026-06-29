package com.agriportal.entity;

import jakarta.persistence.*;
import java.time.LocalDateTime;

@Entity
@Table(name = "livestock_disease_records")
public class LivestockDiseaseRecord {

    @Id
    @GeneratedValue(strategy = GenerationType.IDENTITY)
    private Long id;

    @ManyToOne(fetch = FetchType.LAZY)
    @JoinColumn(name = "farmer_id", nullable = false)
    private User farmer;

    @Column(nullable = false)
    private String animalType;

    @Column(nullable = false)
    private String diseaseName;

    private Double confidence;
    private String severity;

    @Column(columnDefinition = "TEXT")
    private String symptoms;

    @Column(columnDefinition = "TEXT")
    private String treatment;

    @Column(columnDefinition = "TEXT")
    private String isolationGuidance;

    @Column(columnDefinition = "TEXT")
    private String emergencyAdvice;

    @Column(columnDefinition = "TEXT")
    private String nearbyVet;

    private String imageUrl;

    private Double latitude;
    private Double longitude;

    private LocalDateTime timestamp = LocalDateTime.now();

    public LivestockDiseaseRecord() {
    }

    public Long getId() {
        return id;
    }

    public void setId(Long id) {
        this.id = id;
    }

    public User getFarmer() {
        return farmer;
    }

    public void setFarmer(User farmer) {
        this.farmer = farmer;
    }

    public String getAnimalType() {
        return animalType;
    }

    public void setAnimalType(String animalType) {
        this.animalType = animalType;
    }

    public String getDiseaseName() {
        return diseaseName;
    }

    public void setDiseaseName(String diseaseName) {
        this.diseaseName = diseaseName;
    }

    public Double getConfidence() {
        return confidence;
    }

    public void setConfidence(Double confidence) {
        this.confidence = confidence;
    }

    public String getSeverity() {
        return severity;
    }

    public void setSeverity(String severity) {
        this.severity = severity;
    }

    public String getSymptoms() {
        return symptoms;
    }

    public void setSymptoms(String symptoms) {
        this.symptoms = symptoms;
    }

    public String getTreatment() {
        return treatment;
    }

    public void setTreatment(String treatment) {
        this.treatment = treatment;
    }

    public String getIsolationGuidance() {
        return isolationGuidance;
    }

    public void setIsolationGuidance(String isolationGuidance) {
        this.isolationGuidance = isolationGuidance;
    }

    public String getEmergencyAdvice() {
        return emergencyAdvice;
    }

    public void setEmergencyAdvice(String emergencyAdvice) {
        this.emergencyAdvice = emergencyAdvice;
    }

    public String getNearbyVet() {
        return nearbyVet;
    }

    public void setNearbyVet(String nearbyVet) {
        this.nearbyVet = nearbyVet;
    }

    public String getImageUrl() {
        return imageUrl;
    }

    public void setImageUrl(String imageUrl) {
        this.imageUrl = imageUrl;
    }

    public Double getLatitude() {
        return latitude;
    }

    public void setLatitude(Double latitude) {
        this.latitude = latitude;
    }

    public Double getLongitude() {
        return longitude;
    }

    public void setLongitude(Double longitude) {
        this.longitude = longitude;
    }

    public LocalDateTime getTimestamp() {
        return timestamp;
    }

    public void setTimestamp(LocalDateTime timestamp) {
        this.timestamp = timestamp;
    }
}
