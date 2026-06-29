package com.agriportal.entity;

import jakarta.persistence.*;
import java.time.LocalDateTime;

@Entity
@Table(name = "fertilizer_recommendation_records")
public class FertilizerRecommendationRecord {

    @Id
    @GeneratedValue(strategy = GenerationType.IDENTITY)
    private Long id;

    @ManyToOne(fetch = FetchType.LAZY)
    @JoinColumn(name = "farmer_id", nullable = false)
    private User farmer;

    private Double nitrogen;
    private Double phosphorus;
    private Double potassium;
    private Double temperature;
    private Double humidity;
    private Double moisture;
    private String crop;

    private String recommendedFertilizer;
    private String applicationQuantity;

    @Column(columnDefinition = "TEXT")
    private String organicAlternatives;

    private LocalDateTime timestamp = LocalDateTime.now();

    public FertilizerRecommendationRecord() {
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

    public Double getNitrogen() {
        return nitrogen;
    }

    public void setNitrogen(Double nitrogen) {
        this.nitrogen = nitrogen;
    }

    public Double getPhosphorus() {
        return phosphorus;
    }

    public void setPhosphorus(Double phosphorus) {
        this.phosphorus = phosphorus;
    }

    public Double getPotassium() {
        return potassium;
    }

    public void setPotassium(Double potassium) {
        this.potassium = potassium;
    }

    public Double getTemperature() {
        return temperature;
    }

    public void setTemperature(Double temperature) {
        this.temperature = temperature;
    }

    public Double getHumidity() {
        return humidity;
    }

    public void setHumidity(Double humidity) {
        this.humidity = humidity;
    }

    public Double getMoisture() {
        return moisture;
    }

    public void setMoisture(Double moisture) {
        this.moisture = moisture;
    }

    public String getCrop() {
        return crop;
    }

    public void setCrop(String crop) {
        this.crop = crop;
    }

    public String getRecommendedFertilizer() {
        return recommendedFertilizer;
    }

    public void setRecommendedFertilizer(String recommendedFertilizer) {
        this.recommendedFertilizer = recommendedFertilizer;
    }

    public String getApplicationQuantity() {
        return applicationQuantity;
    }

    public void setApplicationQuantity(String applicationQuantity) {
        this.applicationQuantity = applicationQuantity;
    }

    public String getOrganicAlternatives() {
        return organicAlternatives;
    }

    public void setOrganicAlternatives(String organicAlternatives) {
        this.organicAlternatives = organicAlternatives;
    }

    public LocalDateTime getTimestamp() {
        return timestamp;
    }

    public void setTimestamp(LocalDateTime timestamp) {
        this.timestamp = timestamp;
    }
}
