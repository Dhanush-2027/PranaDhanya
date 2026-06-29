package com.agriportal.repository;

import com.agriportal.entity.CropRecommendationRecord;
import org.springframework.data.jpa.repository.JpaRepository;
import org.springframework.stereotype.Repository;

import java.util.List;

@Repository
public interface CropRecommendationRecordRepository extends JpaRepository<CropRecommendationRecord, Long> {
    List<CropRecommendationRecord> findByFarmerIdOrderByTimestampDesc(Long farmerId);
}
