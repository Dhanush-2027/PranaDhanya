package com.agriportal.repository;

import com.agriportal.entity.FertilizerRecommendationRecord;
import org.springframework.data.jpa.repository.JpaRepository;
import org.springframework.stereotype.Repository;

import java.util.List;

@Repository
public interface FertilizerRecommendationRecordRepository extends JpaRepository<FertilizerRecommendationRecord, Long> {
    List<FertilizerRecommendationRecord> findByFarmerIdOrderByTimestampDesc(Long farmerId);
}
