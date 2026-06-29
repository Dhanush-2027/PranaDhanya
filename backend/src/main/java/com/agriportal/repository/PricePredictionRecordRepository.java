package com.agriportal.repository;

import com.agriportal.entity.PricePredictionRecord;
import org.springframework.data.jpa.repository.JpaRepository;
import org.springframework.stereotype.Repository;

import java.util.List;

@Repository
public interface PricePredictionRecordRepository extends JpaRepository<PricePredictionRecord, Long> {
    List<PricePredictionRecord> findByFarmerIdOrderByTimestampDesc(Long farmerId);
}
