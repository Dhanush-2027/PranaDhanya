package com.agriportal.repository;

import com.agriportal.entity.YieldPredictionRecord;
import org.springframework.data.jpa.repository.JpaRepository;
import org.springframework.stereotype.Repository;

import java.util.List;

@Repository
public interface YieldPredictionRecordRepository extends JpaRepository<YieldPredictionRecord, Long> {
    List<YieldPredictionRecord> findByFarmerIdOrderByTimestampDesc(Long farmerId);
}
