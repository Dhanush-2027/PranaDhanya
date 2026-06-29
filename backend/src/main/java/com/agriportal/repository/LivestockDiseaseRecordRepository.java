package com.agriportal.repository;

import com.agriportal.entity.LivestockDiseaseRecord;
import org.springframework.data.jpa.repository.JpaRepository;
import org.springframework.stereotype.Repository;

import java.util.List;

@Repository
public interface LivestockDiseaseRecordRepository extends JpaRepository<LivestockDiseaseRecord, Long> {
    List<LivestockDiseaseRecord> findByFarmerIdOrderByTimestampDesc(Long farmerId);
    List<LivestockDiseaseRecord> findAllByOrderByTimestampDesc();
}
