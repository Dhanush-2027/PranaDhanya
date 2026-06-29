package com.agriportal.repository;

import com.agriportal.entity.PlantDiseaseRecord;
import org.springframework.data.jpa.repository.JpaRepository;
import org.springframework.stereotype.Repository;

import java.util.List;

@Repository
public interface PlantDiseaseRecordRepository extends JpaRepository<PlantDiseaseRecord, Long> {
    List<PlantDiseaseRecord> findByFarmerIdOrderByTimestampDesc(Long farmerId);
    List<PlantDiseaseRecord> findAllByOrderByTimestampDesc();
}
