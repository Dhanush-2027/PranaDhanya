package com.agriportal.repository;

import com.agriportal.entity.Appointment;
import org.springframework.data.jpa.repository.JpaRepository;
import org.springframework.stereotype.Repository;

import java.util.List;

@Repository
public interface AppointmentRepository extends JpaRepository<Appointment, Long> {
    List<Appointment> findByFarmerIdOrderByAppointmentDateDescAppointmentTimeDesc(Long farmerId);
    List<Appointment> findByVetIdOrderByAppointmentDateDescAppointmentTimeDesc(Long vetId);
    List<Appointment> findAllByOrderByAppointmentDateDescAppointmentTimeDesc();
}
