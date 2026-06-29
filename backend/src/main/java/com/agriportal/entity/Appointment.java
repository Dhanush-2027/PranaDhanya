package com.agriportal.entity;

import jakarta.persistence.*;
import java.time.LocalDate;
import java.time.LocalTime;
import java.time.LocalDateTime;

@Entity
@Table(name = "appointments")
public class Appointment {

    @Id
    @GeneratedValue(strategy = GenerationType.IDENTITY)
    private Long id;

    @ManyToOne(fetch = FetchType.LAZY)
    @JoinColumn(name = "farmer_id", nullable = false)
    private User farmer;

    @ManyToOne(fetch = FetchType.LAZY)
    @JoinColumn(name = "vet_id", nullable = false)
    private User vet;

    @Column(nullable = false)
    private LocalDate appointmentDate;

    @Column(nullable = false)
    private LocalTime appointmentTime;

    private String disease;

    @Column(columnDefinition = "TEXT")
    private String notes;

    @Column(nullable = false)
    private String status = "PENDING"; // PENDING, APPROVED, REJECTED, COMPLETED

    @Column(columnDefinition = "TEXT")
    private String vetDiagnosis;

    @Column(columnDefinition = "TEXT")
    private String vetMedicines;

    @Column(columnDefinition = "TEXT")
    private String vetTreatment;

    private boolean isolationRequired = false;
    private boolean hospitalVisitRequired = false;

    private LocalDateTime timestamp = LocalDateTime.now();

    public Appointment() {
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

    public User getVet() {
        return vet;
    }

    public void setVet(User vet) {
        this.vet = vet;
    }

    public LocalDate getAppointmentDate() {
        return appointmentDate;
    }

    public void setAppointmentDate(LocalDate appointmentDate) {
        this.appointmentDate = appointmentDate;
    }

    public LocalTime getAppointmentTime() {
        return appointmentTime;
    }

    public void setAppointmentTime(LocalTime appointmentTime) {
        this.appointmentTime = appointmentTime;
    }

    public String getDisease() {
        return disease;
    }

    public void setDisease(String disease) {
        this.disease = disease;
    }

    public String getNotes() {
        return notes;
    }

    public void setNotes(String notes) {
        this.notes = notes;
    }

    public String getStatus() {
        return status;
    }

    public void setStatus(String status) {
        this.status = status;
    }

    public String getVetDiagnosis() {
        return vetDiagnosis;
    }

    public void setVetDiagnosis(String vetDiagnosis) {
        this.vetDiagnosis = vetDiagnosis;
    }

    public String getVetMedicines() {
        return vetMedicines;
    }

    public void setVetMedicines(String vetMedicines) {
        this.vetMedicines = vetMedicines;
    }

    public String getVetTreatment() {
        return vetTreatment;
    }

    public void setVetTreatment(String vetTreatment) {
        this.vetTreatment = vetTreatment;
    }

    public boolean isIsolationRequired() {
        return isolationRequired;
    }

    public void setIsolationRequired(boolean isolationRequired) {
        this.isolationRequired = isolationRequired;
    }

    public boolean isHospitalVisitRequired() {
        return hospitalVisitRequired;
    }

    public void setHospitalVisitRequired(boolean hospitalVisitRequired) {
        this.hospitalVisitRequired = hospitalVisitRequired;
    }

    public LocalDateTime getTimestamp() {
        return timestamp;
    }

    public void setTimestamp(LocalDateTime timestamp) {
        this.timestamp = timestamp;
    }
}
