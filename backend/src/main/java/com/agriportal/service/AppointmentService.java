package com.agriportal.service;

import com.agriportal.entity.Appointment;
import com.agriportal.entity.User;
import com.agriportal.repository.AppointmentRepository;
import org.springframework.beans.factory.annotation.Autowired;
import org.springframework.stereotype.Service;
import org.springframework.transaction.annotation.Transactional;

import java.time.LocalDate;
import java.time.LocalTime;
import java.util.List;

@Service
@SuppressWarnings("null")
public class AppointmentService {

    @Autowired
    private AppointmentRepository appointmentRepo;

    @Autowired
    private NotificationService notificationService;

    @Transactional
    public Appointment bookAppointment(User farmer, User vet, LocalDate date, LocalTime time, String disease, String notes) {
        Appointment app = new Appointment();
        app.setFarmer(farmer);
        app.setVet(vet);
        app.setAppointmentDate(date);
        app.setAppointmentTime(time);
        app.setDisease(disease);
        app.setNotes(notes);
        app.setStatus("PENDING");
        
        Appointment saved = appointmentRepo.save(app);

        // Notify Vet
        notificationService.createNotification(
                vet,
                "New Appointment Request",
                "Farmer " + farmer.getFullName() + " requested an appointment on " + date + " at " + time + ".",
                "APPOINTMENT"
        );

        return saved;
    }

    public List<Appointment> getAppointmentsForFarmer(Long farmerId) {
        return appointmentRepo.findByFarmerIdOrderByAppointmentDateDescAppointmentTimeDesc(farmerId);
    }

    public List<Appointment> getAppointmentsForVet(Long vetId) {
        return appointmentRepo.findByVetIdOrderByAppointmentDateDescAppointmentTimeDesc(vetId);
    }

    public List<Appointment> getAllAppointments() {
        return appointmentRepo.findAllByOrderByAppointmentDateDescAppointmentTimeDesc();
    }

    @Transactional
    public Appointment updateStatus(Long appointmentId, String status) {
        Appointment app = appointmentRepo.findById(appointmentId)
                .orElseThrow(() -> new RuntimeException("Appointment not found"));
        app.setStatus(status);

        Appointment updated = appointmentRepo.save(app);

        // Notify Farmer
        notificationService.createNotification(
                app.getFarmer(),
                "Appointment Status Update",
                "Your appointment request with " + app.getVet().getFullName() + " has been " + status.toLowerCase() + ".",
                "APPOINTMENT"
        );

        return updated;
    }

    @Transactional
    public Appointment submitVetDiagnosis(Long appointmentId, String diagnosis, String medicines, String treatment,
                                          boolean isolation, boolean hospitalVisit) {
        Appointment app = appointmentRepo.findById(appointmentId)
                .orElseThrow(() -> new RuntimeException("Appointment not found"));
        
        app.setVetDiagnosis(diagnosis);
        app.setVetMedicines(medicines);
        app.setVetTreatment(treatment);
        app.setIsolationRequired(isolation);
        app.setHospitalVisitRequired(hospitalVisit);
        app.setStatus("COMPLETED");

        Appointment updated = appointmentRepo.save(app);

        // Notify Farmer
        notificationService.createNotification(
                app.getFarmer(),
                "Veterinary Diagnosis Submitted",
                "Dr. " + app.getVet().getFullName() + " has updated your case diagnosis. Check details in your dashboard.",
                "DIAGNOSIS"
        );

        return updated;
    }
}
