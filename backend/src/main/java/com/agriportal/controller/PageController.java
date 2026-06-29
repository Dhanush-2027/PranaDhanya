package com.agriportal.controller;

import com.agriportal.entity.*;
import com.agriportal.security.services.UserDetailsImpl;
import com.agriportal.service.*;
import org.springframework.beans.factory.annotation.Autowired;
import org.springframework.security.core.annotation.AuthenticationPrincipal;
import org.springframework.stereotype.Controller;
import org.springframework.ui.Model;
import org.springframework.web.bind.annotation.GetMapping;

import java.util.List;
import java.util.Map;

@Controller
public class PageController {

    @Autowired
    private UserService userService;

    @Autowired
    private RecordService recordService;

    @Autowired
    private AppointmentService appointmentService;

    @Autowired
    private NotificationService notificationService;

    @GetMapping("/")
    public String index(Model model, @AuthenticationPrincipal UserDetailsImpl userDetails) {
        if (userDetails != null) {
            String role = userDetails.getRole();
            if ("ROLE_ADMIN".equals(role)) {
                return "redirect:/admin/dashboard";
            } else if ("ROLE_VET".equals(role)) {
                return "redirect:/vet/dashboard";
            } else {
                return "redirect:/farmer/dashboard";
            }
        }
        
        List<User> vets = userService.getVets();
        model.addAttribute("vets", vets);
        return "index";
    }

    @GetMapping("/farmer/dashboard")
    public String farmerDashboard(Model model, @AuthenticationPrincipal UserDetailsImpl userDetails) {
        if (userDetails == null) {
            return "redirect:/";
        }

        User user = userService.findById(userDetails.getId()).orElseThrow();
        model.addAttribute("user", user);

        // Fetch user records
        List<PlantDiseaseRecord> plantRecords = recordService.getPlantRecordsByFarmer(user.getId());
        List<LivestockDiseaseRecord> animalRecords = recordService.getAnimalRecordsByFarmer(user.getId());
        List<CropRecommendationRecord> cropRecords = recordService.getCropRecordsByFarmer(user.getId());
        List<YieldPredictionRecord> yieldRecords = recordService.getYieldRecordsByFarmer(user.getId());
        List<PricePredictionRecord> priceRecords = recordService.getPriceRecordsByFarmer(user.getId());
        List<FertilizerRecommendationRecord> fertilizerRecords = recordService.getFertilizerRecordsByFarmer(user.getId());
        List<Appointment> appointments = appointmentService.getAppointmentsForFarmer(user.getId());
        List<Notification> notifications = notificationService.getNotificationsForUser(user.getId());
        List<Notification> globalAlerts = notificationService.getGlobalNotifications();
        List<User> vets = userService.getVets();

        model.addAttribute("plantRecords", plantRecords);
        model.addAttribute("animalRecords", animalRecords);
        model.addAttribute("cropRecords", cropRecords);
        model.addAttribute("yieldRecords", yieldRecords);
        model.addAttribute("priceRecords", priceRecords);
        model.addAttribute("fertilizerRecords", fertilizerRecords);
        model.addAttribute("appointments", appointments);
        model.addAttribute("notifications", notifications);
        model.addAttribute("globalAlerts", globalAlerts);
        model.addAttribute("vets", vets);

        return "farmer/dashboard";
    }

    @GetMapping("/vet/dashboard")
    public String vetDashboard(Model model, @AuthenticationPrincipal UserDetailsImpl userDetails) {
        if (userDetails == null) {
            return "redirect:/";
        }

        User user = userService.findById(userDetails.getId()).orElseThrow();
        model.addAttribute("user", user);

        // Fetch vet cases and appointments
        List<Appointment> appointments = appointmentService.getAppointmentsForVet(user.getId());
        List<PlantDiseaseRecord> plantCases = recordService.getAllPlantRecords();
        List<LivestockDiseaseRecord> animalCases = recordService.getAllAnimalRecords();
        List<Notification> notifications = notificationService.getNotificationsForUser(user.getId());
        List<Notification> globalAlerts = notificationService.getGlobalNotifications();

        model.addAttribute("appointments", appointments);
        model.addAttribute("plantCases", plantCases);
        model.addAttribute("animalCases", animalCases);
        model.addAttribute("notifications", notifications);
        model.addAttribute("globalAlerts", globalAlerts);

        return "vet/dashboard";
    }

    @GetMapping("/admin/dashboard")
    public String adminDashboard(Model model, @AuthenticationPrincipal UserDetailsImpl userDetails) {
        if (userDetails == null) {
            return "redirect:/";
        }

        User user = userService.findById(userDetails.getId()).orElseThrow();
        model.addAttribute("user", user);

        // Fetch all data for admin
        List<User> allUsers = userService.getAllUsers();
        List<PlantDiseaseRecord> plantCases = recordService.getAllPlantRecords();
        List<LivestockDiseaseRecord> animalCases = recordService.getAllAnimalRecords();
        List<Appointment> appointments = appointmentService.getAllAppointments();
        List<Notification> globalAlerts = notificationService.getGlobalNotifications();
        Map<String, Object> stats = recordService.getSystemStats();

        model.addAttribute("allUsers", allUsers);
        model.addAttribute("plantCases", plantCases);
        model.addAttribute("animalCases", animalCases);
        model.addAttribute("appointments", appointments);
        model.addAttribute("globalAlerts", globalAlerts);
        model.addAttribute("stats", stats);

        // Mock Server Health
        model.addAttribute("cpuUsage", "24%");
        model.addAttribute("ramUsage", "1.8 GB / 4.0 GB");
        model.addAttribute("storageUsage", "12.4 GB / 40.0 GB");
        model.addAttribute("dbStatus", "Connected (PostgreSQL 15)");
        model.addAttribute("networkTraffic", "128 kb/s");

        return "admin/dashboard";
    }
}
