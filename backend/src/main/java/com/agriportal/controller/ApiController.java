package com.agriportal.controller;

import com.agriportal.entity.*;
import com.agriportal.service.*;
import com.agriportal.security.services.UserDetailsImpl;
import org.springframework.beans.factory.annotation.Autowired;
import org.springframework.http.ResponseEntity;
import org.springframework.security.core.annotation.AuthenticationPrincipal;
import org.springframework.web.bind.annotation.*;
import org.springframework.web.multipart.MultipartFile;

import java.io.File;
import java.io.IOException;
import java.nio.file.Files;
import java.nio.file.Path;
import java.nio.file.Paths;
import java.time.LocalDate;
import java.time.LocalTime;
import java.util.HashMap;
import java.util.Map;
import java.util.UUID;

@RestController
@RequestMapping("/api")
public class ApiController {

    @Autowired
    private AiService aiService;

    @Autowired
    private UserService userService;

    @Autowired
    private RecordService recordService;

    @Autowired
    private AppointmentService appointmentService;

    @Autowired
    private NotificationService notificationService;

    private static final String UPLOAD_DIR = "src/main/resources/static/uploads";

    private String saveUploadedFile(MultipartFile file) {
        if (file.isEmpty()) {
            return null;
        }
        try {
            // Ensure directory exists
            File dir = new File(UPLOAD_DIR);
            if (!dir.exists()) {
                dir.mkdirs();
            }

            String filename = UUID.randomUUID().toString() + "_" + file.getOriginalFilename();
            Path path = Paths.get(UPLOAD_DIR, filename);
            Files.write(path, file.getBytes());
            return "/uploads/" + filename;
        } catch (IOException e) {
            // Fallback: return a default dummy URL or absolute path in case directory doesn't exist
            return "/images/placeholder-leaf.jpg";
        }
    }

    @PostMapping("/predictPlant")
    public ResponseEntity<?> predictPlant(@RequestParam("file") MultipartFile file,
                                          @RequestParam(value = "latitude", required = false) Double lat,
                                          @RequestParam(value = "longitude", required = false) Double lon,
                                          @AuthenticationPrincipal UserDetailsImpl userDetails) {
        if (userDetails == null) {
            return ResponseEntity.status(401).body("Unauthorized");
        }

        User user = userService.findById(userDetails.getId()).orElseThrow();
        String imageUrl = saveUploadedFile(file);

        // Fetch AI Prediction
        Map<String, Object> prediction = aiService.predictPlant(file);
        
        String diseaseName = (String) prediction.get("diseaseName");
        Double confidence = (Double) prediction.get("confidence");
        Double severity = (Double) prediction.get("severityScore");

        // Save Record
        PlantDiseaseRecord record = recordService.savePlantRecord(user, diseaseName, confidence, severity, prediction, imageUrl, lat, lon);

        // NDLM Trigger: Regional Cluster Early Warning System Mock
        if (severity != null && severity > 0.6) {
            notificationService.createGlobalNotification(
                    "NDLM Early Warning: Outbreak Alert",
                    "A high-severity case of " + diseaseName + " was recorded in " + user.getVillage() + ", " + user.getDistrict() + ". Farmers are advised to take preventive measures.",
                    "DISEASE_ALERT"
            );
        }

        Map<String, Object> response = new HashMap<>(prediction);
        response.put("recordId", record.getId());
        response.put("imageUrl", imageUrl);

        return ResponseEntity.ok(response);
    }

    @PostMapping("/predictAnimal")
    public ResponseEntity<?> predictAnimal(@RequestParam("file") MultipartFile file,
                                           @RequestParam(value = "latitude", required = false) Double lat,
                                           @RequestParam(value = "longitude", required = false) Double lon,
                                           @AuthenticationPrincipal UserDetailsImpl userDetails) {
        if (userDetails == null) {
            return ResponseEntity.status(401).body("Unauthorized");
        }

        User user = userService.findById(userDetails.getId()).orElseThrow();
        String imageUrl = saveUploadedFile(file);

        // Fetch AI Prediction
        Map<String, Object> prediction = aiService.predictAnimal(file);

        String animalType = (String) prediction.get("animalType");
        String diseaseName = (String) prediction.get("diseaseName");
        Double confidence = (Double) prediction.get("confidence");
        String severity = (String) prediction.get("severity");

        // Save Record
        LivestockDiseaseRecord record = recordService.saveAnimalRecord(user, animalType, diseaseName, confidence, severity, prediction, imageUrl, lat, lon);

        // NDLM Trigger: Livestock Outbreak Warning
        if ("High".equalsIgnoreCase(severity)) {
            notificationService.createGlobalNotification(
                    "NDLM Early Warning: Animal Outbreak Alert",
                    "A severe case of " + diseaseName + " (" + animalType + ") detected in " + user.getVillage() + ". Restrict livestock movements.",
                    "DISEASE_ALERT"
            );
        }

        Map<String, Object> response = new HashMap<>(prediction);
        response.put("recordId", record.getId());
        response.put("imageUrl", imageUrl);

        return ResponseEntity.ok(response);
    }

    @PostMapping("/cropRecommendation")
    public ResponseEntity<?> cropRecommendation(@RequestBody Map<String, Object> inputs,
                                                @AuthenticationPrincipal UserDetailsImpl userDetails) {
        if (userDetails == null) {
            return ResponseEntity.status(401).body("Unauthorized");
        }

        User user = userService.findById(userDetails.getId()).orElseThrow();
        Map<String, Object> rec = aiService.cropRecommendation(inputs);

        String recCrop = (String) rec.get("recommendedCrop");
        Double confidence = (Double) rec.get("confidence");
        Double expectedYield = Double.parseDouble(rec.get("expectedYield").toString());
        String reason = (String) rec.get("reason");

        CropRecommendationRecord record = recordService.saveCropRecord(
                user,
                (String) inputs.get("state"),
                (String) inputs.get("district"),
                (String) inputs.get("season"),
                (String) inputs.get("soilType"),
                Double.parseDouble(inputs.get("nitrogen").toString()),
                Double.parseDouble(inputs.get("phosphorus").toString()),
                Double.parseDouble(inputs.get("potassium").toString()),
                Double.parseDouble(inputs.get("temperature").toString()),
                Double.parseDouble(inputs.get("humidity").toString()),
                Double.parseDouble(inputs.get("rainfall").toString()),
                recCrop,
                confidence,
                expectedYield,
                reason
        );

        Map<String, Object> response = new HashMap<>(rec);
        response.put("recordId", record.getId());
        return ResponseEntity.ok(response);
    }

    @PostMapping("/yieldPrediction")
    public ResponseEntity<?> yieldPrediction(@RequestBody Map<String, Object> inputs,
                                             @AuthenticationPrincipal UserDetailsImpl userDetails) {
        if (userDetails == null) {
            return ResponseEntity.status(401).body("Unauthorized");
        }

        User user = userService.findById(userDetails.getId()).orElseThrow();
        Map<String, Object> pred = aiService.yieldPrediction(inputs);
        Double yieldVal = (Double) pred.get("predictedYield");

        YieldPredictionRecord record = recordService.saveYieldRecord(
                user,
                Double.parseDouble(inputs.get("area").toString()),
                Double.parseDouble(inputs.get("rainfall").toString()),
                Double.parseDouble(inputs.get("fertilizer").toString()),
                (String) inputs.get("crop"),
                Double.parseDouble(inputs.get("temperature").toString()),
                Double.parseDouble(inputs.get("humidity").toString()),
                (String) inputs.get("soil"),
                yieldVal
        );

        Map<String, Object> response = new HashMap<>(pred);
        response.put("recordId", record.getId());
        return ResponseEntity.ok(response);
    }

    @PostMapping("/pricePrediction")
    public ResponseEntity<?> pricePrediction(@RequestBody Map<String, Object> inputs,
                                             @AuthenticationPrincipal UserDetailsImpl userDetails) {
        if (userDetails == null) {
            return ResponseEntity.status(401).body("Unauthorized");
        }

        User user = userService.findById(userDetails.getId()).orElseThrow();
        Map<String, Object> pred = aiService.pricePrediction(inputs);
        Double priceVal = (Double) pred.get("expectedPrice");
        String trend = (String) pred.get("marketTrend");

        PricePredictionRecord record = recordService.savePriceRecord(
                user,
                (String) inputs.get("crop"),
                (String) inputs.get("state"),
                (String) inputs.get("market"),
                (String) inputs.get("month"),
                priceVal,
                trend
        );

        Map<String, Object> response = new HashMap<>(pred);
        response.put("recordId", record.getId());
        return ResponseEntity.ok(response);
    }

    @PostMapping("/fertilizerRecommendation")
    public ResponseEntity<?> fertilizerRecommendation(@RequestBody Map<String, Object> inputs,
                                                      @AuthenticationPrincipal UserDetailsImpl userDetails) {
        if (userDetails == null) {
            return ResponseEntity.status(401).body("Unauthorized");
        }

        User user = userService.findById(userDetails.getId()).orElseThrow();
        Map<String, Object> rec = aiService.fertilizerRecommendation(inputs);

        String fertilizer = (String) rec.get("recommendedFertilizer");
        String qty = (String) rec.get("applicationQuantity");
        String organic = (String) rec.get("organicAlternatives");

        FertilizerRecommendationRecord record = recordService.saveFertilizerRecord(
                user,
                Double.parseDouble(inputs.get("nitrogen").toString()),
                Double.parseDouble(inputs.get("phosphorus").toString()),
                Double.parseDouble(inputs.get("potassium").toString()),
                Double.parseDouble(inputs.get("temperature").toString()),
                Double.parseDouble(inputs.get("humidity").toString()),
                Double.parseDouble(inputs.get("moisture").toString()),
                (String) inputs.get("crop"),
                fertilizer,
                qty,
                organic
        );

        Map<String, Object> response = new HashMap<>(rec);
        response.put("recordId", record.getId());
        return ResponseEntity.ok(response);
    }

    // --- Notifications API ---

    @PostMapping("/notifications/read/{id}")
    public ResponseEntity<?> markNotificationRead(@PathVariable Long id) {
        notificationService.markAsRead(id);
        return ResponseEntity.ok().build();
    }

    // --- Appointments API ---

    @PostMapping("/appointments/book")
    public ResponseEntity<?> bookAppointment(@RequestBody Map<String, String> payload,
                                             @AuthenticationPrincipal UserDetailsImpl userDetails) {
        if (userDetails == null) {
            return ResponseEntity.status(401).body("Unauthorized");
        }

        User farmer = userService.findById(userDetails.getId()).orElseThrow();
        User vet = userService.findById(Long.parseLong(payload.get("vetId"))).orElseThrow();
        LocalDate date = LocalDate.parse(payload.get("date"));
        LocalTime time = LocalTime.parse(payload.get("time"));
        String disease = payload.get("disease");
        String notes = payload.get("notes");

        Appointment app = appointmentService.bookAppointment(farmer, vet, date, time, disease, notes);
        return ResponseEntity.ok(app);
    }

    @PostMapping("/appointments/status")
    public ResponseEntity<?> updateAppointmentStatus(@RequestBody Map<String, String> payload) {
        Long id = Long.parseLong(payload.get("appointmentId"));
        String status = payload.get("status"); // APPROVED or REJECTED
        Appointment app = appointmentService.updateStatus(id, status);
        return ResponseEntity.ok(app);
    }

    @PostMapping("/appointments/diagnosis")
    public ResponseEntity<?> updateAppointmentDiagnosis(@RequestBody Map<String, Object> payload) {
        Long id = Long.parseLong(payload.get("appointmentId").toString());
        String diagnosis = (String) payload.get("diagnosis");
        String medicines = (String) payload.get("medicines");
        String treatment = (String) payload.get("treatment");
        boolean isolation = (Boolean) payload.get("isolation");
        boolean hospital = (Boolean) payload.get("hospital");

        Appointment app = appointmentService.submitVetDiagnosis(id, diagnosis, medicines, treatment, isolation, hospital);
        return ResponseEntity.ok(app);
    }

    // --- Records delete ---

    @DeleteMapping("/records/plant/delete/{id}")
    public ResponseEntity<?> deletePlantRecord(@PathVariable Long id) {
        recordService.deletePlantRecord(id);
        return ResponseEntity.ok().build();
    }

    @DeleteMapping("/records/animal/delete/{id}")
    public ResponseEntity<?> deleteAnimalRecord(@PathVariable Long id) {
        recordService.deleteAnimalRecord(id);
        return ResponseEntity.ok().build();
    }

    // --- Admin Operations ---

    @PostMapping("/admin/users/status")
    public ResponseEntity<?> changeUserStatus(@RequestBody Map<String, Object> payload) {
        Long id = Long.parseLong(payload.get("userId").toString());
        boolean active = (Boolean) payload.get("active");
        User user = userService.updateUserStatus(id, active);
        return ResponseEntity.ok(user);
    }

    @PostMapping("/admin/users/reset-password")
    public ResponseEntity<?> resetUserPassword(@RequestBody Map<String, String> payload) {
        Long id = Long.parseLong(payload.get("userId"));
        String newPassword = payload.get("password");
        User user = userService.resetPassword(id, newPassword);
        return ResponseEntity.ok(user);
    }

    @PostMapping("/admin/alerts/broadcast")
    public ResponseEntity<?> broadcastAlert(@RequestBody Map<String, String> payload) {
        String title = payload.get("title");
        String message = payload.get("message");
        Notification notification = notificationService.createGlobalNotification(title, message, "GOVT_ALERT");
        return ResponseEntity.ok(notification);
    }

    @PutMapping("/profile/update")
    public ResponseEntity<?> updateProfile(@RequestBody Map<String, String> payload,
                                           @AuthenticationPrincipal UserDetailsImpl userDetails) {
        if (userDetails == null) {
            return ResponseEntity.status(401).body("Unauthorized");
        }

        User user = userService.updateProfile(
                userDetails.getUsername(),
                payload.get("fullName"),
                payload.get("phone"),
                payload.get("village"),
                payload.get("district"),
                payload.get("state")
        );
        return ResponseEntity.ok(user);
    }

    // --- Dashboard Data Endpoints for Decoupled Frontend ---

    @GetMapping("/farmer/dashboard-data")
    public ResponseEntity<?> getFarmerDashboardData(@AuthenticationPrincipal UserDetailsImpl userDetails) {
        if (userDetails == null) {
            return ResponseEntity.status(401).body("Unauthorized");
        }
        User user = userService.findById(userDetails.getId()).orElseThrow();
        
        Map<String, Object> data = new HashMap<>();
        data.put("user", user);
        data.put("plantRecords", recordService.getPlantRecordsByFarmer(user.getId()));
        data.put("animalRecords", recordService.getAnimalRecordsByFarmer(user.getId()));
        data.put("cropRecords", recordService.getCropRecordsByFarmer(user.getId()));
        data.put("yieldRecords", recordService.getYieldRecordsByFarmer(user.getId()));
        data.put("priceRecords", recordService.getPriceRecordsByFarmer(user.getId()));
        data.put("fertilizerRecords", recordService.getFertilizerRecordsByFarmer(user.getId()));
        data.put("appointments", appointmentService.getAppointmentsForFarmer(user.getId()));
        data.put("notifications", notificationService.getNotificationsForUser(user.getId()));
        data.put("globalAlerts", notificationService.getGlobalNotifications());
        data.put("vets", userService.getVets());
        return ResponseEntity.ok(data);
    }

    @GetMapping("/vet/dashboard-data")
    public ResponseEntity<?> getVetDashboardData(@AuthenticationPrincipal UserDetailsImpl userDetails) {
        if (userDetails == null) {
            return ResponseEntity.status(401).body("Unauthorized");
        }
        User user = userService.findById(userDetails.getId()).orElseThrow();
        
        Map<String, Object> data = new HashMap<>();
        data.put("user", user);
        data.put("appointments", appointmentService.getAppointmentsForVet(user.getId()));
        data.put("plantCases", recordService.getAllPlantRecords());
        data.put("animalCases", recordService.getAllAnimalRecords());
        data.put("notifications", notificationService.getNotificationsForUser(user.getId()));
        data.put("globalAlerts", notificationService.getGlobalNotifications());
        return ResponseEntity.ok(data);
    }

    @GetMapping("/admin/dashboard-data")
    public ResponseEntity<?> getAdminDashboardData(@AuthenticationPrincipal UserDetailsImpl userDetails) {
        if (userDetails == null) {
            return ResponseEntity.status(401).body("Unauthorized");
        }
        User user = userService.findById(userDetails.getId()).orElseThrow();
        
        Map<String, Object> data = new HashMap<>();
        data.put("user", user);
        data.put("allUsers", userService.getAllUsers());
        data.put("plantCases", recordService.getAllPlantRecords());
        data.put("animalCases", recordService.getAllAnimalRecords());
        data.put("appointments", appointmentService.getAllAppointments());
        data.put("globalAlerts", notificationService.getGlobalNotifications());
        data.put("stats", recordService.getSystemStats());
        
        // Mock Server Health
        data.put("cpuUsage", "24%");
        data.put("ramUsage", "1.8 GB / 4.0 GB");
        data.put("storageUsage", "12.4 GB / 40.0 GB");
        data.put("dbStatus", "Connected (PostgreSQL 15)");
        data.put("networkTraffic", "128 kb/s");
        return ResponseEntity.ok(data);
    }
}
