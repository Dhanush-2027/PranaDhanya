package com.agriportal.service;

import java.util.List;
import java.util.Optional;

import org.slf4j.Logger;
import org.slf4j.LoggerFactory;
import org.springframework.beans.factory.annotation.Autowired;
import org.springframework.security.crypto.password.PasswordEncoder;
import org.springframework.stereotype.Service;
import org.springframework.transaction.annotation.Transactional;

import com.agriportal.dto.SignupRequest;
import com.agriportal.entity.User;
import com.agriportal.entity.UserRole;
import com.agriportal.repository.UserRepository;

import jakarta.annotation.PostConstruct;
import jakarta.persistence.EntityManager;
import jakarta.persistence.PersistenceContext;

@Service
public class UserService {

    private static final Logger logger = LoggerFactory.getLogger(UserService.class);

    @Autowired
    private UserRepository userRepository;

    @Autowired
    private PasswordEncoder passwordEncoder;

    @PersistenceContext
    private EntityManager entityManager;

    @Transactional
    public User registerUser(SignupRequest request) {
        logger.info("=== REGISTRATION ATTEMPT ===");
        logger.info("Username: {}", request.getUsername());
        logger.info("Email: {} (nullable: {})", 
            (request.getEmail() == null || request.getEmail().isEmpty() ? "EMPTY/NULL" : request.getEmail()),
            (request.getEmail() == null || request.getEmail().isEmpty() ? "YES" : "NO"));
        logger.info("Full Name: {}", request.getFullName());
        logger.info("Role: {}", request.getRole());
        
        if (userRepository.existsByUsername(request.getUsername())) {
            String error = "Error: Username is already taken!";
            logger.error(error);
            throw new RuntimeException(error);
        }
        logger.info("✓ Username is unique");

        if (request.getEmail() != null && !request.getEmail().isEmpty() && userRepository.existsByEmail(request.getEmail())) {
            String error = "Error: Email is already in use!";
            logger.error(error);
            throw new RuntimeException(error);
        }
        logger.info("✓ Email is unique (or empty)");

        UserRole role;
        try {
            role = UserRole.valueOf("ROLE_" + request.getRole().toUpperCase());
            logger.info("✓ Role parsed: {}", role);
        } catch (IllegalArgumentException e) {
            logger.warn("Invalid role: {}, defaulting to ROLE_FARMER", request.getRole());
            role = UserRole.ROLE_FARMER;
        }

        // Handle null or empty email
        String emailToStore = (request.getEmail() == null || request.getEmail().trim().isEmpty()) 
            ? null 
            : request.getEmail();

        String encodedPassword = passwordEncoder.encode(request.getPassword());
        logger.info("✓ Password encoded with BCrypt");

        User user = new User(
                request.getUsername(),
                encodedPassword,
                emailToStore,
                request.getFullName(),
                request.getPhone(),
                request.getVillage(),
                request.getDistrict(),
                request.getState(),
                role
        );

        try {
            User savedUser = userRepository.save(user);
            logger.info("✓✓✓ USER SAVED SUCCESSFULLY ✓✓✓");
            logger.info("New User ID: {}", savedUser.getId());
            logger.info("Username: {}", savedUser.getUsername());
            logger.info("Email: {}", savedUser.getEmail());
            logger.info("Role: {}", savedUser.getRole());
            logger.info("=== REGISTRATION COMPLETE ===\n");
            return savedUser;
        } catch (Exception e) {
            logger.error("✗✗✗ FAILED TO SAVE USER ✗✗✗");
            logger.error("Error: {}", e.getMessage());
            logger.error("Full Stack Trace:", e);
            throw new RuntimeException("Failed to save user: " + e.getMessage(), e);
        }
    }

    public Optional<User> findById(Long id) {
        return userRepository.findById(id);
    }

    public Optional<User> findByUsername(String username) {
        return userRepository.findByUsername(username);
    }

    public List<User> getVets() {
        return userRepository.findByRoleAndActive(UserRole.ROLE_VET, true);
    }

    public List<User> getAllUsers() {
        return userRepository.findAll();
    }

    @Transactional
    public User updateUserStatus(Long id, boolean active) {
        User user = userRepository.findById(id)
                .orElseThrow(() -> new RuntimeException("User not found"));
        user.setActive(active);
        return userRepository.save(user);
    }

    @Transactional
    public User resetPassword(Long id, String newPassword) {
        User user = userRepository.findById(id)
                .orElseThrow(() -> new RuntimeException("User not found"));
        user.setPassword(passwordEncoder.encode(newPassword));
        return userRepository.save(user);
    }

    @Transactional
    public User updateProfile(String username, String fullName, String phone, String village, String district, String state) {
        User user = userRepository.findByUsername(username)
                .orElseThrow(() -> new RuntimeException("User not found"));
        user.setFullName(fullName);
        user.setPhone(phone);
        user.setVillage(village);
        user.setDistrict(district);
        user.setState(state);
        return userRepository.save(user);
    }

    @PostConstruct
    @Transactional
    public void initDefaultUsers() {
        // Drop email NOT NULL constraint to align database with entity nullable=true
        try {
            entityManager.createNativeQuery("ALTER TABLE users ALTER COLUMN email DROP NOT NULL").executeUpdate();
            logger.info("✓ Successfully altered users table to drop NOT NULL constraint on email column");
        } catch (Exception e) {
            logger.warn("Could not alter users table email column: {}", e.getMessage());
        }

        // Create or Update Default Admin
        userRepository.findByUsername("admin").ifPresentOrElse(
            admin -> {
                admin.setPassword(passwordEncoder.encode("admin123"));
                userRepository.save(admin);
            },
            () -> {
                User admin = new User(
                        "admin",
                        passwordEncoder.encode("admin123"),
                        "admin@agriportal.gov.in",
                        "System Administrator",
                        "+91 9999999999",
                        "New Delhi",
                        "New Delhi",
                        "Delhi",
                        UserRole.ROLE_ADMIN
                );
                userRepository.save(admin);
            }
        );

        // Create or Update Default Vet
        userRepository.findByUsername("vet1").ifPresentOrElse(
            vet -> {
                vet.setPassword(passwordEncoder.encode("vet123"));
                userRepository.save(vet);
            },
            () -> {
                User vet = new User(
                        "vet1",
                        passwordEncoder.encode("vet123"),
                        "vet1@agriportal.gov.in",
                        "Dr. Rajesh Sharma (Senior Vet Officer)",
                        "+91 9888888888",
                        "Hebbal",
                        "Bengaluru",
                        "Karnataka",
                        UserRole.ROLE_VET
                );
                userRepository.save(vet);
            }
        );

        // Create or Update Default Farmer
        userRepository.findByUsername("farmer1").ifPresentOrElse(
            farmer -> {
                farmer.setPassword(passwordEncoder.encode("farmer123"));
                userRepository.save(farmer);
            },
            () -> {
                User farmer = new User(
                        "farmer1",
                        passwordEncoder.encode("farmer123"),
                        "farmer1@gmail.com",
                        "Ramesh Gowda",
                        "+91 9777777777",
                        "Mandya Village",
                        "Mandya",
                        "Karnataka",
                        UserRole.ROLE_FARMER
                );
                userRepository.save(farmer);
            }
        );
    }
}
