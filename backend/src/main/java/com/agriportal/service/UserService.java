package com.agriportal.service;

import com.agriportal.dto.SignupRequest;
import com.agriportal.entity.User;
import com.agriportal.entity.UserRole;
import com.agriportal.repository.UserRepository;
import jakarta.annotation.PostConstruct;
import org.springframework.beans.factory.annotation.Autowired;
import org.springframework.security.crypto.password.PasswordEncoder;
import org.springframework.stereotype.Service;
import org.springframework.transaction.annotation.Transactional;

import java.util.List;
import java.util.Optional;

@Service
@SuppressWarnings("null")
public class UserService {

    @Autowired
    private UserRepository userRepository;

    @Autowired
    private PasswordEncoder passwordEncoder;

    @Transactional
    public User registerUser(SignupRequest request) {
        if (userRepository.existsByUsername(request.getUsername())) {
            throw new RuntimeException("Error: Username is already taken!");
        }

        if (userRepository.existsByEmail(request.getEmail())) {
            throw new RuntimeException("Error: Email is already in use!");
        }

        UserRole role;
        try {
            role = UserRole.valueOf("ROLE_" + request.getRole().toUpperCase());
        } catch (IllegalArgumentException e) {
            role = UserRole.ROLE_FARMER;
        }

        User user = new User(
                request.getUsername(),
                passwordEncoder.encode(request.getPassword()),
                request.getEmail(),
                request.getFullName(),
                request.getPhone(),
                request.getVillage(),
                request.getDistrict(),
                request.getState(),
                role
        );

        return userRepository.save(user);
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
        // Create Default Admin
        if (!userRepository.existsByUsername("admin")) {
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

        // Create Default Vet
        if (!userRepository.existsByUsername("vet1")) {
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

        // Create Default Farmer
        if (!userRepository.existsByUsername("farmer1")) {
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
    }
}
