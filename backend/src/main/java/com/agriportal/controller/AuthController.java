package com.agriportal.controller;

import org.slf4j.Logger;
import org.slf4j.LoggerFactory;
import org.springframework.beans.factory.annotation.Autowired;
import org.springframework.http.HttpHeaders;
import org.springframework.http.ResponseCookie;
import org.springframework.http.ResponseEntity;
import org.springframework.security.authentication.AuthenticationManager;
import org.springframework.security.authentication.UsernamePasswordAuthenticationToken;
import org.springframework.security.core.Authentication;
import org.springframework.security.core.context.SecurityContextHolder;
import org.springframework.stereotype.Controller;
import org.springframework.web.bind.annotation.GetMapping;
import org.springframework.web.bind.annotation.ModelAttribute;
import org.springframework.web.bind.annotation.PostMapping;
import org.springframework.web.bind.annotation.RequestBody;
import org.springframework.web.bind.annotation.RequestParam;
import org.springframework.web.bind.annotation.ResponseBody;

import com.agriportal.dto.LoginRequest;
import com.agriportal.dto.SignupRequest;
import com.agriportal.entity.User;
import com.agriportal.security.jwt.JwtUtils;
import com.agriportal.security.services.UserDetailsImpl;
import com.agriportal.service.UserService;

import java.util.HashMap;
import java.util.Map;

import jakarta.servlet.http.HttpServletResponse;
import jakarta.validation.Valid;

@Controller
public class AuthController {

    private static final Logger logger = LoggerFactory.getLogger(AuthController.class);

    @Autowired
    AuthenticationManager authenticationManager;

    @Autowired
    UserService userService;

    @Autowired
    JwtUtils jwtUtils;

    // --- REST API Endpoints ---

    @PostMapping("/api/auth/signin")
    @ResponseBody
    public ResponseEntity<?> authenticateUser(@Valid @RequestBody LoginRequest loginRequest, HttpServletResponse response) {
        logger.info("=== API LOGIN ATTEMPT ===");
        logger.info("Username: {}", loginRequest.getUsername());
        
        try {
            Authentication authentication = authenticationManager.authenticate(
                    new UsernamePasswordAuthenticationToken(loginRequest.getUsername(), loginRequest.getPassword()));

            SecurityContextHolder.getContext().setAuthentication(authentication);
            UserDetailsImpl userDetails = (UserDetailsImpl) authentication.getPrincipal();

            logger.info("✓ Login successful for user: {}", userDetails.getUsername());

            ResponseCookie jwtCookie = jwtUtils.generateJwtCookie(userDetails);
            
            // Also set cookie in response for potential hybrid clients
            response.addHeader(HttpHeaders.SET_COOKIE, jwtCookie.toString());

            String jwt = jwtUtils.generateTokenFromUsername(userDetails.getUsername());

            logger.info("✓ JWT token generated");
            logger.info("=== LOGIN COMPLETE ===\n");

            Map<String, Object> responseBody = new HashMap<>();
            responseBody.put("token", jwt);
            responseBody.put("accessToken", jwt);
            responseBody.put("refreshToken", "");
            responseBody.put("id", userDetails.getId());
            responseBody.put("username", userDetails.getUsername());
            responseBody.put("email", userDetails.getEmail());
            responseBody.put("fullName", userDetails.getFullName());
            responseBody.put("role", userDetails.getRole());

            Map<String, Object> userMap = new HashMap<>();
            userMap.put("id", userDetails.getId());
            userMap.put("name", userDetails.getFullName());
            userMap.put("email", userDetails.getEmail());
            
            String cleanRole = userDetails.getRole();
            if (cleanRole != null && cleanRole.startsWith("ROLE_")) {
                cleanRole = cleanRole.substring(5);
            }
            userMap.put("role", cleanRole);
            responseBody.put("user", userMap);

            return ResponseEntity.ok()
                    .header(HttpHeaders.SET_COOKIE, jwtCookie.toString())
                    .body(responseBody);
        } catch (Exception e) {
            logger.error("✗ Login failed: {}", e.getMessage());
            return ResponseEntity.badRequest().body("Login failed: Invalid credentials");
        }
    }

    @PostMapping("/api/auth/signup")
    @ResponseBody
    public ResponseEntity<?> registerUser(@Valid @RequestBody SignupRequest signUpRequest, HttpServletResponse response) {
        try {
            User user = userService.registerUser(signUpRequest);
            logger.info("✓ User registered via API: {}", user.getUsername());

            // Automatically authenticate user after registration
            Authentication authentication = authenticationManager.authenticate(
                    new UsernamePasswordAuthenticationToken(signUpRequest.getUsername(), signUpRequest.getPassword()));

            SecurityContextHolder.getContext().setAuthentication(authentication);
            UserDetailsImpl userDetails = (UserDetailsImpl) authentication.getPrincipal();

            ResponseCookie jwtCookie = jwtUtils.generateJwtCookie(userDetails);
            response.addHeader(HttpHeaders.SET_COOKIE, jwtCookie.toString());

            String jwt = jwtUtils.generateTokenFromUsername(userDetails.getUsername());

            logger.info("✓ API registration auto-login successful for user: {}", userDetails.getUsername());

            Map<String, Object> responseBody = new HashMap<>();
            responseBody.put("token", jwt);
            responseBody.put("accessToken", jwt);
            responseBody.put("refreshToken", "");
            responseBody.put("id", userDetails.getId());
            responseBody.put("username", userDetails.getUsername());
            responseBody.put("email", userDetails.getEmail());
            responseBody.put("fullName", userDetails.getFullName());
            responseBody.put("role", userDetails.getRole());

            Map<String, Object> userMap = new HashMap<>();
            userMap.put("id", userDetails.getId());
            userMap.put("name", userDetails.getFullName());
            userMap.put("email", userDetails.getEmail());
            
            String cleanRole = userDetails.getRole();
            if (cleanRole != null && cleanRole.startsWith("ROLE_")) {
                cleanRole = cleanRole.substring(5);
            }
            userMap.put("role", cleanRole);
            responseBody.put("user", userMap);

            return ResponseEntity.ok()
                    .header(HttpHeaders.SET_COOKIE, jwtCookie.toString())
                    .body(responseBody);
        } catch (RuntimeException ex) {
            logger.error("Registration failed: {}", ex.getMessage());
            return ResponseEntity.badRequest().body(ex.getMessage());
        }
    }

    // --- Thymeleaf Web Form Endpoints ---

    @PostMapping("/auth/login")
    public String formLogin(@RequestParam String username, @RequestParam String password, HttpServletResponse response) {
        logger.info("=== FORM LOGIN ATTEMPT ===");
        logger.info("Username: {}", username);
        
        try {
            Authentication authentication = authenticationManager.authenticate(
                    new UsernamePasswordAuthenticationToken(username, password));

            SecurityContextHolder.getContext().setAuthentication(authentication);
            UserDetailsImpl userDetails = (UserDetailsImpl) authentication.getPrincipal();

            logger.info("✓ Form login successful for user: {}", userDetails.getUsername());

            ResponseCookie jwtCookie = jwtUtils.generateJwtCookie(userDetails);
            response.addHeader(HttpHeaders.SET_COOKIE, jwtCookie.toString());

            // Redirect based on role
            String role = userDetails.getRole();
            logger.info("User role: {}", role);
            return switch (role) {
                case "ROLE_ADMIN" -> "redirect:/admin/dashboard";
                case "ROLE_VET" -> "redirect:/vet/dashboard";
                default -> "redirect:/farmer/dashboard";
            };
        } catch (RuntimeException ex) {
            logger.error("✗ Form login failed: {}", ex.getMessage());
            return "redirect:/?loginerror=true";
        }
    }

    @PostMapping("/auth/register")
    public String formRegister(@ModelAttribute SignupRequest signupRequest, HttpServletResponse response) {
        logger.info("=== FORM REGISTRATION ATTEMPT ===");
        logger.info("Username: {}", signupRequest.getUsername());
        
        try {
            userService.registerUser(signupRequest);
            logger.info("✓ User registered successfully!");
            
            // Automatically log in the user after successful registration
            Authentication authentication = authenticationManager.authenticate(
                    new UsernamePasswordAuthenticationToken(signupRequest.getUsername(), signupRequest.getPassword()));

            SecurityContextHolder.getContext().setAuthentication(authentication);
            UserDetailsImpl userDetails = (UserDetailsImpl) authentication.getPrincipal();

            ResponseCookie jwtCookie = jwtUtils.generateJwtCookie(userDetails);
            response.addHeader(HttpHeaders.SET_COOKIE, jwtCookie.toString());

            // Redirect based on role
            String role = userDetails.getRole();
            logger.info("✓ Form registration auto-login successful for user: {}", userDetails.getUsername());
            return switch (role) {
                case "ROLE_ADMIN" -> "redirect:/admin/dashboard";
                case "ROLE_VET" -> "redirect:/vet/dashboard";
                default -> "redirect:/farmer/dashboard";
            };
        } catch (RuntimeException ex) {
            logger.error("✗ Form registration failed: {}", ex.getMessage());
            return "redirect:/?registererror=" + ex.getMessage();
        }
    }

    @GetMapping("/auth/logout")
    public String formLogout(HttpServletResponse response) {
        logger.info("=== LOGOUT ===");
        SecurityContextHolder.clearContext();
        ResponseCookie cleanCookie = jwtUtils.getCleanJwtCookie();
        response.addHeader(HttpHeaders.SET_COOKIE, cleanCookie.toString());
        logger.info("✓ User logged out");
        return "redirect:/?loggedout=true";
    }

    @PostMapping("/auth/forgot")
    public String formForgotPassword(@RequestParam String email) {
        logger.info("Forgot password request for email: {}", email);
        // Mock success for demonstration
        return "redirect:/?forgot_success=true";
    }

    @PostMapping("/auth/verify-otp")
    public String formVerifyOtp(@RequestParam String otp) {
        logger.info("OTP verification attempt");
        // Mock success for demonstration
        return "redirect:/?otp_success=true";
    }
}
