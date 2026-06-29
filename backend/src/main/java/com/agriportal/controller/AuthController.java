package com.agriportal.controller;

import com.agriportal.dto.LoginRequest;
import com.agriportal.dto.JwtResponse;
import com.agriportal.dto.SignupRequest;
import com.agriportal.entity.User;
import com.agriportal.security.jwt.JwtUtils;
import com.agriportal.security.services.UserDetailsImpl;
import com.agriportal.service.UserService;
import jakarta.servlet.http.HttpServletResponse;
import jakarta.validation.Valid;
import org.springframework.beans.factory.annotation.Autowired;
import org.springframework.http.HttpHeaders;
import org.springframework.http.ResponseCookie;
import org.springframework.http.ResponseEntity;
import org.springframework.security.authentication.AuthenticationManager;
import org.springframework.security.authentication.UsernamePasswordAuthenticationToken;
import org.springframework.security.core.Authentication;
import org.springframework.security.core.context.SecurityContextHolder;
import org.springframework.stereotype.Controller;
import org.springframework.web.bind.annotation.*;

@Controller
public class AuthController {

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
        Authentication authentication = authenticationManager.authenticate(
                new UsernamePasswordAuthenticationToken(loginRequest.getUsername(), loginRequest.getPassword()));

        SecurityContextHolder.getContext().setAuthentication(authentication);
        UserDetailsImpl userDetails = (UserDetailsImpl) authentication.getPrincipal();

        ResponseCookie jwtCookie = jwtUtils.generateJwtCookie(userDetails);
        
        // Also set cookie in response for potential hybrid clients
        response.addHeader(HttpHeaders.SET_COOKIE, jwtCookie.toString());

        String jwt = jwtUtils.generateTokenFromUsername(userDetails.getUsername());

        return ResponseEntity.ok()
                .header(HttpHeaders.SET_COOKIE, jwtCookie.toString())
                .body(new JwtResponse(jwt,
                        userDetails.getId(),
                        userDetails.getUsername(),
                        userDetails.getEmail(),
                        userDetails.getFullName(),
                        userDetails.getRole()));
    }

    @PostMapping("/api/auth/signup")
    @ResponseBody
    public ResponseEntity<?> registerUser(@Valid @RequestBody SignupRequest signUpRequest) {
        try {
            User user = userService.registerUser(signUpRequest);
            return ResponseEntity.ok("User registered successfully! ID: " + user.getId());
        } catch (Exception e) {
            return ResponseEntity.badRequest().body(e.getMessage());
        }
    }

    // --- Thymeleaf Web Form Endpoints ---

    @PostMapping("/auth/login")
    public String formLogin(@RequestParam String username, @RequestParam String password, HttpServletResponse response) {
        try {
            Authentication authentication = authenticationManager.authenticate(
                    new UsernamePasswordAuthenticationToken(username, password));

            SecurityContextHolder.getContext().setAuthentication(authentication);
            UserDetailsImpl userDetails = (UserDetailsImpl) authentication.getPrincipal();

            ResponseCookie jwtCookie = jwtUtils.generateJwtCookie(userDetails);
            response.addHeader(HttpHeaders.SET_COOKIE, jwtCookie.toString());

            // Redirect based on role
            String role = userDetails.getRole();
            if ("ROLE_ADMIN".equals(role)) {
                return "redirect:/admin/dashboard";
            } else if ("ROLE_VET".equals(role)) {
                return "redirect:/vet/dashboard";
            } else {
                return "redirect:/farmer/dashboard";
            }
        } catch (Exception e) {
            return "redirect:/?loginerror=true";
        }
    }

    @PostMapping("/auth/register")
    public String formRegister(@ModelAttribute SignupRequest signupRequest) {
        try {
            userService.registerUser(signupRequest);
            return "redirect:/?registersuccess=true";
        } catch (Exception e) {
            return "redirect:/?registererror=" + e.getMessage();
        }
    }

    @GetMapping("/auth/logout")
    public String formLogout(HttpServletResponse response) {
        SecurityContextHolder.clearContext();
        ResponseCookie cleanCookie = jwtUtils.getCleanJwtCookie();
        response.addHeader(HttpHeaders.SET_COOKIE, cleanCookie.toString());
        return "redirect:/?loggedout=true";
    }

    @PostMapping("/auth/forgot")
    public String formForgotPassword(@RequestParam String email) {
        // Mock success for demonstration
        return "redirect:/?forgot_success=true";
    }

    @PostMapping("/auth/verify-otp")
    public String formVerifyOtp(@RequestParam String otp) {
        // Mock success for demonstration
        return "redirect:/?otp_success=true";
    }
}
