package com.agriportal.service;

import java.io.IOException;
import java.util.Map;

import org.slf4j.Logger;
import org.slf4j.LoggerFactory;
import org.springframework.beans.factory.annotation.Value;
import org.springframework.core.io.ByteArrayResource;
import org.springframework.http.HttpEntity;
import org.springframework.http.HttpHeaders;
import org.springframework.http.HttpStatus;
import org.springframework.http.MediaType;
import org.springframework.http.ResponseEntity;
import org.springframework.stereotype.Service;
import org.springframework.util.LinkedMultiValueMap;
import org.springframework.util.MultiValueMap;
import org.springframework.web.client.RestClientException;
import org.springframework.web.client.RestTemplate;
import org.springframework.web.multipart.MultipartFile;

@Service
public class AiService {
    private static final Logger logger = LoggerFactory.getLogger(AiService.class);

    @Value("${app.aiServiceUrl}")
    private String aiServiceUrl;

    private final RestTemplate restTemplate = new RestTemplate();

    @SuppressWarnings({"unchecked", "rawtypes"})
    public Map<String, Object> predictPlant(MultipartFile file) {
        try {
            HttpHeaders headers = new HttpHeaders();
            headers.setContentType(MediaType.MULTIPART_FORM_DATA);

            MultiValueMap<String, Object> body = new LinkedMultiValueMap<>();
            body.add("file", new ByteArrayResource(file.getBytes()) {
                @Override
                public String getFilename() {
                    return file.getOriginalFilename();
                }
            });

            HttpEntity<MultiValueMap<String, Object>> requestEntity = new HttpEntity<>(body, headers);
            ResponseEntity<Map> response = restTemplate.postForEntity(aiServiceUrl + "/predictPlant", requestEntity, Map.class);
            
            if (response.getStatusCode() == HttpStatus.OK && response.getBody() != null) {
                return response.getBody();
            }
            throw new RuntimeException("FastAPI predictPlant returned status: " + response.getStatusCode());
        } catch (IOException | RestClientException | IllegalArgumentException ex) {
            logger.error("FastAPI predictPlant call failed. Error: {}", ex.getMessage());
            throw new RuntimeException("AI service connection failed: " + ex.getMessage(), ex);
        }
    }

    @SuppressWarnings({"unchecked", "rawtypes"})
    public Map<String, Object> predictAnimal(MultipartFile file) {
        try {
            HttpHeaders headers = new HttpHeaders();
            headers.setContentType(MediaType.MULTIPART_FORM_DATA);

            MultiValueMap<String, Object> body = new LinkedMultiValueMap<>();
            body.add("file", new ByteArrayResource(file.getBytes()) {
                @Override
                public String getFilename() {
                    return file.getOriginalFilename();
                }
            });

            HttpEntity<MultiValueMap<String, Object>> requestEntity = new HttpEntity<>(body, headers);
            ResponseEntity<Map> response = restTemplate.postForEntity(aiServiceUrl + "/predictAnimal", requestEntity, Map.class);
            
            if (response.getStatusCode() == HttpStatus.OK && response.getBody() != null) {
                return response.getBody();
            }
            throw new RuntimeException("FastAPI predictAnimal returned status: " + response.getStatusCode());
        } catch (IOException | RestClientException | IllegalArgumentException ex) {
            logger.error("FastAPI predictAnimal call failed. Error: {}", ex.getMessage());
            throw new RuntimeException("AI service connection failed: " + ex.getMessage(), ex);
        }
    }

    @SuppressWarnings({"unchecked", "rawtypes"})
    public Map<String, Object> cropRecommendation(Map<String, Object> inputs) {
        try {
            HttpHeaders headers = new HttpHeaders();
            headers.setContentType(MediaType.APPLICATION_JSON);

            HttpEntity<Map<String, Object>> requestEntity = new HttpEntity<>(inputs, headers);
            ResponseEntity<Map> response = restTemplate.postForEntity(aiServiceUrl + "/cropRecommendation", requestEntity, Map.class);

            if (response.getStatusCode() == HttpStatus.OK && response.getBody() != null) {
                return response.getBody();
            }
            throw new RuntimeException("FastAPI cropRecommendation returned status: " + response.getStatusCode());
        } catch (RestClientException | IllegalArgumentException ex) {
            logger.error("FastAPI cropRecommendation failed. Error: {}", ex.getMessage());
            throw new RuntimeException("AI service connection failed: " + ex.getMessage(), ex);
        }
    }

    @SuppressWarnings({"unchecked", "rawtypes"})
    public Map<String, Object> yieldPrediction(Map<String, Object> inputs) {
        try {
            HttpHeaders headers = new HttpHeaders();
            headers.setContentType(MediaType.APPLICATION_JSON);

            HttpEntity<Map<String, Object>> requestEntity = new HttpEntity<>(inputs, headers);
            ResponseEntity<Map> response = restTemplate.postForEntity(aiServiceUrl + "/yieldPrediction", requestEntity, Map.class);

            if (response.getStatusCode() == HttpStatus.OK && response.getBody() != null) {
                return response.getBody();
            }
            throw new RuntimeException("FastAPI yieldPrediction returned status: " + response.getStatusCode());
        } catch (RestClientException | IllegalArgumentException ex) {
            logger.error("FastAPI yieldPrediction failed. Error: {}", ex.getMessage());
            throw new RuntimeException("AI service connection failed: " + ex.getMessage(), ex);
        }
    }

    @SuppressWarnings({"unchecked", "rawtypes"})
    public Map<String, Object> pricePrediction(Map<String, Object> inputs) {
        try {
            HttpHeaders headers = new HttpHeaders();
            headers.setContentType(MediaType.APPLICATION_JSON);

            HttpEntity<Map<String, Object>> requestEntity = new HttpEntity<>(inputs, headers);
            ResponseEntity<Map> response = restTemplate.postForEntity(aiServiceUrl + "/pricePrediction", requestEntity, Map.class);

            if (response.getStatusCode() == HttpStatus.OK && response.getBody() != null) {
                return response.getBody();
            }
            throw new RuntimeException("FastAPI pricePrediction returned status: " + response.getStatusCode());
        } catch (RestClientException | IllegalArgumentException ex) {
            logger.error("FastAPI pricePrediction failed. Error: {}", ex.getMessage());
            throw new RuntimeException("AI service connection failed: " + ex.getMessage(), ex);
        }
    }

    @SuppressWarnings({"unchecked", "rawtypes"})
    public Map<String, Object> fertilizerRecommendation(Map<String, Object> inputs) {
        try {
            HttpHeaders headers = new HttpHeaders();
            headers.setContentType(MediaType.APPLICATION_JSON);

            HttpEntity<Map<String, Object>> requestEntity = new HttpEntity<>(inputs, headers);
            ResponseEntity<Map> response = restTemplate.postForEntity(aiServiceUrl + "/fertilizerRecommendation", requestEntity, Map.class);

            if (response.getStatusCode() == HttpStatus.OK && response.getBody() != null) {
                return response.getBody();
            }
            throw new RuntimeException("FastAPI fertilizerRecommendation returned status: " + response.getStatusCode());
        } catch (RestClientException | IllegalArgumentException ex) {
            logger.error("FastAPI fertilizerRecommendation failed. Error: {}", ex.getMessage());
            throw new RuntimeException("AI service connection failed: " + ex.getMessage(), ex);
        }
    }
}
