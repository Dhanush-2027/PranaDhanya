# AI Powered Smart Agriculture & Livestock Disease Diagnosis Portal

This repository contains the complete production-ready source code for the **AI Powered Smart Agriculture & Livestock Disease Diagnosis Portal**.

## Project Modules
1. **`backend/`**: Spring Boot 3 web application using Spring Security JWT authentication and Thymeleaf frontend layout engine.
2. **`ai_service/`**: Python FastAPI microservice providing CNN classification (for crop and cattle pathologies) and XGBoost/RF prediction engines (for crop selection, yield prediction, and market pricing forecasts).
3. **`database/`**: Contains `init.sql` schema definitions and initial seed data.
4. **`docs/`**: Includes the Postman integration collection (`postman_collection.json`) for endpoint validation.

---

## Quick Start (Docker Compose)
Ensure you have Docker and Docker Compose installed. From the root directory, execute:

```bash
docker compose up --build -d
```

- **Web Portal Access**: [http://localhost:8080](http://localhost:8080)
- **FastAPI Documentation**: [http://localhost:8000/docs](http://localhost:8000/docs)
- **PostgreSQL Database**: Port `5432`

---

## Seed Accounts
Log in using any of the following accounts:
- **Admin**: Username: `admin` | Password: `admin123`
- **Veterinary Officer**: Username: `vet1` | Password: `vet123`
- **Farmer**: Username: `farmer1` | Password: `farmer123`

---

## Manual Execution (Standalone)
For standalone execution:
1. Create a PostgreSQL database `smart_agri_db` and execute the script in `database/init.sql`.
2. Launch the FastAPI service:
   ```bash
   cd ai_service
   pip install -r requirements.txt
   python app.py
   ```
3. Launch the Spring Boot backend:
   ```bash
   cd backend
   mvn clean spring-boot:run
   ```
