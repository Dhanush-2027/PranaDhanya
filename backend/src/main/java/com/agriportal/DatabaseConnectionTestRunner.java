package com.agriportal;

import org.slf4j.Logger;
import org.slf4j.LoggerFactory;
import org.springframework.beans.factory.annotation.Autowired;
import org.springframework.boot.CommandLineRunner;
import org.springframework.stereotype.Component;

import javax.sql.DataSource;
import java.sql.Connection;
import java.sql.DatabaseMetaData;
import java.sql.ResultSet;
import java.sql.Statement;

@Component
public class DatabaseConnectionTestRunner implements CommandLineRunner {

    private static final Logger logger = LoggerFactory.getLogger(DatabaseConnectionTestRunner.class);

    @Autowired
    private DataSource dataSource;

    @Override
    public void run(String... args) throws Exception {
        logger.info("=================================================================");
        logger.info("Auditing Database Connection Layer on Application Startup...");
        logger.info("=================================================================");

        try (Connection connection = dataSource.getConnection()) {
            if (connection != null && !connection.isClosed()) {
                DatabaseMetaData metaData = connection.getMetaData();
                logger.info("✓ DATABASE CONNECTION SUCCESSFUL!");
                logger.info("Database Product Name: {}", metaData.getDatabaseProductName());
                logger.info("Database Product Version: {}", metaData.getDatabaseProductVersion());
                logger.info("Driver Name: {}", metaData.getDriverName());
                logger.info("Driver Version: {}", metaData.getDriverVersion());
                logger.info("JDBC URL: {}", metaData.getURL());
                logger.info("Database User: {}", metaData.getUserName());

                // Execute a test query to confirm query execution works
                try (Statement statement = connection.createStatement();
                     ResultSet resultSet = statement.executeQuery("SELECT 1")) {
                    if (resultSet.next()) {
                        logger.info("✓ Database query execution (SELECT 1) verified successfully!");
                    }
                }
            } else {
                logger.error("✗ Database connection is null or closed.");
            }
        } catch (Exception e) {
            logger.error("✗ DATABASE CONNECTION FAILED!");
            logger.error("Error Details: {}", e.getMessage(), e);
            throw e; // Fail application startup if database connection fails
        }
        logger.info("=================================================================");
    }
}
