-- ====================================================================
-- ANEMEIA — Database Schema
-- Deep Learning Framework for Non-Invasive Anemia Screening from Conjunctival Images
-- Target Database: anemeia_db (MySQL 8.0+)
-- ====================================================================

-- Step 1: Create Database safely
CREATE DATABASE IF NOT EXISTS `anemeia_db`
    CHARACTER SET utf8mb4
    COLLATE utf8mb4_unicode_ci;

USE `anemeia_db`;

-- Step 2: Users Table (Clinicians, Researchers, Patients)
CREATE TABLE IF NOT EXISTS `users` (
    `id` BIGINT AUTO_INCREMENT PRIMARY KEY,
    `full_name` VARCHAR(120) NOT NULL,
    `email` VARCHAR(191) NOT NULL UNIQUE,
    `password_hash` VARCHAR(255) NOT NULL,
    `phone_number` VARCHAR(20) DEFAULT NULL,
    `created_at` DATETIME NOT NULL DEFAULT CURRENT_TIMESTAMP,
    INDEX `idx_users_email` (`email`)
) ENGINE=InnoDB DEFAULT CHARSET=utf8mb4 COLLATE=utf8mb4_unicode_ci;

-- Step 3: Patients Table (Tracked Individuals across Longitudinal Screenings)
CREATE TABLE IF NOT EXISTS `patients` (
    `id` VARCHAR(50) PRIMARY KEY,
    `name` VARCHAR(120) NOT NULL,
    `age` INT DEFAULT NULL,
    `gender` VARCHAR(20) DEFAULT NULL,
    `created_at` DATETIME NOT NULL DEFAULT CURRENT_TIMESTAMP,
    `updated_at` DATETIME NOT NULL DEFAULT CURRENT_TIMESTAMP ON UPDATE CURRENT_TIMESTAMP,
    INDEX `idx_patient_name` (`name`)
) ENGINE=InnoDB DEFAULT CHARSET=utf8mb4 COLLATE=utf8mb4_unicode_ci;

-- Step 4: Screening History Table (AI Prediction & Segmentation Audit Trail)
CREATE TABLE IF NOT EXISTS `screening_history` (
    `id` BIGINT AUTO_INCREMENT PRIMARY KEY,
    `patient_id` VARCHAR(50) DEFAULT NULL,
    `patient_name` VARCHAR(120) DEFAULT 'Anonymous Patient',
    `patient_age` INT DEFAULT NULL,
    `patient_gender` VARCHAR(20) DEFAULT NULL,
    `user_id` BIGINT DEFAULT NULL,
    `original_image_path` VARCHAR(255) NOT NULL,
    `segmented_image_path` VARCHAR(255) NOT NULL,
    `extracted_roi_path` VARCHAR(255) NOT NULL,
    `gradcam_image_path` VARCHAR(255) DEFAULT NULL,
    `prediction` VARCHAR(50) NOT NULL,
    `confidence_score` FLOAT NOT NULL,
    `threshold` FLOAT NOT NULL DEFAULT 0.3837,
    `model_name` VARCHAR(100) NOT NULL DEFAULT 'EfficientNet-B0 + U-Net',
    `tissue_quality` VARCHAR(50) DEFAULT 'accepted',
    `created_at` DATETIME NOT NULL DEFAULT CURRENT_TIMESTAMP,
    CONSTRAINT `fk_screening_patient`
        FOREIGN KEY (`patient_id`) REFERENCES `patients` (`id`)
        ON DELETE SET NULL
        ON UPDATE CASCADE,
    INDEX `idx_screening_patient_id` (`patient_id`),
    INDEX `idx_screening_name` (`patient_name`),
    INDEX `idx_screening_created_at` (`created_at`)
) ENGINE=InnoDB DEFAULT CHARSET=utf8mb4 COLLATE=utf8mb4_unicode_ci;
