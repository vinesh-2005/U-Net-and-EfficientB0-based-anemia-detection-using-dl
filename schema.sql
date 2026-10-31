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

-- Step 3: Screening History Table (AI Prediction & Segmentation Audit Trail)
CREATE TABLE IF NOT EXISTS `screening_history` (
    `id` BIGINT AUTO_INCREMENT PRIMARY KEY,
    `user_id` BIGINT DEFAULT NULL,
    `patient_age` INT DEFAULT NULL,
    `patient_gender` VARCHAR(20) DEFAULT NULL,
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
    CONSTRAINT `fk_screening_user`
        FOREIGN KEY (`user_id`) REFERENCES `users` (`id`)
        ON DELETE SET NULL
        ON UPDATE CASCADE,
    INDEX `idx_screening_user_id` (`user_id`),
    INDEX `idx_screening_created_at` (`created_at`)
) ENGINE=InnoDB DEFAULT CHARSET=utf8mb4 COLLATE=utf8mb4_unicode_ci;
