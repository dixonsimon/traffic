-- MySQL schema for the Smart Traffic Congestion Prediction System
-- Usage:  mysql -u root -p < schema.sql
CREATE DATABASE IF NOT EXISTS traffic_db CHARACTER SET utf8mb4;
USE traffic_db;

CREATE TABLE IF NOT EXISTS roads (
    road_id     INT PRIMARY KEY,
    road_name   VARCHAR(100) NOT NULL,
    area        VARCHAR(60)  NOT NULL,
    road_type   VARCHAR(30)  NOT NULL,
    lanes       INT          NOT NULL,
    speed_limit INT          NOT NULL,
    capacity    INT          NOT NULL,   -- vehicles / hour
    INDEX idx_road_name (road_name)
);

CREATE TABLE IF NOT EXISTS traffic_records (
    record_id      BIGINT AUTO_INCREMENT PRIMARY KEY,
    road_id        INT          NOT NULL,
    ts             DATETIME     NOT NULL,
    hour           TINYINT      NOT NULL,
    day_of_week    TINYINT      NOT NULL,   -- 0 = Monday
    is_weekend     TINYINT      NOT NULL,
    weather        VARCHAR(20)  NOT NULL,
    temperature    FLOAT        NOT NULL,
    visibility_km  FLOAT        NOT NULL,
    road_condition VARCHAR(20)  NOT NULL,
    traffic_volume INT          NOT NULL,
    avg_speed      FLOAT        NOT NULL,
    congestion     VARCHAR(10)  NOT NULL,   -- Low / Medium / High
    FOREIGN KEY (road_id) REFERENCES roads(road_id),
    INDEX idx_road_time (road_id, ts),
    INDEX idx_hour_day (hour, day_of_week)
);
