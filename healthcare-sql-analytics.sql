-- Healthcare Research Data Analytics Queries
-- Author: Justin Christopher Weaver
-- Description: Advanced SQL queries for Epic-style healthcare analytics with security considerations

-- =====================================================
-- SCHEMA DESIGN FOR EPIC-STYLE DATA WAREHOUSE
-- =====================================================

-- Patient Dimension (PHI/PII Protected)
CREATE TABLE dim_patient (
    patient_id BIGINT PRIMARY KEY,
    mrn_hash VARCHAR(64) NOT NULL UNIQUE, -- SHA-256 hashed MRN
    birth_year INT NOT NULL, -- Only year for privacy
    gender VARCHAR(10),
    race VARCHAR(50),
    ethnicity VARCHAR(50),
    zip_code_prefix VARCHAR(3), -- First 3 digits only
    insurance_category VARCHAR(50),
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP ON UPDATE CURRENT_TIMESTAMP,
    INDEX idx_patient_demographics (gender, race, ethnicity),
    INDEX idx_patient_mrn_hash (mrn_hash)
) ENGINE=InnoDB DEFAULT CHARSET=utf8mb4 COLLATE=utf8mb4_unicode_ci;

-- Provider Dimension
CREATE TABLE dim_provider (
    provider_id BIGINT PRIMARY KEY,
    provider_type VARCHAR(50),
    specialty VARCHAR(100),
    department_id INT,
    years_experience INT,
    is_active BOOLEAN DEFAULT TRUE,
    INDEX idx_provider_specialty (specialty),
    INDEX idx_provider_dept (department_id)
) ENGINE=InnoDB;

-- Time Dimension
CREATE TABLE dim_time (
    time_id INT PRIMARY KEY,
    full_date DATE NOT NULL UNIQUE,
    year INT,
    quarter INT,
    month INT,
    week INT,
    day_of_week INT,
    is_weekend BOOLEAN,
    is_holiday BOOLEAN,
    fiscal_year INT,
    fiscal_quarter INT,
    INDEX idx_time_date (full_date),
    INDEX idx_time_year_month (year, month)
) ENGINE=InnoDB;

-- Diagnosis Dimension (ICD-10)
CREATE TABLE dim_diagnosis (
    diagnosis_id BIGINT PRIMARY KEY,
    icd10_code VARCHAR(10) NOT NULL UNIQUE,
    diagnosis_description TEXT,
    category VARCHAR(100),
    is_chronic BOOLEAN,
    severity_score DECIMAL(3,2),
    INDEX idx_diagnosis_code (icd10_code),
    INDEX idx_diagnosis_category (category)
) ENGINE=InnoDB;

-- Encounter Fact Table
CREATE TABLE fact_encounters (
    encounter_id BIGINT PRIMARY KEY,
    patient_id BIGINT NOT NULL,
    provider_id BIGINT,
    admission_time_id INT,
    discharge_time_id INT,
    encounter_type VARCHAR(50),
    primary_diagnosis_id BIGINT,
    admission_source VARCHAR(50),
    discharge_disposition VARCHAR(50),
    length_of_stay INT,
    icu_days INT DEFAULT 0,
    total_charges DECIMAL(12,2),
    readmission_flag BOOLEAN DEFAULT FALSE,
    mortality_flag BOOLEAN DEFAULT FALSE,
    encrypted_notes MEDIUMTEXT, -- AES-256 encrypted
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    FOREIGN KEY (patient_id) REFERENCES dim_patient(patient_id),
    FOREIGN KEY (provider_id) REFERENCES dim_provider(provider_id),
    FOREIGN KEY (admission_time_id) REFERENCES dim_time(time_id),
    FOREIGN KEY (discharge_time_id) REFERENCES dim_time(time_id),
    FOREIGN KEY (primary_diagnosis_id) REFERENCES dim_diagnosis(diagnosis_id),
    INDEX idx_encounter_patient (patient_id),
    INDEX idx_encounter_dates (admission_time_id, discharge_time_id),
    INDEX idx_encounter_readmit (readmission_flag, discharge_time_id)
) ENGINE=InnoDB;

-- Lab Results Fact Table
CREATE TABLE fact_lab_results (
    result_id BIGINT PRIMARY KEY AUTO_INCREMENT,
    encounter_id BIGINT NOT NULL,
    patient_id BIGINT NOT NULL,
    lab_time_id INT,
    lab_test_code VARCHAR(20),
    lab_test_name VARCHAR(100),
    result_value DECIMAL(10,3),
    result_unit VARCHAR(20),
    reference_low DECIMAL(10,3),
    reference_high DECIMAL(10,3),
    abnormal_flag CHAR(2), -- HH, H, N, L, LL
    critical_flag BOOLEAN DEFAULT FALSE,
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    FOREIGN KEY (encounter_id) REFERENCES fact_encounters(encounter_id),
    FOREIGN KEY (patient_id) REFERENCES dim_patient(patient_id),
    FOREIGN KEY (lab_time_id) REFERENCES dim_time(time_id),
    INDEX idx_lab_encounter (encounter_id),
    INDEX idx_lab_patient_test (patient_id, lab_test_code),
    INDEX idx_lab_critical (critical_flag, lab_time_id)
) ENGINE=InnoDB;

-- Medication Orders Fact Table
CREATE TABLE fact_medication_orders (
    order_id BIGINT PRIMARY KEY,
    encounter_id BIGINT NOT NULL,
    patient_id BIGINT NOT NULL,
    provider_id BIGINT,
    medication_id BIGINT,
    order_time_id INT,
    start_time_id INT,
    end_time_id INT,
    dose_amount DECIMAL(10,3),
    dose_unit VARCHAR(20),
    frequency VARCHAR(50),
    route VARCHAR(50),
    order_status VARCHAR(20),
    is_stat BOOLEAN DEFAULT FALSE,
    is_prn BOOLEAN DEFAULT FALSE,
    FOREIGN KEY (encounter_id) REFERENCES fact_encounters(encounter_id),
    FOREIGN KEY (patient_id) REFERENCES dim_patient(patient_id),
    FOREIGN KEY (provider_id) REFERENCES dim_provider(provider_id),
    INDEX idx_med_encounter (encounter_id),
    INDEX idx_med_patient_time (patient_id, order_time_id)
) ENGINE=InnoDB;

-- =====================================================
-- ADVANCED ANALYTICS QUERIES
-- =====================================================

-- 1. READMISSION RISK ANALYSIS WITH MULTIPLE FACTORS
WITH ReadmissionAnalysis AS (
    SELECT 
        p.patient_id,
        p.gender,
        p.race,
        CASE 
            WHEN 2025 - p.birth_year < 65 THEN 'Under 65'
            WHEN 2025 - p.birth_year BETWEEN 65 AND 79 THEN '65-79'
            ELSE '80+'
        END AS age_group,
        d.category AS diagnosis_category,
        d.is_chronic,
        d.severity_score,
        e.length_of_stay,
        e.icu_days,
        e.total_charges,
        e.discharge_disposition,
        -- Calculate if readmitted within 30 days
        EXISTS (
            SELECT 1 
            FROM fact_encounters e2
            WHERE e2.patient_id = e.patient_id
            AND e2.encounter_id != e.encounter_id
            AND e2.admission_time_id BETWEEN e.discharge_time_id 
                AND e.discharge_time_id + 30
        ) AS readmitted_30day,
        -- Count abnormal labs
        (
            SELECT COUNT(DISTINCT lab_test_code)
            FROM fact_lab_results lr
            WHERE lr.encounter_id = e.encounter_id
            AND lr.abnormal_flag IN ('H', 'HH', 'L', 'LL')
        ) AS abnormal_lab_count,
        -- Count medications
        (
            SELECT COUNT(DISTINCT medication_id)
            FROM fact_medication_orders mo
            WHERE mo.encounter_id = e.encounter_id
        ) AS medication_count
    FROM fact_encounters e
    JOIN dim_patient p ON e.patient_id = p.patient_id
    JOIN dim_diagnosis d ON e.primary_diagnosis_id = d.diagnosis_id
    JOIN dim_time t ON e.discharge_time_id = t.time_id
    WHERE t.year = 2024
    AND e.encounter_type = 'Inpatient'
    AND e.mortality_flag = FALSE
)
SELECT 
    diagnosis_category,
    age_group,
    COUNT(*) AS total_discharges,
    SUM(readmitted_30day) AS readmissions,
    ROUND(SUM(readmitted_30day) * 100.0 / COUNT(*), 2) AS readmission_rate,
    ROUND(AVG(length_of_stay), 1) AS avg_los,
    ROUND(AVG(icu_days), 1) AS avg_icu_days,
    ROUND(AVG(abnormal_lab_count), 1) AS avg_abnormal_labs,
    ROUND(AVG(medication_count), 1) AS avg_medications,
    ROUND(AVG(total_charges), 0) AS avg_charges,
    ROUND(STDDEV(length_of_stay), 2) AS los_std_dev
FROM ReadmissionAnalysis
GROUP BY diagnosis_category, age_group
HAVING COUNT(*) >= 30  -- Statistical significance
ORDER BY readmission_rate DESC;

-- 2. CLINICAL QUALITY METRICS - SEPSIS BUNDLE COMPLIANCE
WITH SepsisPatients AS (
    SELECT DISTINCT
        e.encounter_id,
        e.patient_id,
        e.admission_time_id,
        MIN(lr.lab_time_id) AS first_abnormal_lab_time
    FROM fact_encounters e
    JOIN dim_diagnosis d ON e.primary_diagnosis_id = d.diagnosis_id
    JOIN fact_lab_results lr ON e.encounter_id = lr.encounter_id
    WHERE d.icd10_code LIKE 'A41%'  -- Sepsis codes
    OR (
        -- SIRS criteria
        lr.lab_test_code IN ('WBC', 'TEMP', 'HR', 'RR')
        AND lr.critical_flag = TRUE
    )
    GROUP BY e.encounter_id, e.patient_id, e.admission_time_id
),
BundleCompliance AS (
    SELECT 
        sp.encounter_id,
        sp.patient_id,
        -- Check lactate measurement within 3 hours
        EXISTS (
            SELECT 1 
            FROM fact_lab_results lr
            WHERE lr.encounter_id = sp.encounter_id
            AND lr.lab_test_code = 'LACT'
            AND lr.lab_time_id <= sp.first_abnormal_lab_time + 3
        ) AS lactate_measured,
        -- Check blood cultures before antibiotics
        EXISTS (
            SELECT 1 
            FROM fact_lab_results lr
            WHERE lr.encounter_id = sp.encounter_id
            AND lr.lab_test_code LIKE 'BLOOD_CX%'
            AND lr.lab_time_id < (
                SELECT MIN(mo.start_time_id)
                FROM fact_medication_orders mo
                WHERE mo.encounter_id = sp.encounter_id
                AND mo.medication_id IN (
                    SELECT medication_id 
                    FROM dim_medication 
                    WHERE medication_class = 'Antibiotic'
                )
            )
        ) AS blood_culture_before_abx,
        -- Check antibiotic administration within 1 hour
        EXISTS (
            SELECT 1
            FROM fact_medication_orders mo
            JOIN dim_medication m ON mo.medication_id = m.medication_id
            WHERE mo.encounter_id = sp.encounter_id
            AND m.medication_class = 'Antibiotic'
            AND mo.start_time_id <= sp.first_abnormal_lab_time + 1
        ) AS antibiotics_within_hour,
        -- Check fluid resuscitation
        EXISTS (
            SELECT 1
            FROM fact_medication_orders mo
            WHERE mo.encounter_id = sp.encounter_id
            AND mo.medication_id IN (
                SELECT medication_id 
                FROM dim_medication 
                WHERE medication_name LIKE '%saline%'
                OR medication_name LIKE '%lactated ringers%'
            )
            AND mo.dose_amount >= 30  -- 30mL/kg
        ) AS fluid_resuscitation
    FROM SepsisPatients sp
)
SELECT 
    COUNT(*) AS total_sepsis_patients,
    SUM(lactate_measured) AS lactate_compliant,
    SUM(blood_culture_before_abx) AS culture_compliant,
    SUM(antibiotics_within_hour) AS antibiotic_compliant,
    SUM(fluid_resuscitation) AS fluid_compliant,
    SUM(
        CASE WHEN lactate_measured 
            AND blood_culture_before_abx 
            AND antibiotics_within_hour 
            AND fluid_resuscitation 
        THEN 1 ELSE 0 END
    ) AS fully_compliant,
    ROUND(
        SUM(
            CASE WHEN lactate_measured 
                AND blood_culture_before_abx 
                AND antibiotics_within_hour 
                AND fluid_resuscitation 
            THEN 1 ELSE 0 END
        ) * 100.0 / COUNT(*), 2
    ) AS bundle_compliance_rate
FROM BundleCompliance;

-- 3. PROVIDER PERFORMANCE ANALYTICS WITH RISK ADJUSTMENT
WITH ProviderMetrics AS (
    SELECT 
        p.provider_id,
        p.specialty,
        COUNT(DISTINCT e.encounter_id) AS total_encounters,
        COUNT(DISTINCT e.patient_id) AS unique_patients,
        AVG(e.length_of_stay) AS avg_los,
        SUM(e.readmission_flag) AS readmissions,
        SUM(e.mortality_flag) AS mortalities,
        AVG(e.total_charges) AS avg_charges,
        -- Calculate case mix index
        AVG(d.severity_score) AS avg_case_severity,
        -- Patient satisfaction proxy (discharge disposition)
        SUM(CASE WHEN e.discharge_disposition = 'Home' THEN 1 ELSE 0 END) * 100.0 
            / COUNT(*) AS home_discharge_rate
    FROM fact_encounters e
    JOIN dim_provider p ON e.provider_id = p.provider_id
    JOIN dim_diagnosis d ON e.primary_diagnosis_id = d.diagnosis_id
    JOIN dim_time t ON e.admission_time_id = t.time_id
    WHERE t.year = 2024
    AND p.is_active = TRUE
    GROUP BY p.provider_id, p.specialty
    HAVING COUNT(DISTINCT e.encounter_id) >= 20  -- Minimum volume
),
SpecialtyBenchmarks AS (
    SELECT 
        specialty,
        AVG(avg_los) AS specialty_avg_los,
        AVG(readmissions * 100.0 / total_encounters) AS specialty_readmit_rate,
        AVG(avg_case_severity) AS specialty_avg_severity
    FROM ProviderMetrics
    GROUP BY specialty
)
SELECT 
    pm.provider_id,
    pm.specialty,
    pm.total_encounters,
    pm.unique_patients,
    ROUND(pm.avg_los, 1) AS provider_avg_los,
    ROUND(sb.specialty_avg_los, 1) AS specialty_avg_los,
    ROUND(pm.avg_los - sb.specialty_avg_los, 1) AS los_variance,
    ROUND(pm.readmissions * 100.0 / pm.total_encounters, 2) AS provider_readmit_rate,
    ROUND(sb.specialty_readmit_rate, 2) AS specialty_readmit_rate,
    ROUND(pm.avg_case_severity, 2) AS case_mix_index,
    ROUND(pm.home_discharge_rate, 1) AS home_discharge_rate,
    ROUND(pm.avg_charges, 0) AS avg_charges,
    -- Risk-adjusted performance score
    ROUND(
        100 - (
            (pm.avg_los / sb.specialty_avg_los * 25) +
            (pm.readmissions * 100.0 / pm.total_encounters / sb.specialty_readmit_rate * 25) +
            ((100 - pm.home_discharge_rate) / 50 * 25) +
            (CASE WHEN pm.mortalities > 0 THEN 25 ELSE 0 END)
        ) * (sb.specialty_avg_severity / pm.avg_case_severity),  -- Risk adjustment
        1
    ) AS performance_score
FROM ProviderMetrics pm
JOIN SpecialtyBenchmarks sb ON pm.specialty = sb.specialty
ORDER BY performance_score DESC;

-- 4. POPULATION HEALTH - CHRONIC DISEASE MANAGEMENT
WITH ChronicDiseasePopulation AS (
    SELECT 
        p.patient_id,
        p.gender,
        p.race,
        2025 - p.birth_year AS age,
        COUNT(DISTINCT 
            CASE WHEN d.icd10_code LIKE 'E11%' THEN 1 END
        ) AS has_diabetes,
        COUNT(DISTINCT 
            CASE WHEN d.icd10_code LIKE 'I10%' THEN 1 END
        ) AS has_hypertension,
        COUNT(DISTINCT 
            CASE WHEN d.icd10_code LIKE 'J44%' THEN 1 END
        ) AS has_copd,
        COUNT(DISTINCT 
            CASE WHEN d.icd10_code LIKE 'N18%' THEN 1 END
        ) AS has_ckd,
        COUNT(DISTINCT e.encounter_id) AS total_encounters_year,
        SUM(e.total_charges) AS total_charges_year
    FROM dim_patient p
    JOIN fact_encounters e ON p.patient_id = e.patient_id
    JOIN dim_diagnosis d ON e.primary_diagnosis_id = d.diagnosis_id
    JOIN dim_time t ON e.admission_time_id = t.time_id
    WHERE t.year = 2024
    GROUP BY p.patient_id, p.gender, p.race, p.birth_year
),
DiseaseManagement AS (
    SELECT 
        patient_id,
        -- Check A1C testing for diabetics
        CASE 
            WHEN has_diabetes > 0 THEN
                (SELECT COUNT(DISTINCT DATE_FORMAT(t.full_date, '%Y-%m'))
                 FROM fact_lab_results lr
                 JOIN dim_time t ON lr.lab_time_id = t.time_id
                 WHERE lr.patient_id = cdp.patient_id
                 AND lr.lab_test_code = 'HBA1C'
                 AND t.year = 2024)
            ELSE NULL
        END AS a1c_tests_count,
        -- Check BP monitoring for hypertension
        CASE 
            WHEN has_hypertension > 0 THEN
                (SELECT COUNT(DISTINCT encounter_id)
                 FROM fact_encounters e2
                 WHERE e2.patient_id = cdp.patient_id
                 AND e2.encounter_type = 'Outpatient')
            ELSE NULL
        END AS bp_check_visits,
        has_diabetes + has_hypertension + has_copd + has_ckd AS chronic_conditions_count,
        total_encounters_year,
        total_charges_year
    FROM ChronicDiseasePopulation cdp
)
SELECT 
    CASE 
        WHEN chronic_conditions_count = 0 THEN 'No Chronic Conditions'
        WHEN chronic_conditions_count = 1 THEN '1 Chronic Condition'
        WHEN chronic_conditions_count = 2 THEN '2 Chronic Conditions'
        ELSE '3+ Chronic Conditions'
    END AS condition_category,
    COUNT(*) AS patient_count,
    ROUND(AVG(total_encounters_year), 1) AS avg_encounters_per_year,
    ROUND(AVG(total_charges_year), 0) AS avg_annual_charges,
    -- Diabetes management metrics
    COUNT(CASE WHEN a1c_tests_count >= 2 THEN 1 END) AS diabetics_with_adequate_a1c,
    COUNT(CASE WHEN a1c_tests_count IS NOT NULL THEN 1 END) AS total_diabetics,
    -- Hypertension management
    COUNT(CASE WHEN bp_check_visits >= 4 THEN 1 END) AS htn_with_adequate_monitoring,
    COUNT(CASE WHEN bp_check_visits IS NOT NULL THEN 1 END) AS total_hypertensive
FROM DiseaseManagement
GROUP BY condition_category
ORDER BY chronic_conditions_count;

-- 5. REAL-TIME CLINICAL SURVEILLANCE (Potential HAI Detection)
WITH CurrentInpatients AS (
    SELECT 
        e.encounter_id,
        e.patient_id,
        e.admission_time_id,
        DATEDIFF(CURRENT_DATE, t.full_date) AS days_admitted,
        e.icu_days > 0 AS in_icu
    FROM fact_encounters e
    JOIN dim_time t ON e.admission_time_id = t.time_id
    WHERE e.discharge_time_id IS NULL  -- Still admitted
),
InfectionRiskScoring AS (
    SELECT 
        ci.encounter_id,
        ci.patient_id,
        ci.days_admitted,
        ci.in_icu,
        -- Central line days
        (SELECT COUNT(DISTINCT mo.order_time_id)
         FROM fact_medication_orders mo
         WHERE mo.encounter_id = ci.encounter_id
         AND mo.route = 'Central Line') AS central_line_days,
        -- Recent positive cultures
        (SELECT COUNT(*)
         FROM fact_lab_results lr
         WHERE lr.encounter_id = ci.encounter_id
         AND lr.lab_test_code LIKE '%CULTURE%'
         AND lr.result_value > 0
         AND lr.lab_time_id >= ci.admission_time_id + 2) AS positive_cultures,
        -- Fever episodes
        (SELECT COUNT(*)
         FROM fact_lab_results lr
         WHERE lr.encounter_id = ci.encounter_id
         AND lr.lab_test_code = 'TEMP'
         AND lr.result_value > 38.3) AS fever_episodes,
        -- WBC trend
        (SELECT 
            CASE 
                WHEN MAX(lr.result_value) - MIN(lr.result_value) > 5 THEN 1
                ELSE 0
            END
         FROM fact_lab_results lr
         WHERE lr.encounter_id = ci.encounter_id
         AND lr.lab_test_code = 'WBC'
         AND lr.lab_time_id >= ci.admission_time_id) AS wbc_trending_up
    FROM CurrentInpatients ci
),
RiskStratification AS (
    SELECT 
        *,
        -- Calculate HAI risk score
        (CASE WHEN days_admitted > 7 THEN 2 ELSE 0 END) +
        (CASE WHEN in_icu THEN 3 ELSE 0 END) +
        (CASE WHEN central_line_days > 0 THEN 3 ELSE 0 END) +
        (positive_cultures * 4) +
        (fever_episodes * 2) +
        (wbc_trending_up * 2) AS hai_risk_score
    FROM InfectionRiskScoring
)
SELECT 
    encounter_id,
    patient_id,
    days_admitted,
    CASE 
        WHEN hai_risk_score >= 10 THEN 'HIGH'
        WHEN hai_risk_score >= 5 THEN 'MODERATE'
        ELSE 'LOW'
    END AS risk_level,
    hai_risk_score,
    in_icu,
    central_line_days,
    positive_cultures,
    fever_episodes
FROM RiskStratification
WHERE hai_risk_score >= 5  -- Moderate or high risk
ORDER BY hai_risk_score DESC;

-- 6. FINANCIAL ANALYSIS - DRG PROFITABILITY WITH OUTLIER DETECTION
WITH DRGAnalysis AS (
    SELECT 
        d.category AS drg_category,
        e.encounter_id,
        e.length_of_stay,
        e.total_charges,
        e.icu_days,
        -- Calculate expected LOS based on diagnosis
        AVG(e.length_of_stay) OVER (PARTITION BY d.category) AS expected_los,
        STDDEV(e.length_of_stay) OVER (PARTITION BY d.category) AS los_stddev,
        -- Calculate expected charges
        AVG(e.total_charges) OVER (PARTITION BY d.category) AS expected_charges,
        STDDEV(e.total_charges) OVER (PARTITION BY d.category) AS charges_stddev
    FROM fact_encounters e
    JOIN dim_diagnosis d ON e.primary_diagnosis_id = d.diagnosis_id
    JOIN dim_time t ON e.discharge_time_id = t.time_id
    WHERE t.year = 2024
    AND e.encounter_type = 'Inpatient'
),
OutlierDetection AS (
    SELECT 
        *,
        -- Identify LOS outliers (beyond 2 standard deviations)
        CASE 
            WHEN length_of_stay > expected_los + (2 * los_stddev) THEN 'High LOS Outlier'
            WHEN length_of_stay < expected_los - (2 * los_stddev) THEN 'Low LOS Outlier'
            ELSE 'Normal'
        END AS los_outlier_status,
        -- Identify charge outliers
        CASE 
            WHEN total_charges > expected_charges + (2 * charges_stddev) THEN 'High Cost Outlier'
            WHEN total_charges < expected_charges - (2 * charges_stddev) THEN 'Low Cost Outlier'
            ELSE 'Normal'
        END AS cost_outlier_status
    FROM DRGAnalysis
)
SELECT 
    drg_category,
    COUNT(*) AS total_cases,
    ROUND(AVG(length_of_stay), 1) AS avg_los,
    ROUND(AVG(total_charges), 0) AS avg_charges,
    ROUND(SUM(total_charges), 0) AS total_revenue,
    -- Outlier analysis
    SUM(CASE WHEN los_outlier_status = 'High LOS Outlier' THEN 1 ELSE 0 END) AS high_los_outliers,
    SUM(CASE WHEN cost_outlier_status = 'High Cost Outlier' THEN 1 ELSE 0 END) AS high_cost_outliers,
    -- Financial impact of outliers
    SUM(CASE 
        WHEN los_outlier_status = 'High LOS Outlier' 
        THEN total_charges - expected_charges 
        ELSE 0 
    END) AS excess_charges_from_outliers,
    -- Contribution margin estimate (assuming 40% variable cost)
    ROUND(SUM(total_charges) * 0.6, 0) AS estimated_contribution_margin,
    -- ICU utilization
    ROUND(AVG(CASE WHEN icu_days > 0 THEN icu_days END), 1) AS avg_icu_days_when_used,
    SUM(CASE WHEN icu_days > 0 THEN 1 ELSE 0 END) * 100.0 / COUNT(*) AS icu_utilization_rate
FROM OutlierDetection
GROUP BY drg_category
HAVING COUNT(*) >= 10
ORDER BY total_revenue DESC;

-- 7. EPIC CLARITY-STYLE RESEARCH COHORT IDENTIFICATION
-- Complex query to identify diabetes patients with poor control for clinical trial
WITH DiabetesCohort AS (
    SELECT DISTINCT
        p.patient_id,
        p.mrn_hash,
        2025 - p.birth_year AS age,
        p.gender,
        p.race,
        -- Get most recent A1C
        (SELECT lr.result_value
         FROM fact_lab_results lr
         JOIN dim_time t ON lr.lab_time_id = t.time_id
         WHERE lr.patient_id = p.patient_id
         AND lr.lab_test_code = 'HBA1C'
         ORDER BY t.full_date DESC
         LIMIT 1) AS latest_a1c,
        -- Count diabetes encounters
        (SELECT COUNT(DISTINCT e.encounter_id)
         FROM fact_encounters e
         JOIN dim_diagnosis d ON e.primary_diagnosis_id = d.diagnosis_id
         WHERE e.patient_id = p.patient_id
         AND d.icd10_code LIKE 'E11%') AS diabetes_encounters,
        -- Check for complications
        EXISTS (
            SELECT 1
            FROM fact_encounters e
            JOIN dim_diagnosis d ON e.primary_diagnosis_id = d.diagnosis_id
            WHERE e.patient_id = p.patient_id
            AND (d.icd10_code LIKE 'E11.2%'  -- Diabetic nephropathy
                OR d.icd10_code LIKE 'E11.3%'  -- Diabetic retinopathy
                OR d.icd10_code LIKE 'E11.4%'  -- Diabetic neuropathy
                OR d.icd10_code LIKE 'E11.5%'  -- Diabetic circulatory
            )
        ) AS has_complications
    FROM dim_patient p
    WHERE EXISTS (
        SELECT 1
        FROM fact_encounters e
        JOIN dim_diagnosis d ON e.primary_diagnosis_id = d.diagnosis_id
        WHERE e.patient_id = p.patient_id
        AND d.icd10_code LIKE 'E11%'
    )
),
EligibilityScreening AS (
    SELECT 
        *,
        -- Inclusion criteria
        CASE 
            WHEN age BETWEEN 40 AND 75
                AND latest_a1c > 8.0
                AND diabetes_encounters >= 2
            THEN TRUE
            ELSE FALSE
        END AS meets_inclusion,
        -- Exclusion criteria
        CASE 
            WHEN EXISTS (
                SELECT 1
                FROM fact_encounters e
                JOIN dim_diagnosis d ON e.primary_diagnosis_id = d.diagnosis_id
                WHERE e.patient_id = patient_id
                AND (d.icd10_code LIKE 'N18.5%'  -- CKD Stage 5
                    OR d.icd10_code LIKE 'Z94%'   -- Organ transplant
                    OR d.icd10_code LIKE 'C%'     -- Active cancer
                )
            ) THEN TRUE
            ELSE FALSE
        END AS has_exclusion
    FROM DiabetesCohort
)
SELECT 
    patient_id,
    mrn_hash,
    age,
    gender,
    race,
    latest_a1c,
    diabetes_encounters,
    has_complications,
    CASE 
        WHEN meets_inclusion AND NOT has_exclusion THEN 'ELIGIBLE'
        WHEN meets_inclusion AND has_exclusion THEN 'EXCLUDED'
        ELSE 'NOT ELIGIBLE'
    END AS trial_status
FROM EligibilityScreening
WHERE meets_inclusion = TRUE
ORDER BY latest_a1c DESC, age;

-- 8. OPERATIONAL EFFICIENCY - ED THROUGHPUT ANALYSIS
WITH EDVisits AS (
    SELECT 
        e.encounter_id,
        e.patient_id,
        t_arr.full_date AS arrival_date,
        t_arr.hour AS arrival_hour,
        t_dep.full_date AS departure_date,
        t_dep.hour AS departure_hour,
        TIMESTAMPDIFF(MINUTE, 
            CONCAT(t_arr.full_date, ' ', t_arr.hour, ':00:00'),
            CONCAT(t_dep.full_date, ' ', t_dep.hour, ':00:00')
        ) AS total_ed_minutes,
        e.discharge_disposition,
        d.severity_score,
        -- Identify if admitted
        CASE 
            WHEN e.discharge_disposition IN ('Admit', 'Transfer') THEN 1 
            ELSE 0 
        END AS admitted_flag
    FROM fact_encounters e
    JOIN dim_time t_arr ON e.admission_time_id = t_arr.time_id
    JOIN dim_time t_dep ON e.discharge_time_id = t_dep.time_id
    JOIN dim_diagnosis d ON e.primary_diagnosis_id = d.diagnosis_id
    WHERE e.encounter_type = 'Emergency'
),
HourlyMetrics AS (
    SELECT 
        arrival_hour,
        COUNT(*) AS volume,
        AVG(total_ed_minutes) AS avg_minutes,
        PERCENTILE_CONT(0.5) WITHIN GROUP (ORDER BY total_ed_minutes) AS median_minutes,
        PERCENTILE_CONT(0.9) WITHIN GROUP (ORDER BY total_ed_minutes) AS p90_minutes,
        SUM(admitted_flag) * 100.0 / COUNT(*) AS admission_rate,
        AVG(severity_score) AS avg_acuity
    FROM EDVisits
    GROUP BY arrival_hour
)
SELECT 
    arrival_hour,
    volume,
    ROUND(avg_minutes / 60, 1) AS avg_hours,
    ROUND(median_minutes / 60, 1) AS median_hours,
    ROUND(p90_minutes / 60, 1) AS p90_hours,
    ROUND(admission_rate, 1) AS admission_rate_pct,
    ROUND(avg_acuity, 2) AS avg_acuity_score,
    -- Categorize performance
    CASE 
        WHEN median_minutes <= 120 THEN 'EXCELLENT'
        WHEN median_minutes <= 180 THEN 'GOOD'
        WHEN median_minutes <= 240 THEN 'FAIR'
        ELSE 'NEEDS IMPROVEMENT'
    END AS performance_category
FROM HourlyMetrics
ORDER BY arrival_hour;

-- =====================================================
-- SECURITY AND AUDIT QUERIES
-- =====================================================

-- 9. HIPAA COMPLIANCE AUDIT - Access Pattern Analysis
CREATE VIEW v_access_audit AS
SELECT 
    al.audit_id,
    al.user_id,
    al.access_timestamp,
    al.patient_id,
    al.access_type,
    al.ip_address,
    -- Identify suspicious patterns
    CASE 
        WHEN al.access_timestamp NOT BETWEEN '08:00:00' AND '20:00:00' THEN 'After Hours'
        WHEN (
            SELECT COUNT(DISTINCT patient_id) 
            FROM audit_log al2 
            WHERE al2.user_id = al.user_id
            AND DATE(al2.access_timestamp) = DATE(al.access_timestamp)
        ) > 50 THEN 'High Volume'
        WHEN NOT EXISTS (
            SELECT 1 
            FROM user_patient_relationship upr
            WHERE upr.user_id = al.user_id
            AND upr.patient_id = al.patient_id
        ) THEN 'No Relationship'
        ELSE 'Normal'
    END AS access_flag
FROM audit_log al;

-- 10. DATA QUALITY MONITORING
CREATE VIEW v_data_quality_metrics AS
SELECT 
    'fact_encounters' AS table_name,
    COUNT(*) AS total_records,
    SUM(CASE WHEN patient_id IS NULL THEN 1 ELSE 0 END) AS null_patient_ids,
    SUM(CASE WHEN admission_time_id IS NULL THEN 1 ELSE 0 END) AS null_admission_times,
    SUM(CASE WHEN primary_diagnosis_id IS NULL THEN 1 ELSE 0 END) AS null_diagnoses,
    SUM(CASE WHEN total_charges < 0 THEN 1 ELSE 0 END) AS negative_charges,
    SUM(CASE WHEN length_of_stay < 0 THEN 1 ELSE 0 END) AS negative_los
FROM fact_encounters
UNION ALL
SELECT 
    'fact_lab_results' AS table_name,
    COUNT(*) AS total_records,
    SUM(CASE WHEN patient_id IS NULL THEN 1 ELSE 0 END) AS null_patient_ids,
    SUM(CASE WHEN lab_test_code IS NULL THEN 1 ELSE 0 END) AS null_test_codes,
    SUM(CASE WHEN result_value IS NULL THEN 1 ELSE 0 END) AS null_results,
    SUM(CASE WHEN result_value < 0 AND lab_test_code NOT IN ('TEMP_DELTA') THEN 1 ELSE 0 END) AS unexpected_negatives,
    0 AS placeholder
FROM fact_lab_results;