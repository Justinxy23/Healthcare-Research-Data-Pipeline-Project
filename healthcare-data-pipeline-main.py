#!/usr/bin/env python3
"""
Healthcare Research Data Pipeline
Author: Justin Christopher Weaver
Date: 2025
Description: Secure research data engineering solution for healthcare analytics
"""

import os
import json
import logging
import hashlib
import datetime
import pandas as pd
import numpy as np
from typing import Dict, List, Any, Optional
from dataclasses import dataclass
from cryptography.fernet import Fernet
from sqlalchemy import create_engine, Column, Integer, String, DateTime, Float, Boolean, ForeignKey, Text
from sqlalchemy.ext.declarative import declarative_base
from sqlalchemy.orm import sessionmaker, relationship
from sqlalchemy.sql import text
import asyncio
import aiohttp
from abc import ABC, abstractmethod
import jwt
from functools import wraps
import re

# Security Configuration
Base = declarative_base()
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# ==================== Security Layer ====================
class SecurityManager:
    """Handles all security operations including encryption, authentication, and audit logging"""
    
    def __init__(self, secret_key: str = None):
        self.secret_key = secret_key or os.environ.get('SECRET_KEY', Fernet.generate_key())
        self.cipher = Fernet(self.secret_key)
        self.jwt_secret = os.environ.get('JWT_SECRET', 'your-jwt-secret-key')
        
    def encrypt_data(self, data: str) -> bytes:
        """Encrypt sensitive data using AES-256"""
        return self.cipher.encrypt(data.encode())
    
    def decrypt_data(self, encrypted_data: bytes) -> str:
        """Decrypt sensitive data"""
        return self.cipher.decrypt(encrypted_data).decode()
    
    def hash_pii(self, pii_data: str) -> str:
        """Create irreversible hash for PII data"""
        return hashlib.sha256(pii_data.encode()).hexdigest()
    
    def generate_token(self, user_id: str, role: str) -> str:
        """Generate JWT token for API authentication"""
        payload = {
            'user_id': user_id,
            'role': role,
            'exp': datetime.datetime.utcnow() + datetime.timedelta(hours=24)
        }
        return jwt.encode(payload, self.jwt_secret, algorithm='HS256')
    
    def verify_token(self, token: str) -> Dict:
        """Verify JWT token"""
        try:
            return jwt.decode(token, self.jwt_secret, algorithms=['HS256'])
        except jwt.InvalidTokenError:
            return None
    
    def audit_log(self, action: str, user: str, details: Dict):
        """Log security-relevant events"""
        log_entry = {
            'timestamp': datetime.datetime.utcnow().isoformat(),
            'action': action,
            'user': user,
            'details': details
        }
        logger.info(f"AUDIT: {json.dumps(log_entry)}")

# ==================== Database Models ====================
class Patient(Base):
    """Patient dimension table"""
    __tablename__ = 'dim_patient'
    
    patient_id = Column(Integer, primary_key=True)
    mrn_hash = Column(String(64), unique=True, index=True)  # Hashed MRN for privacy
    birth_year = Column(Integer)  # Year only for privacy
    gender = Column(String(10))
    race = Column(String(50))
    ethnicity = Column(String(50))
    created_at = Column(DateTime, default=datetime.datetime.utcnow)
    updated_at = Column(DateTime, onupdate=datetime.datetime.utcnow)
    
    encounters = relationship("Encounter", back_populates="patient")

class Encounter(Base):
    """Encounter fact table"""
    __tablename__ = 'fact_encounters'
    
    encounter_id = Column(Integer, primary_key=True)
    patient_id = Column(Integer, ForeignKey('dim_patient.patient_id'))
    encounter_date = Column(DateTime)
    discharge_date = Column(DateTime)
    encounter_type = Column(String(50))
    diagnosis_code = Column(String(20))
    procedure_code = Column(String(20))
    length_of_stay = Column(Integer)
    readmission_flag = Column(Boolean, default=False)
    total_charges = Column(Float)
    encrypted_notes = Column(Text)  # Encrypted clinical notes
    
    patient = relationship("Patient", back_populates="encounters")
    lab_results = relationship("LabResult", back_populates="encounter")

class LabResult(Base):
    """Lab results fact table"""
    __tablename__ = 'fact_lab_results'
    
    result_id = Column(Integer, primary_key=True)
    encounter_id = Column(Integer, ForeignKey('fact_encounters.encounter_id'))
    lab_date = Column(DateTime)
    lab_name = Column(String(100))
    lab_value = Column(Float)
    lab_units = Column(String(20))
    reference_range_low = Column(Float)
    reference_range_high = Column(Float)
    abnormal_flag = Column(Boolean)
    
    encounter = relationship("Encounter", back_populates="lab_results")

# ==================== Data Ingestion Layer ====================
class DataIngestionPipeline:
    """Handles secure data ingestion from multiple sources"""
    
    def __init__(self, security_manager: SecurityManager):
        self.security = security_manager
        self.supported_formats = ['csv', 'json', 'hl7', 'fhir']
        
    async def ingest_epic_clarity(self, connection_string: str, query: str) -> pd.DataFrame:
        """Ingest data from Epic Clarity database"""
        try:
            # Simulate Epic Clarity connection (in production, use actual Epic APIs)
            engine = create_engine(connection_string)
            
            # Epic-specific query optimization
            optimized_query = self._optimize_epic_query(query)
            
            # Execute with security context
            with engine.connect() as conn:
                df = pd.read_sql(optimized_query, conn)
                
            # Apply data masking
            df = self._mask_sensitive_columns(df)
            
            self.security.audit_log('data_ingestion', 'system', {
                'source': 'epic_clarity',
                'records': len(df)
            })
            
            return df
            
        except Exception as e:
            logger.error(f"Epic Clarity ingestion error: {str(e)}")
            raise
    
    def _optimize_epic_query(self, query: str) -> str:
        """Optimize queries for Epic's data model"""
        # Add Epic-specific optimizations
        optimizations = {
            'PAT_ENC': 'WITH (NOLOCK)',  # Prevent locking on high-traffic tables
            'ORDER_MED': 'WITH (INDEX(IX_ORDER_MED_PAT_ENC))',  # Use specific indexes
        }
        
        for table, hint in optimizations.items():
            if table in query:
                query = query.replace(f'FROM {table}', f'FROM {table} {hint}')
                
        return query
    
    def _mask_sensitive_columns(self, df: pd.DataFrame) -> pd.DataFrame:
        """Mask PII/PHI columns"""
        sensitive_patterns = {
            'ssn': r'\d{3}-\d{2}-\d{4}',
            'phone': r'\d{3}-\d{3}-\d{4}',
            'email': r'[\w\.-]+@[\w\.-]+\.\w+'
        }
        
        for column in df.columns:
            if any(pattern in column.lower() for pattern in ['ssn', 'phone', 'email', 'mrn']):
                if column.lower() == 'mrn':
                    df[column] = df[column].apply(self.security.hash_pii)
                else:
                    df[column] = df[column].apply(lambda x: '***MASKED***' if pd.notna(x) else x)
                    
        return df

# ==================== ETL Pipeline ====================
class ETLPipeline:
    """Extract, Transform, Load pipeline with data quality checks"""
    
    def __init__(self, security_manager: SecurityManager):
        self.security = security_manager
        self.quality_thresholds = {
            'completeness': 0.95,
            'uniqueness': 0.99,
            'validity': 0.98
        }
        
    def transform_patient_data(self, raw_data: pd.DataFrame) -> pd.DataFrame:
        """Transform patient data according to research requirements"""
        transformed = raw_data.copy()
        
        # Age calculation (store only year for privacy)
        if 'birth_date' in transformed.columns:
            transformed['birth_year'] = pd.to_datetime(transformed['birth_date']).dt.year
            transformed.drop('birth_date', axis=1, inplace=True)
        
        # Standardize categorical variables
        transformed['gender'] = transformed['gender'].map({
            'M': 'Male', 'F': 'Female', 'O': 'Other', 'U': 'Unknown'
        }).fillna('Unknown')
        
        # Data quality scoring
        quality_score = self._calculate_data_quality(transformed)
        
        if quality_score['overall'] < 0.9:
            logger.warning(f"Data quality below threshold: {quality_score}")
            
        return transformed
    
    def _calculate_data_quality(self, df: pd.DataFrame) -> Dict[str, float]:
        """Calculate data quality metrics"""
        metrics = {}
        
        # Completeness: percentage of non-null values
        metrics['completeness'] = df.notna().sum().sum() / (len(df) * len(df.columns))
        
        # Validity: check data types and ranges
        valid_count = 0
        total_count = len(df) * len(df.columns)
        
        for column in df.columns:
            if df[column].dtype in ['int64', 'float64']:
                # Check for outliers using IQR
                Q1 = df[column].quantile(0.25)
                Q3 = df[column].quantile(0.75)
                IQR = Q3 - Q1
                valid = df[column].between(Q1 - 1.5 * IQR, Q3 + 1.5 * IQR)
                valid_count += valid.sum()
            else:
                valid_count += df[column].notna().sum()
                
        metrics['validity'] = valid_count / total_count
        metrics['overall'] = np.mean(list(metrics.values()))
        
        return metrics

# ==================== Root Cause Analysis ====================
class RootCauseAnalyzer:
    """Automated root cause analysis for healthcare metrics"""
    
    def __init__(self):
        self.anomaly_threshold = 2.5  # Standard deviations
        
    def analyze_readmissions(self, engine) -> Dict[str, Any]:
        """Analyze factors contributing to readmissions"""
        query = """
        WITH ReadmissionFactors AS (
            SELECT 
                e1.patient_id,
                e1.encounter_id as initial_encounter,
                e2.encounter_id as readmission_encounter,
                e1.diagnosis_code,
                e1.length_of_stay,
                DATEDIFF(day, e1.discharge_date, e2.encounter_date) as days_to_readmission,
                COUNT(DISTINCT lr.result_id) as abnormal_lab_count
            FROM fact_encounters e1
            JOIN fact_encounters e2 
                ON e1.patient_id = e2.patient_id
                AND e2.encounter_date BETWEEN e1.discharge_date AND DATEADD(day, 30, e1.discharge_date)
            LEFT JOIN fact_lab_results lr 
                ON e1.encounter_id = lr.encounter_id 
                AND lr.abnormal_flag = 1
            WHERE e1.discharge_date >= DATEADD(month, -6, GETDATE())
            GROUP BY e1.patient_id, e1.encounter_id, e2.encounter_id, 
                     e1.diagnosis_code, e1.length_of_stay, e1.discharge_date, e2.encounter_date
        )
        SELECT 
            diagnosis_code,
            AVG(length_of_stay) as avg_los,
            AVG(days_to_readmission) as avg_days_to_readmit,
            AVG(abnormal_lab_count) as avg_abnormal_labs,
            COUNT(DISTINCT patient_id) as patient_count,
            COUNT(DISTINCT initial_encounter) as readmission_count
        FROM ReadmissionFactors
        GROUP BY diagnosis_code
        HAVING COUNT(DISTINCT patient_id) >= 10
        ORDER BY readmission_count DESC
        """
        
        with engine.connect() as conn:
            results = pd.read_sql(query, conn)
            
        # Identify anomalies
        anomalies = self._detect_anomalies(results)
        
        # Generate insights
        insights = self._generate_insights(results, anomalies)
        
        return {
            'data': results.to_dict('records'),
            'anomalies': anomalies,
            'insights': insights,
            'generated_at': datetime.datetime.utcnow().isoformat()
        }
    
    def _detect_anomalies(self, df: pd.DataFrame) -> List[Dict]:
        """Detect statistical anomalies in the data"""
        anomalies = []
        
        for column in df.select_dtypes(include=[np.number]).columns:
            mean = df[column].mean()
            std = df[column].std()
            
            # Find outliers
            outliers = df[np.abs(df[column] - mean) > self.anomaly_threshold * std]
            
            if not outliers.empty:
                anomalies.append({
                    'metric': column,
                    'outlier_count': len(outliers),
                    'outlier_values': outliers[column].tolist(),
                    'threshold': mean + self.anomaly_threshold * std
                })
                
        return anomalies
    
    def _generate_insights(self, data: pd.DataFrame, anomalies: List[Dict]) -> List[str]:
        """Generate actionable insights from analysis"""
        insights = []
        
        # High readmission diagnoses
        high_readmit = data.nlargest(5, 'readmission_count')
        insights.append(
            f"Top 5 diagnoses with highest readmissions: {', '.join(high_readmit['diagnosis_code'].tolist())}"
        )
        
        # Correlation analysis
        if len(data) > 10:
            corr = data[['avg_los', 'avg_days_to_readmit', 'avg_abnormal_labs']].corr()
            strong_corr = np.where(np.abs(corr) > 0.7)
            
            for i, j in zip(strong_corr[0], strong_corr[1]):
                if i < j:
                    insights.append(
                        f"Strong correlation ({corr.iloc[i, j]:.2f}) between "
                        f"{corr.columns[i]} and {corr.columns[j]}"
                    )
                    
        return insights

# ==================== Reporting Service ====================
class ReportingService:
    """Generate automated reports compatible with SQL Server Reporting Services"""
    
    def __init__(self, security_manager: SecurityManager):
        self.security = security_manager
        
    def generate_research_report(self, engine, report_type: str = 'monthly') -> Dict:
        """Generate comprehensive research report"""
        report_data = {}
        
        # Executive Summary
        report_data['executive_summary'] = self._generate_executive_summary(engine)
        
        # Detailed Metrics
        report_data['patient_metrics'] = self._get_patient_metrics(engine)
        report_data['quality_metrics'] = self._get_quality_metrics(engine)
        report_data['operational_metrics'] = self._get_operational_metrics(engine)
        
        # Security audit
        self.security.audit_log('report_generated', 'system', {
            'report_type': report_type,
            'timestamp': datetime.datetime.utcnow().isoformat()
        })
        
        return report_data
    
    def _generate_executive_summary(self, engine) -> Dict:
        """Generate executive summary with key metrics"""
        query = """
        SELECT 
            COUNT(DISTINCT patient_id) as total_patients,
            COUNT(DISTINCT encounter_id) as total_encounters,
            AVG(length_of_stay) as avg_los,
            SUM(CASE WHEN readmission_flag = 1 THEN 1 ELSE 0 END) * 100.0 / 
                COUNT(DISTINCT encounter_id) as readmission_rate,
            AVG(total_charges) as avg_charges
        FROM fact_encounters
        WHERE encounter_date >= DATEADD(month, -1, GETDATE())
        """
        
        with engine.connect() as conn:
            result = conn.execute(text(query)).fetchone()
            
        return {
            'total_patients': result[0],
            'total_encounters': result[1],
            'average_length_of_stay': round(result[2], 2),
            'readmission_rate': round(result[3], 2),
            'average_charges': round(result[4], 2)
        }
    
    def _get_patient_metrics(self, engine) -> List[Dict]:
        """Get detailed patient demographic metrics"""
        query = """
        SELECT 
            gender,
            race,
            COUNT(DISTINCT patient_id) as patient_count,
            AVG(2025 - birth_year) as avg_age
        FROM dim_patient
        GROUP BY gender, race
        ORDER BY patient_count DESC
        """
        
        with engine.connect() as conn:
            results = pd.read_sql(query, conn)
            
        return results.to_dict('records')
    
    def _get_quality_metrics(self, engine) -> Dict:
        """Get data quality metrics"""
        query = """
        SELECT 
            'Encounters' as table_name,
            COUNT(*) as total_records,
            SUM(CASE WHEN diagnosis_code IS NULL THEN 1 ELSE 0 END) as null_diagnosis,
            SUM(CASE WHEN procedure_code IS NULL THEN 1 ELSE 0 END) as null_procedure
        FROM fact_encounters
        UNION ALL
        SELECT 
            'Lab Results' as table_name,
            COUNT(*) as total_records,
            SUM(CASE WHEN lab_value IS NULL THEN 1 ELSE 0 END) as null_values,
            SUM(CASE WHEN abnormal_flag IS NULL THEN 1 ELSE 0 END) as null_flags
        FROM fact_lab_results
        """
        
        with engine.connect() as conn:
            results = pd.read_sql(query, conn)
            
        return results.to_dict('records')
    
    def _get_operational_metrics(self, engine) -> Dict:
        """Get operational efficiency metrics"""
        query = """
        WITH DailyMetrics AS (
            SELECT 
                CAST(encounter_date as DATE) as encounter_day,
                COUNT(DISTINCT patient_id) as daily_patients,
                COUNT(DISTINCT encounter_id) as daily_encounters,
                AVG(length_of_stay) as avg_los,
                AVG(total_charges) as avg_charges
            FROM fact_encounters
            WHERE encounter_date >= DATEADD(day, -30, GETDATE())
            GROUP BY CAST(encounter_date as DATE)
        )
        SELECT 
            AVG(daily_patients) as avg_daily_patients,
            AVG(daily_encounters) as avg_daily_encounters,
            AVG(avg_los) as overall_avg_los,
            AVG(avg_charges) as overall_avg_charges,
            STDEV(daily_patients) as patient_volume_variance
        FROM DailyMetrics
        """
        
        with engine.connect() as conn:
            result = conn.execute(text(query)).fetchone()
            
        return {
            'avg_daily_patients': round(result[0], 2),
            'avg_daily_encounters': round(result[1], 2),
            'avg_length_of_stay': round(result[2], 2),
            'avg_charges': round(result[3], 2),
            'patient_volume_variance': round(result[4], 2)
        }

# ==================== Main Application ====================
class HealthcareDataPipeline:
    """Main application orchestrator"""
    
    def __init__(self, database_url: str):
        self.engine = create_engine(database_url)
        Base.metadata.create_all(self.engine)
        self.Session = sessionmaker(bind=self.engine)
        
        # Initialize components
        self.security = SecurityManager()
        self.ingestion = DataIngestionPipeline(self.security)
        self.etl = ETLPipeline(self.security)
        self.analyzer = RootCauseAnalyzer()
        self.reporting = ReportingService(self.security)
        
    async def run_pipeline(self):
        """Run the complete data pipeline"""
        logger.info("Starting Healthcare Research Data Pipeline")
        
        try:
            # 1. Data Ingestion (simulated)
            logger.info("Phase 1: Data Ingestion")
            # In production, this would connect to real Epic systems
            sample_data = self._generate_sample_data()
            
            # 2. ETL Processing
            logger.info("Phase 2: ETL Processing")
            transformed_patients = self.etl.transform_patient_data(sample_data['patients'])
            
            # 3. Load to Data Warehouse
            logger.info("Phase 3: Loading to Data Warehouse")
            self._load_to_warehouse(transformed_patients, sample_data['encounters'], sample_data['labs'])
            
            # 4. Root Cause Analysis
            logger.info("Phase 4: Running Root Cause Analysis")
            rca_results = self.analyzer.analyze_readmissions(self.engine)
            logger.info(f"RCA Insights: {rca_results['insights']}")
            
            # 5. Generate Reports
            logger.info("Phase 5: Generating Reports")
            report = self.reporting.generate_research_report(self.engine)
            logger.info(f"Report Summary: {report['executive_summary']}")
            
            # 6. Security Audit
            self.security.audit_log('pipeline_completed', 'system', {
                'duration': 'simulated',
                'records_processed': len(transformed_patients)
            })
            
            logger.info("Pipeline completed successfully")
            
        except Exception as e:
            logger.error(f"Pipeline error: {str(e)}")
            self.security.audit_log('pipeline_error', 'system', {'error': str(e)})
            raise
    
    def _generate_sample_data(self) -> Dict[str, pd.DataFrame]:
        """Generate sample healthcare data for demonstration"""
        np.random.seed(42)
        
        # Generate patients
        n_patients = 1000
        patients = pd.DataFrame({
            'patient_id': range(1, n_patients + 1),
            'mrn': [f'MRN{i:06d}' for i in range(1, n_patients + 1)],
            'birth_date': pd.date_range('1940-01-01', '2005-01-01', periods=n_patients),
            'gender': np.random.choice(['M', 'F', 'O'], n_patients, p=[0.48, 0.48, 0.04]),
            'race': np.random.choice(['White', 'Black', 'Asian', 'Hispanic', 'Other'], 
                                   n_patients, p=[0.6, 0.13, 0.06, 0.18, 0.03]),
            'ethnicity': np.random.choice(['Hispanic', 'Non-Hispanic'], n_patients, p=[0.18, 0.82])
        })
        
        # Generate encounters
        n_encounters = 5000
        encounters = pd.DataFrame({
            'encounter_id': range(1, n_encounters + 1),
            'patient_id': np.random.choice(patients['patient_id'], n_encounters),
            'encounter_date': pd.date_range('2024-01-01', '2024-12-31', periods=n_encounters),
            'encounter_type': np.random.choice(['Inpatient', 'Outpatient', 'Emergency'], 
                                             n_encounters, p=[0.2, 0.6, 0.2]),
            'diagnosis_code': np.random.choice(['I10', 'E11.9', 'J44.1', 'N18.3', 'F32.9'], 
                                             n_encounters),
            'procedure_code': np.random.choice(['99213', '99214', '99285', '99232', None], 
                                             n_encounters, p=[0.3, 0.3, 0.2, 0.15, 0.05]),
            'length_of_stay': np.random.poisson(3, n_encounters),
            'total_charges': np.random.gamma(2, 2000, n_encounters),
            'readmission_flag': np.random.choice([True, False], n_encounters, p=[0.15, 0.85])
        })
        
        encounters['discharge_date'] = encounters['encounter_date'] + pd.to_timedelta(
            encounters['length_of_stay'], unit='D'
        )
        
        # Generate lab results
        n_labs = 10000
        labs = pd.DataFrame({
            'result_id': range(1, n_labs + 1),
            'encounter_id': np.random.choice(encounters['encounter_id'], n_labs),
            'lab_date': pd.date_range('2024-01-01', '2024-12-31', periods=n_labs),
            'lab_name': np.random.choice(['Glucose', 'Creatinine', 'Hemoglobin', 'WBC', 'Platelet'], 
                                       n_labs),
            'lab_value': np.random.normal(100, 20, n_labs),
            'lab_units': 'mg/dL',
            'reference_range_low': 70,
            'reference_range_high': 130,
            'abnormal_flag': np.random.choice([True, False], n_labs, p=[0.2, 0.8])
        })
        
        return {
            'patients': patients,
            'encounters': encounters,
            'labs': labs
        }
    
    def _load_to_warehouse(self, patients_df: pd.DataFrame, encounters_df: pd.DataFrame, 
                          labs_df: pd.DataFrame):
        """Load transformed data to the data warehouse"""
        session = self.Session()
        
        try:
            # Load patients
            for _, row in patients_df.iterrows():
                patient = Patient(
                    patient_id=row['patient_id'],
                    mrn_hash=self.security.hash_pii(row['mrn']),
                    birth_year=row['birth_year'],
                    gender=row['gender'],
                    race=row['race'],
                    ethnicity=row['ethnicity']
                )
                session.merge(patient)
            
            # Load encounters
            for _, row in encounters_df.iterrows():
                # Encrypt clinical notes (simulated)
                encrypted_notes = self.security.encrypt_data(
                    f"Clinical notes for encounter {row['encounter_id']}"
                )
                
                encounter = Encounter(
                    encounter_id=row['encounter_id'],
                    patient_id=row['patient_id'],
                    encounter_date=row['encounter_date'],
                    discharge_date=row['discharge_date'],
                    encounter_type=row['encounter_type'],
                    diagnosis_code=row['diagnosis_code'],
                    procedure_code=row['procedure_code'],
                    length_of_stay=row['length_of_stay'],
                    total_charges=row['total_charges'],
                    readmission_flag=row['readmission_flag'],
                    encrypted_notes=encrypted_notes.decode()
                )
                session.merge(encounter)
            
            # Load lab results
            for _, row in labs_df.iterrows():
                lab = LabResult(
                    result_id=row['result_id'],
                    encounter_id=row['encounter_id'],
                    lab_date=row['lab_date'],
                    lab_name=row['lab_name'],
                    lab_value=row['lab_value'],
                    lab_units=row['lab_units'],
                    reference_range_low=row['reference_range_low'],
                    reference_range_high=row['reference_range_high'],
                    abnormal_flag=row['abnormal_flag']
                )
                session.merge(lab)
            
            session.commit()
            logger.info(f"Successfully loaded {len(patients_df)} patients, "
                       f"{len(encounters_df)} encounters, {len(labs_df)} lab results")
            
        except Exception as e:
            session.rollback()
            logger.error(f"Data loading error: {str(e)}")
            raise
        finally:
            session.close()

# ==================== CLI Interface ====================
def main():
    """Main entry point"""
    import argparse
    
    parser = argparse.ArgumentParser(description='Healthcare Research Data Pipeline')
    parser.add_argument('--database-url', default='sqlite:///healthcare_research.db',
                       help='Database connection URL')
    parser.add_argument('--run-pipeline', action='store_true',
                       help='Run the complete pipeline')
    parser.add_argument('--generate-report', action='store_true',
                       help='Generate research report')
    parser.add_argument('--run-rca', action='store_true',
                       help='Run root cause analysis')
    
    args = parser.parse_args()
    
    # Initialize pipeline
    pipeline = HealthcareDataPipeline(args.database_url)
    
    if args.run_pipeline:
        asyncio.run(pipeline.run_pipeline())
    elif args.generate_report:
        report = pipeline.reporting.generate_research_report(pipeline.engine)
        print(json.dumps(report, indent=2))
    elif args.run_rca:
        rca_results = pipeline.analyzer.analyze_readmissions(pipeline.engine)
        print(json.dumps(rca_results, indent=2))
    else:
        # Run full pipeline by default
        asyncio.run(pipeline.run_pipeline())

if __name__ == "__main__":
    main()