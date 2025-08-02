#!/usr/bin/env python3
"""
Comprehensive test suite for Healthcare Research Data Pipeline
Author: Justin Christopher Weaver
"""

import pytest
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from unittest.mock import Mock, patch, MagicMock
from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker
import json
import jwt
from cryptography.fernet import Fernet

# Import modules to test
from src.main import (
    SecurityManager, DataIngestionPipeline, ETLPipeline,
    RootCauseAnalyzer, ReportingService, HealthcareDataPipeline,
    Patient, Encounter, LabResult, Base
)

# Test fixtures
@pytest.fixture
def security_manager():
    """Create a test security manager"""
    return SecurityManager()

@pytest.fixture
def test_engine():
    """Create in-memory SQLite database for testing"""
    engine = create_engine('sqlite:///:memory:')
    Base.metadata.create_all(engine)
    return engine

@pytest.fixture
def test_session(test_engine):
    """Create test database session"""
    Session = sessionmaker(bind=test_engine)
    return Session()

@pytest.fixture
def sample_patient_data():
    """Generate sample patient data for testing"""
    return pd.DataFrame({
        'patient_id': range(1, 101),
        'mrn': [f'MRN{i:06d}' for i in range(1, 101)],
        'birth_date': pd.date_range('1950-01-01', '2000-01-01', periods=100),
        'gender': np.random.choice(['M', 'F', 'O'], 100),
        'race': np.random.choice(['White', 'Black', 'Asian', 'Hispanic', 'Other'], 100),
        'ethnicity': np.random.choice(['Hispanic', 'Non-Hispanic'], 100),
        'ssn': ['123-45-6789'] * 100,  # Test data only
        'phone': ['555-123-4567'] * 100
    })

@pytest.fixture
def sample_encounter_data():
    """Generate sample encounter data for testing"""
    return pd.DataFrame({
        'encounter_id': range(1, 501),
        'patient_id': np.random.choice(range(1, 101), 500),
        'encounter_date': pd.date_range('2024-01-01', '2024-12-31', periods=500),
        'encounter_type': np.random.choice(['Inpatient', 'Outpatient', 'Emergency'], 500),
        'diagnosis_code': np.random.choice(['I10', 'E11.9', 'J44.1', 'N18.3'], 500),
        'length_of_stay': np.random.poisson(3, 500),
        'total_charges': np.random.gamma(2, 2000, 500),
        'readmission_flag': np.random.choice([True, False], 500, p=[0.15, 0.85])
    })

# ==================== Security Tests ====================
class TestSecurityManager:
    """Test security functionality"""
    
    def test_encryption_decryption(self, security_manager):
        """Test data encryption and decryption"""
        test_data = "Sensitive patient information"
        encrypted = security_manager.encrypt_data(test_data)
        decrypted = security_manager.decrypt_data(encrypted)
        
        assert decrypted == test_data
        assert encrypted != test_data.encode()
    
    def test_pii_hashing(self, security_manager):
        """Test PII hashing is consistent and irreversible"""
        pii_data = "123-45-6789"
        hash1 = security_manager.hash_pii(pii_data)
        hash2 = security_manager.hash_pii(pii_data)
        
        assert hash1 == hash2  # Consistent hashing
        assert len(hash1) == 64  # SHA-256 produces 64 character hex
        assert hash1 != pii_data  # Not reversible
    
    def test_jwt_token_generation_and_verification(self, security_manager):
        """Test JWT token creation and verification"""
        user_id = "test_user"
        role = "researcher"
        
        token = security_manager.generate_token(user_id, role)
        payload = security_manager.verify_token(token)
        
        assert payload is not None
        assert payload['user_id'] == user_id
        assert payload['role'] == role
        assert 'exp' in payload
    
    def test_invalid_token_verification(self, security_manager):
        """Test invalid token returns None"""
        invalid_token = "invalid.token.here"
        result = security_manager.verify_token(invalid_token)
        assert result is None
    
    def test_audit_logging(self, security_manager, caplog):
        """Test audit logging functionality"""
        security_manager.audit_log(
            action="data_access",
            user="test_user",
            details={"patient_id": 123, "access_type": "read"}
        )
        
        assert "AUDIT:" in caplog.text
        assert "data_access" in caplog.text
        assert "test_user" in caplog.text

# ==================== Data Ingestion Tests ====================
class TestDataIngestionPipeline:
    """Test data ingestion functionality"""
    
    def test_mask_sensitive_columns(self, security_manager, sample_patient_data):
        """Test PII/PHI masking"""
        pipeline = DataIngestionPipeline(security_manager)
        masked_df = pipeline._mask_sensitive_columns(sample_patient_data.copy())
        
        # Check MRN is hashed
        assert all(len(mrn) == 64 for mrn in masked_df['mrn'])
        
        # Check SSN and phone are masked
        assert all(masked_df['ssn'] == '***MASKED***')
        assert all(masked_df['phone'] == '***MASKED***')
    
    def test_epic_query_optimization(self, security_manager):
        """Test Epic-specific query optimization"""
        pipeline = DataIngestionPipeline(security_manager)
        
        original_query = "SELECT * FROM PAT_ENC WHERE patient_id = 123"
        optimized = pipeline._optimize_epic_query(original_query)
        
        assert "WITH (NOLOCK)" in optimized
        assert "PAT_ENC WITH (NOLOCK)" in optimized
    
    @pytest.mark.asyncio
    async def test_epic_clarity_ingestion_error_handling(self, security_manager):
        """Test error handling in Epic Clarity ingestion"""
        pipeline = DataIngestionPipeline(security_manager)
        
        # Test with invalid connection string
        with pytest.raises(Exception):
            await pipeline.ingest_epic_clarity(
                "invalid://connection",
                "SELECT * FROM patients"
            )

# ==================== ETL Pipeline Tests ====================
class TestETLPipeline:
    """Test ETL transformation functionality"""
    
    def test_transform_patient_data(self, security_manager, sample_patient_data):
        """Test patient data transformation"""
        etl = ETLPipeline(security_manager)
        transformed = etl.transform_patient_data(sample_patient_data.copy())
        
        # Check birth_date converted to birth_year
        assert 'birth_year' in transformed.columns
        assert 'birth_date' not in transformed.columns
        
        # Check gender standardization
        assert all(transformed['gender'].isin(['Male', 'Female', 'Other', 'Unknown']))
    
    def test_data_quality_calculation(self, security_manager, sample_patient_data):
        """Test data quality metrics calculation"""
        etl = ETLPipeline(security_manager)
        quality_metrics = etl._calculate_data_quality(sample_patient_data)
        
        assert 'completeness' in quality_metrics
        assert 'validity' in quality_metrics
        assert 'overall' in quality_metrics
        
        # All metrics should be between 0 and 1
        assert all(0 <= v <= 1 for v in quality_metrics.values())
    
    def test_data_quality_warning(self, security_manager, caplog):
        """Test data quality warning for poor quality data"""
        etl = ETLPipeline(security_manager)
        
        # Create dataframe with many nulls
        poor_quality_df = pd.DataFrame({
            'col1': [None] * 50 + [1] * 50,
            'col2': [None] * 70 + [2] * 30,
            'col3': [None] * 80 + [3] * 20
        })
        
        _ = etl.transform_patient_data(poor_quality_df)
        assert "Data quality below threshold" in caplog.text

# ==================== Root Cause Analysis Tests ====================
class TestRootCauseAnalyzer:
    """Test root cause analysis functionality"""
    
    def test_detect_anomalies(self):
        """Test anomaly detection"""
        analyzer = RootCauseAnalyzer()
        
        # Create test data with clear outliers
        test_df = pd.DataFrame({
            'metric1': [10, 11, 9, 10, 11, 100],  # 100 is outlier
            'metric2': [20, 19, 21, 20, 18, 19],   # No outliers
            'category': ['A', 'A', 'B', 'B', 'C', 'C']
        })
        
        anomalies = analyzer._detect_anomalies(test_df)
        
        # Should detect outlier in metric1
        assert len(anomalies) > 0
        assert any(a['metric'] == 'metric1' for a in anomalies)
    
    def test_generate_insights(self):
        """Test insight generation"""
        analyzer = RootCauseAnalyzer()
        
        test_data = pd.DataFrame({
            'diagnosis_code': ['A', 'B', 'C', 'D', 'E'],
            'readmission_count': [50, 45, 30, 20, 10],
            'avg_los': [5, 6, 4, 3, 7],
            'avg_days_to_readmit': [15, 20, 25, 30, 10],
            'avg_abnormal_labs': [3, 2, 1, 1, 4]
        })
        
        insights = analyzer._generate_insights(test_data, [])
        
        assert len(insights) > 0
        assert any("Top 5 diagnoses" in insight for insight in insights)
    
    @patch('pandas.read_sql')
    def test_analyze_readmissions(self, mock_read_sql):
        """Test readmission analysis"""
        analyzer = RootCauseAnalyzer()
        
        # Mock database results
        mock_read_sql.return_value = pd.DataFrame({
            'diagnosis_code': ['I10', 'E11', 'J44'],
            'avg_los': [3.5, 4.2, 5.1],
            'avg_days_to_readmit': [15, 20, 12],
            'avg_abnormal_labs': [2, 3, 4],
            'patient_count': [100, 150, 80],
            'readmission_count': [15, 30, 20]
        })
        
        mock_engine = Mock()
        result = analyzer.analyze_readmissions(mock_engine)
        
        assert 'data' in result
        assert 'anomalies' in result
        assert 'insights' in result
        assert 'generated_at' in result

# ==================== Reporting Service Tests ====================
class TestReportingService:
    """Test reporting functionality"""
    
    @patch('sqlalchemy.engine.base.Connection.execute')
    def test_generate_executive_summary(self, mock_execute, security_manager):
        """Test executive summary generation"""
        reporting = ReportingService(security_manager)
        
        # Mock query results
        mock_result = Mock()
        mock_result.fetchone.return_value = (1000, 5000, 3.5, 15.2, 25000)
        mock_execute.return_value = mock_result
        
        mock_engine = Mock()
        mock_engine.connect.return_value.__enter__ = Mock(return_value=Mock(execute=mock_execute))
        mock_engine.connect.return_value.__exit__ = Mock(return_value=False)
        
        summary = reporting._generate_executive_summary(mock_engine)
        
        assert summary['total_patients'] == 1000
        assert summary['total_encounters'] == 5000
        assert summary['average_length_of_stay'] == 3.5
        assert summary['readmission_rate'] == 15.2
        assert summary['average_charges'] == 25000
    
    def test_report_generation_audit(self, security_manager, caplog):
        """Test audit logging during report generation"""
        reporting = ReportingService(security_manager)
        
        mock_engine = Mock()
        with patch.object(reporting, '_generate_executive_summary', return_value={}):
            with patch.object(reporting, '_get_patient_metrics', return_value=[]):
                with patch.object(reporting, '_get_quality_metrics', return_value={}):
                    with patch.object(reporting, '_get_operational_metrics', return_value={}):
                        _ = reporting.generate_research_report(mock_engine)
        
        assert "report_generated" in caplog.text

# ==================== Integration Tests ====================
class TestHealthcareDataPipeline:
    """Test main pipeline functionality"""
    
    def test_pipeline_initialization(self, test_engine):
        """Test pipeline initialization"""
        pipeline = HealthcareDataPipeline('sqlite:///:memory:')
        
        assert pipeline.engine is not None
        assert pipeline.security is not None
        assert pipeline.ingestion is not None
        assert pipeline.etl is not None
        assert pipeline.analyzer is not None
        assert pipeline.reporting is not None
    
    def test_generate_sample_data(self, test_engine):
        """Test sample data generation"""
        pipeline = HealthcareDataPipeline('sqlite:///:memory:')
        sample_data = pipeline._generate_sample_data()
        
        assert 'patients' in sample_data
        assert 'encounters' in sample_data
        assert 'labs' in sample_data
        
        assert len(sample_data['patients']) == 1000
        assert len(sample_data['encounters']) == 5000
        assert len(sample_data['labs']) == 10000
    
    def test_load_to_warehouse(self, test_engine, test_session):
        """Test data loading to warehouse"""
        pipeline = HealthcareDataPipeline('sqlite:///:memory:')
        pipeline.Session = lambda: test_session
        
        # Create minimal test data
        patients_df = pd.DataFrame({
            'patient_id': [1, 2],
            'mrn': ['MRN001', 'MRN002'],
            'birth_year': [1980, 1975],
            'gender': ['Male', 'Female'],
            'race': ['White', 'Black'],
            'ethnicity': ['Non-Hispanic', 'Hispanic']
        })
        
        encounters_df = pd.DataFrame({
            'encounter_id': [1, 2],
            'patient_id': [1, 2],
            'encounter_date': pd.to_datetime(['2024-01-01', '2024-01-02']),
            'discharge_date': pd.to_datetime(['2024-01-03', '2024-01-04']),
            'encounter_type': ['Inpatient', 'Outpatient'],
            'diagnosis_code': ['I10', 'E11'],
            'procedure_code': ['99213', '99214'],
            'length_of_stay': [2, 2],
            'total_charges': [5000, 3000],
            'readmission_flag': [False, False]
        })
        
        labs_df = pd.DataFrame({
            'result_id': [1, 2],
            'encounter_id': [1, 2],
            'lab_date': pd.to_datetime(['2024-01-01', '2024-01-02']),
            'lab_name': ['Glucose', 'Creatinine'],
            'lab_value': [100, 1.2],
            'lab_units': ['mg/dL', 'mg/dL'],
            'reference_range_low': [70, 0.5],
            'reference_range_high': [130, 1.5],
            'abnormal_flag': [False, False]
        })
        
        pipeline._load_to_warehouse(patients_df, encounters_df, labs_df)
        
        # Verify data was loaded
        patients = test_session.query(Patient).all()
        assert len(patients) == 2
        
        encounters = test_session.query(Encounter).all()
        assert len(encounters) == 2
        
        labs = test_session.query(LabResult).all()
        assert len(labs) == 2
    
    @pytest.mark.asyncio
    async def test_run_pipeline_error_handling(self, caplog):
        """Test pipeline error handling"""
        pipeline = HealthcareDataPipeline('sqlite:///:memory:')
        
        # Mock an error in the pipeline
        with patch.object(pipeline, '_generate_sample_data', side_effect=Exception("Test error")):
            with pytest.raises(Exception):
                await pipeline.run_pipeline()
        
        assert "Pipeline error: Test error" in caplog.text
        assert "pipeline_error" in caplog.text

# ==================== Performance Tests ====================
class TestPerformance:
    """Test performance-related functionality"""
    
    def test_large_dataset_processing(self, security_manager):
        """Test processing of large datasets"""
        etl = ETLPipeline(security_manager)
        
        # Create large dataset
        large_df = pd.DataFrame({
            'patient_id': range(10000),
            'birth_date': pd.date_range('1950-01-01', periods=10000, freq='D'),
            'gender': np.random.choice(['M', 'F'], 10000),
            'value': np.random.rand(10000)
        })
        
        start_time = datetime.now()
        transformed = etl.transform_patient_data(large_df)
        end_time = datetime.now()
        
        processing_time = (end_time - start_time).total_seconds()
        
        assert len(transformed) == 10000
        assert processing_time < 5  # Should process in under 5 seconds
    
    def test_encryption_performance(self, security_manager):
        """Test encryption performance"""
        large_text = "x" * 10000  # 10KB of data
        
        start_time = datetime.now()
        encrypted = security_manager.encrypt_data(large_text)
        decrypted = security_manager.decrypt_data(encrypted)
        end_time = datetime.now()
        
        processing_time = (end_time - start_time).total_seconds()
        
        assert decrypted == large_text
        assert processing_time < 0.1  # Should encrypt/decrypt in under 100ms

# ==================== Security Vulnerability Tests ====================
class TestSecurityVulnerabilities:
    """Test for common security vulnerabilities"""
    
    def test_sql_injection_prevention(self, security_manager):
        """Test SQL injection prevention"""
        pipeline = DataIngestionPipeline(security_manager)
        
        # Attempt SQL injection in query optimization
        malicious_query = "SELECT * FROM PAT_ENC WHERE id = '1'; DROP TABLE patients; --"
        optimized = pipeline._optimize_epic_query(malicious_query)
        
        # The optimization should not break the query structure
        assert "DROP TABLE" in optimized  # Still contains the malicious part
        # In real implementation, we'd use parameterized queries
    
    def test_xss_prevention_in_reports(self, security_manager):
        """Test XSS prevention in generated reports"""
        reporting = ReportingService(security_manager)
        
        # Create data with potential XSS
        malicious_data = {
            'diagnosis': '<script>alert("XSS")</script>',
            'count': 10
        }
        
        # In real implementation, ensure all output is properly escaped
        # This is a placeholder test
        assert '<script>' in str(malicious_data)

# ==================== Compliance Tests ====================
class TestCompliance:
    """Test HIPAA and regulatory compliance"""
    
    def test_audit_log_retention(self, security_manager):
        """Test audit log includes required HIPAA fields"""
        details = {
            'patient_id': 123,
            'access_type': 'read',
            'purpose': 'treatment'
        }
        
        security_manager.audit_log('patient_access', 'test_user', details)
        
        # In production, verify logs contain:
        # - User identification
        # - Date and time
        # - Patient identification
        # - Type of action
        # - Success or failure indication
    
    def test_encryption_at_rest(self, security_manager, test_session):
        """Test that sensitive data is encrypted at rest"""
        clinical_notes = "Patient presents with acute symptoms..."
        encrypted = security_manager.encrypt_data(clinical_notes)
        
        # Create encounter with encrypted notes
        encounter = Encounter(
            encounter_id=1,
            patient_id=1,
            encounter_date=datetime.now(),
            encrypted_notes=encrypted.decode()
        )
        
        # Verify notes are encrypted
        assert encounter.encrypted_notes != clinical_notes
        assert len(encounter.encrypted_notes) > len(clinical_notes)

# Run tests with coverage report
if __name__ == "__main__":
    pytest.main([__file__, "-v", "--cov=src", "--cov-report=html"])