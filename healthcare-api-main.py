#!/usr/bin/env python3
"""
Healthcare Research Data Pipeline - RESTful API
Author: Justin Christopher Weaver
Description: Secure API endpoints for healthcare data access and analytics
"""

from fastapi import FastAPI, HTTPException, Depends, Security, status, Request
from fastapi.security import HTTPBearer, HTTPAuthorizationCredentials
from fastapi.middleware.cors import CORSMiddleware
from fastapi.middleware.trustedhost import TrustedHostMiddleware
from fastapi.responses import JSONResponse
from pydantic import BaseModel, Field, validator
from typing import List, Optional, Dict, Any
from datetime import datetime, timedelta
import os
from sqlalchemy.orm import Session
from sqlalchemy import func, and_, or_
import logging
from functools import wraps
import time
import redis
import json
from contextlib import asynccontextmanager

# Import from main application
from ..main import (
    SecurityManager, RootCauseAnalyzer, ReportingService,
    Patient, Encounter, LabResult, Base, create_engine, sessionmaker
)

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Initialize Redis for caching and rate limiting
redis_client = redis.from_url(os.environ.get('REDIS_URL', 'redis://localhost:6379'))

# Security scheme
security = HTTPBearer()

# ==================== Lifespan Management ====================
@asynccontextmanager
async def lifespan(app: FastAPI):
    """Manage application lifecycle"""
    # Startup
    logger.info("Starting Healthcare API...")
    app.state.security_manager = SecurityManager()
    app.state.analyzer = RootCauseAnalyzer()
    app.state.reporting = ReportingService(app.state.security_manager)
    
    # Initialize database
    database_url = os.environ.get('DATABASE_URL', 'sqlite:///healthcare_research.db')
    app.state.engine = create_engine(database_url)
    app.state.SessionLocal = sessionmaker(bind=app.state.engine)
    
    yield
    
    # Shutdown
    logger.info("Shutting down Healthcare API...")
    redis_client.close()

# Initialize FastAPI app
app = FastAPI(
    title="Healthcare Research Data Pipeline API",
    description="Secure API for healthcare analytics and research data management",
    version="1.0.0",
    docs_url="/api/docs",
    redoc_url="/api/redoc",
    lifespan=lifespan
)

# ==================== Middleware ====================
# CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=os.environ.get('ALLOWED_ORIGINS', '*').split(','),
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Trusted host middleware
app.add_middleware(
    TrustedHostMiddleware,
    allowed_hosts=os.environ.get('ALLOWED_HOSTS', '*').split(',')
)

# ==================== Request/Response Models ====================
class TokenRequest(BaseModel):
    username: str = Field(..., min_length=3, max_length=50)
    password: str = Field(..., min_length=8)
    role: str = Field(..., regex="^(researcher|analyst|admin|viewer)$")

class TokenResponse(BaseModel):
    access_token: str
    token_type: str = "bearer"
    expires_in: int = 86400

class PatientSearchRequest(BaseModel):
    gender: Optional[str] = None
    race: Optional[str] = None
    min_age: Optional[int] = Field(None, ge=0, le=120)
    max_age: Optional[int] = Field(None, ge=0, le=120)
    diagnosis_codes: Optional[List[str]] = None
    
    @validator('max_age')
    def validate_age_range(cls, v, values):
        if v and 'min_age' in values and values['min_age'] and v < values['min_age']:
            raise ValueError('max_age must be greater than min_age')
        return v

class EncounterMetricsRequest(BaseModel):
    start_date: datetime
    end_date: datetime
    encounter_type: Optional[str] = None
    group_by: str = Field("month", regex="^(day|week|month|quarter)$")
    
    @validator('end_date')
    def validate_date_range(cls, v, values):
        if 'start_date' in values and v < values['start_date']:
            raise ValueError('end_date must be after start_date')
        if v > datetime.now():
            raise ValueError('end_date cannot be in the future')
        return v

class QualityMetricsResponse(BaseModel):
    metric_name: str
    value: float
    benchmark: float
    status: str
    trend: str
    details: Dict[str, Any]

class RCARequest(BaseModel):
    analysis_type: str = Field(..., regex="^(readmissions|infections|mortality|los)$")
    time_period_days: int = Field(90, ge=30, le=365)
    min_sample_size: int = Field(30, ge=10)

# ==================== Dependencies ====================
def get_db():
    """Database session dependency"""
    db = app.state.SessionLocal()
    try:
        yield db
    finally:
        db.close()

def verify_token(credentials: HTTPAuthorizationCredentials = Security(security)) -> Dict:
    """Verify JWT token"""
    token = credentials.credentials
    payload = app.state.security_manager.verify_token(token)
    
    if not payload:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Invalid or expired token"
        )
    
    return payload

def require_role(required_roles: List[str]):
    """Role-based access control decorator"""
    def decorator(func):
        @wraps(func)
        async def wrapper(*args, **kwargs):
            # Get token from kwargs
            current_user = kwargs.get('current_user', {})
            user_role = current_user.get('role', '')
            
            if user_role not in required_roles:
                raise HTTPException(
                    status_code=status.HTTP_403_FORBIDDEN,
                    detail=f"Insufficient permissions. Required roles: {required_roles}"
                )
            
            return await func(*args, **kwargs)
        return wrapper
    return decorator

def rate_limit(max_calls: int = 100, window_seconds: int = 3600):
    """Rate limiting decorator"""
    def decorator(func):
        @wraps(func)
        async def wrapper(request: Request, *args, **kwargs):
            # Get client IP
            client_ip = request.client.host
            key = f"rate_limit:{client_ip}:{func.__name__}"
            
            try:
                current = redis_client.incr(key)
                if current == 1:
                    redis_client.expire(key, window_seconds)
                
                if current > max_calls:
                    raise HTTPException(
                        status_code=status.HTTP_429_TOO_MANY_REQUESTS,
                        detail=f"Rate limit exceeded. Max {max_calls} calls per {window_seconds} seconds"
                    )
            except redis.RedisError:
                # If Redis is down, allow the request
                logger.error("Redis error during rate limiting")
            
            return await func(request, *args, **kwargs)
        return wrapper
    return decorator

# ==================== Authentication Endpoints ====================
@app.post("/api/v1/auth/token", response_model=TokenResponse)
async def login(request: TokenRequest):
    """Authenticate user and generate JWT token"""
    # In production, verify credentials against database
    # This is a simplified version
    if len(request.password) < 8:  # Basic validation
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Invalid credentials"
        )
    
    # Generate token
    token = app.state.security_manager.generate_token(
        user_id=request.username,
        role=request.role
    )
    
    # Audit log
    app.state.security_manager.audit_log(
        action="login",
        user=request.username,
        details={"role": request.role, "ip": "127.0.0.1"}
    )
    
    return TokenResponse(access_token=token, expires_in=86400)

# ==================== Patient Data Endpoints ====================
@app.post("/api/v1/patients/search")
@rate_limit(max_calls=100)
async def search_patients(
    request: Request,
    search_request: PatientSearchRequest,
    db: Session = Depends(get_db),
    current_user: Dict = Depends(verify_token)
):
    """Search for patients based on demographics (de-identified)"""
    query = db.query(Patient)
    
    # Apply filters
    if search_request.gender:
        query = query.filter(Patient.gender == search_request.gender)
    
    if search_request.race:
        query = query.filter(Patient.race == search_request.race)
    
    # Age filter
    current_year = datetime.now().year
    if search_request.min_age:
        query = query.filter(Patient.birth_year <= current_year - search_request.min_age)
    if search_request.max_age:
        query = query.filter(Patient.birth_year >= current_year - search_request.max_age)
    
    # Diagnosis filter
    if search_request.diagnosis_codes:
        query = query.join(Encounter).filter(
            Encounter.diagnosis_code.in_(search_request.diagnosis_codes)
        )
    
    # Execute query with limit
    patients = query.limit(1000).all()
    
    # Audit log
    app.state.security_manager.audit_log(
        action="patient_search",
        user=current_user['user_id'],
        details={
            "filters": search_request.dict(),
            "results_count": len(patients)
        }
    )
    
    # Return de-identified results
    return {
        "count": len(patients),
        "demographics": {
            "gender_distribution": _calculate_distribution(patients, 'gender'),
            "race_distribution": _calculate_distribution(patients, 'race'),
            "age_distribution": _calculate_age_distribution(patients)
        }
    }

# ==================== Analytics Endpoints ====================
@app.post("/api/v1/analytics/encounters")
@rate_limit(max_calls=50)
async def get_encounter_metrics(
    request: Request,
    metrics_request: EncounterMetricsRequest,
    db: Session = Depends(get_db),
    current_user: Dict = Depends(verify_token)
):
    """Get encounter metrics for specified time period"""
    # Build query
    query = db.query(
        func.date_trunc(metrics_request.group_by, Encounter.encounter_date).label('period'),
        func.count(Encounter.encounter_id).label('count'),
        func.avg(Encounter.length_of_stay).label('avg_los'),
        func.avg(Encounter.total_charges).label('avg_charges'),
        func.sum(Encounter.readmission_flag).label('readmissions')
    ).join(
        Patient
    ).filter(
        and_(
            Encounter.encounter_date >= metrics_request.start_date,
            Encounter.encounter_date <= metrics_request.end_date
        )
    )
    
    if metrics_request.encounter_type:
        query = query.filter(Encounter.encounter_type == metrics_request.encounter_type)
    
    # Group by period
    results = query.group_by('period').order_by('period').all()
    
    # Format results
    metrics = []
    for row in results:
        metrics.append({
            "period": row.period.isoformat() if row.period else None,
            "encounters": row.count,
            "average_los": round(float(row.avg_los), 2) if row.avg_los else 0,
            "average_charges": round(float(row.avg_charges), 2) if row.avg_charges else 0,
            "readmission_rate": round(row.readmissions / row.count * 100, 2) if row.count > 0 else 0
        })
    
    return {
        "time_period": {
            "start": metrics_request.start_date.isoformat(),
            "end": metrics_request.end_date.isoformat(),
            "grouping": metrics_request.group_by
        },
        "metrics": metrics
    }

@app.get("/api/v1/analytics/quality-metrics", response_model=List[QualityMetricsResponse])
@require_role(['researcher', 'analyst', 'admin'])
async def get_quality_metrics(
    db: Session = Depends(get_db),
    current_user: Dict = Depends(verify_token)
):
    """Get current quality metrics with benchmarks"""
    metrics = []
    
    # Readmission Rate
    total_encounters = db.query(func.count(Encounter.encounter_id)).scalar()
    readmissions = db.query(func.count(Encounter.encounter_id)).filter(
        Encounter.readmission_flag == True
    ).scalar()
    
    readmission_rate = (readmissions / total_encounters * 100) if total_encounters > 0 else 0
    
    metrics.append(QualityMetricsResponse(
        metric_name="30-Day Readmission Rate",
        value=round(readmission_rate, 2),
        benchmark=15.0,  # National benchmark
        status="Good" if readmission_rate < 15.0 else "Needs Improvement",
        trend="stable",
        details={
            "total_encounters": total_encounters,
            "readmissions": readmissions
        }
    ))
    
    # Average Length of Stay
    avg_los = db.query(func.avg(Encounter.length_of_stay)).scalar()
    
    metrics.append(QualityMetricsResponse(
        metric_name="Average Length of Stay",
        value=round(float(avg_los), 1) if avg_los else 0,
        benchmark=4.5,  # National benchmark
        status="Good" if avg_los and avg_los < 4.5 else "Needs Improvement",
        trend="improving",
        details={
            "unit": "days",
            "specialty_adjusted": False
        }
    ))
    
    # Add more metrics as needed...
    
    return metrics

# ==================== Root Cause Analysis Endpoints ====================
@app.post("/api/v1/analytics/rca")
@require_role(['researcher', 'analyst'])
@rate_limit(max_calls=20)
async def run_root_cause_analysis(
    request: Request,
    rca_request: RCARequest,
    db: Session = Depends(get_db),
    current_user: Dict = Depends(verify_token)
):
    """Run root cause analysis for specified metric"""
    # Check cache first
    cache_key = f"rca:{rca_request.analysis_type}:{rca_request.time_period_days}"
    cached_result = redis_client.get(cache_key)
    
    if cached_result:
        return json.loads(cached_result)
    
    # Run analysis based on type
    if rca_request.analysis_type == "readmissions":
        result = app.state.analyzer.analyze_readmissions(app.state.engine)
    else:
        # Implement other analysis types
        raise HTTPException(
            status_code=status.HTTP_501_NOT_IMPLEMENTED,
            detail=f"Analysis type '{rca_request.analysis_type}' not yet implemented"
        )
    
    # Cache result for 1 hour
    redis_client.setex(cache_key, 3600, json.dumps(result, default=str))
    
    # Audit log
    app.state.security_manager.audit_log(
        action="rca_analysis",
        user=current_user['user_id'],
        details={
            "analysis_type": rca_request.analysis_type,
            "parameters": rca_request.dict()
        }
    )
    
    return result

# ==================== Reporting Endpoints ====================
@app.get("/api/v1/reports/generate/{report_type}")
@require_role(['researcher', 'analyst', 'admin'])
async def generate_report(
    report_type: str,
    db: Session = Depends(get_db),
    current_user: Dict = Depends(verify_token)
):
    """Generate specified report type"""
    valid_report_types = ['monthly', 'quarterly', 'annual', 'executive']
    
    if report_type not in valid_report_types:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=f"Invalid report type. Must be one of: {valid_report_types}"
        )
    
    # Generate report
    report_data = app.state.reporting.generate_research_report(
        app.state.engine,
        report_type=report_type
    )
    
    # Audit log
    app.state.security_manager.audit_log(
        action="report_generated",
        user=current_user['user_id'],
        details={"report_type": report_type}
    )
    
    return {
        "report_type": report_type,
        "generated_at": datetime.now().isoformat(),
        "data": report_data
    }

# ==================== Health Check Endpoints ====================
@app.get("/health")
async def health_check():
    """Health check endpoint"""
    health_status = {
        "status": "healthy",
        "timestamp": datetime.now().isoformat(),
        "version": "1.0.0",
        "services": {}
    }
    
    # Check database
    try:
        db = app.state.SessionLocal()
        db.execute("SELECT 1")
        db.close()
        health_status["services"]["database"] = "healthy"
    except Exception as e:
        health_status["services"]["database"] = "unhealthy"
        health_status["status"] = "degraded"
    
    # Check Redis
    try:
        redis_client.ping()
        health_status["services"]["redis"] = "healthy"
    except:
        health_status["services"]["redis"] = "unhealthy"
        health_status["status"] = "degraded"
    
    return health_status

@app.get("/api/v1/status")
@require_role(['admin'])
async def system_status(current_user: Dict = Depends(verify_token)):
    """Get detailed system status (admin only)"""
    # Get database statistics
    db = app.state.SessionLocal()
    
    stats = {
        "database": {
            "total_patients": db.query(func.count(Patient.patient_id)).scalar(),
            "total_encounters": db.query(func.count(Encounter.encounter_id)).scalar(),
            "total_lab_results": db.query(func.count(LabResult.result_id)).scalar()
        },
        "cache": {
            "keys": redis_client.dbsize(),
            "memory_usage": redis_client.info()['used_memory_human']
        },
        "api": {
            "uptime_seconds": time.time() - app.state.get('start_time', time.time()),
            "total_requests": app.state.get('request_count', 0)
        }
    }
    
    db.close()
    return stats

# ==================== Utility Functions ====================
def _calculate_distribution(items: List[Any], attribute: str) -> Dict[str, float]:
    """Calculate percentage distribution of an attribute"""
    total = len(items)
    if total == 0:
        return {}
    
    distribution = {}
    for item in items:
        value = getattr(item, attribute)
        distribution[value] = distribution.get(value, 0) + 1
    
    # Convert to percentages
    for key in distribution:
        distribution[key] = round(distribution[key] / total * 100, 2)
    
    return distribution

def _calculate_age_distribution(patients: List[Patient]) -> Dict[str, float]:
    """Calculate age distribution in ranges"""
    current_year = datetime.now().year
    age_ranges = {
        "0-17": 0,
        "18-34": 0,
        "35-49": 0,
        "50-64": 0,
        "65-79": 0,
        "80+": 0
    }
    
    for patient in patients:
        age = current_year - patient.birth_year
        if age < 18:
            age_ranges["0-17"] += 1
        elif age < 35:
            age_ranges["18-34"] += 1
        elif age < 50:
            age_ranges["35-49"] += 1
        elif age < 65:
            age_ranges["50-64"] += 1
        elif age < 80:
            age_ranges["65-79"] += 1
        else:
            age_ranges["80+"] += 1
    
    total = len(patients)
    if total > 0:
        for key in age_ranges:
            age_ranges[key] = round(age_ranges[key] / total * 100, 2)
    
    return age_ranges

# ==================== Exception Handlers ====================
@app.exception_handler(HTTPException)
async def http_exception_handler(request: Request, exc: HTTPException):
    """Custom HTTP exception handler"""
    return JSONResponse(
        status_code=exc.status_code,
        content={
            "error": {
                "code": exc.status_code,
                "message": exc.detail,
                "timestamp": datetime.now().isoformat()
            }
        }
    )

@app.exception_handler(Exception)
async def general_exception_handler(request: Request, exc: Exception):
    """General exception handler"""
    logger.error(f"Unhandled exception: {str(exc)}", exc_info=True)
    
    return JSONResponse(
        status_code=500,
        content={
            "error": {
                "code": 500,
                "message": "Internal server error",
                "timestamp": datetime.now().isoformat()
            }
        }
    )

# ==================== Startup Event ====================
@app.on_event("startup")
async def startup_event():
    """Initialize application state"""
    app.state.start_time = time.time()
    app.state.request_count = 0
    logger.info("Healthcare API started successfully")

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(
        "main:app",
        host=os.environ.get("API_HOST", "0.0.0.0"),
        port=int(os.environ.get("API_PORT", 8001)),
        reload=os.environ.get("DEBUG", "False").lower() == "true"
    )