# Contributing to Healthcare Research Data Pipeline

Thank you for your interest in contributing to the Healthcare Research Data Pipeline! This document provides guidelines and instructions for contributing to this project.

## Table of Contents

- [Code of Conduct](#code-of-conduct)
- [Getting Started](#getting-started)
- [Development Setup](#development-setup)
- [How to Contribute](#how-to-contribute)
- [Pull Request Process](#pull-request-process)
- [Coding Standards](#coding-standards)
- [Security Guidelines](#security-guidelines)
- [Testing Guidelines](#testing-guidelines)
- [Documentation](#documentation)
- [Community](#community)

## Code of Conduct

By participating in this project, you agree to abide by our Code of Conduct:

- **Be respectful**: Treat everyone with respect and kindness
- **Be collaborative**: Work together to solve problems
- **Be inclusive**: Welcome newcomers and help them get started
- **Be professional**: Maintain professional conduct in all interactions

## Getting Started

1. **Fork the repository** on GitHub
2. **Clone your fork** locally:
   ```bash
   git clone https://github.com/your-username/healthcare-research-data-pipeline.git
   cd healthcare-research-data-pipeline
   ```
3. **Add the upstream remote**:
   ```bash
   git remote add upstream https://github.com/original-repo/healthcare-research-data-pipeline.git
   ```

## Development Setup

### Prerequisites

- Python 3.9+
- PostgreSQL 13+
- Redis 6+
- Docker and Docker Compose (optional)
- Git

### Local Environment Setup

1. **Create a virtual environment**:
   ```bash
   python -m venv venv
   source venv/bin/activate  # On Windows: venv\Scripts\activate
   ```

2. **Install dependencies**:
   ```bash
   pip install -r requirements.txt
   pip install -r requirements-dev.txt
   ```

3. **Set up environment variables**:
   ```bash
   cp .env.example .env
   # Edit .env with your local configuration
   ```

4. **Initialize the database**:
   ```bash
   python scripts/init_db.py
   python scripts/migrate.py
   ```

5. **Run tests to verify setup**:
   ```bash
   pytest tests/
   ```

### Docker Setup

For a containerized development environment:

```bash
docker-compose -f docker-compose.dev.yml up -d
```

## How to Contribute

### Reporting Issues

- Check if the issue already exists
- Use the issue templates provided
- Include detailed steps to reproduce
- Provide system information and error logs
- For security issues, see [SECURITY.md](SECURITY.md)

### Suggesting Features

- Check the roadmap and existing feature requests
- Open a discussion in the GitHub Discussions
- Use the feature request template
- Explain the use case and benefits

### Contributing Code

1. **Create a feature branch**:
   ```bash
   git checkout -b feature/your-feature-name
   ```

2. **Make your changes**:
   - Write clean, documented code
   - Follow the coding standards
   - Add tests for new functionality
   - Update documentation as needed

3. **Commit your changes**:
   ```bash
   git add .
   git commit -m "feat: add new feature description"
   ```
   
   Follow [Conventional Commits](https://www.conventionalcommits.org/):
   - `feat:` new feature
   - `fix:` bug fix
   - `docs:` documentation changes
   - `style:` formatting changes
   - `refactor:` code restructuring
   - `test:` test additions/changes
   - `chore:` maintenance tasks

4. **Push to your fork**:
   ```bash
   git push origin feature/your-feature-name
   ```

## Pull Request Process

1. **Update your branch** with the latest upstream changes:
   ```bash
   git fetch upstream
   git rebase upstream/main
   ```

2. **Run all checks locally**:
   ```bash
   # Format code
   black src/ tests/
   
   # Run linters
   flake8 src/ tests/
   mypy src/
   pylint src/
   
   # Run security checks
   bandit -r src/
   safety check
   
   # Run tests
   pytest tests/ -v --cov=src
   ```

3. **Create a Pull Request**:
   - Use a clear, descriptive title
   - Reference any related issues
   - Describe what changes you made and why
   - Include screenshots for UI changes
   - Ensure all CI checks pass

4. **Code Review**:
   - Respond to feedback promptly
   - Make requested changes
   - Be open to suggestions
   - Thank reviewers for their time

## Coding Standards

### Python Style Guide

- Follow [PEP 8](https://www.python.org/dev/peps/pep-0008/)
- Use [Black](https://github.com/psf/black) for formatting
- Maximum line length: 100 characters
- Use type hints for all functions
- Document all public APIs

### Code Organization

```python
# Good example
from typing import List, Optional
from dataclasses import dataclass

@dataclass
class PatientRecord:
    """Represents a patient record with PHI removed."""
    patient_id: int
    age_group: str
    gender: Optional[str] = None
    
    def is_valid(self) -> bool:
        """Check if the patient record is valid."""
        return self.patient_id > 0 and self.age_group != ""
```

### SQL Standards

- Use uppercase for SQL keywords
- Indent complex queries properly
- Add comments for complex logic
- Use CTEs for readability

```sql
-- Good example
WITH PatientCohort AS (
    SELECT 
        patient_id,
        diagnosis_code,
        encounter_date
    FROM fact_encounters
    WHERE encounter_date >= '2024-01-01'
)
SELECT 
    diagnosis_code,
    COUNT(DISTINCT patient_id) AS patient_count
FROM PatientCohort
GROUP BY diagnosis_code
ORDER BY patient_count DESC;
```

## Security Guidelines

### HIPAA Compliance

- Never commit real patient data
- Always use encryption for sensitive fields
- Implement proper access controls
- Maintain audit logs for all data access

### Security Best Practices

- Sanitize all user inputs
- Use parameterized queries
- Keep dependencies updated
- Run security scans regularly
- Follow the principle of least privilege

### Handling Sensitive Data

```python
# Good example
def store_patient_data(mrn: str, data: dict) -> None:
    """Store patient data with proper security measures."""
    # Hash the MRN
    hashed_mrn = security_manager.hash_pii(mrn)
    
    # Encrypt sensitive fields
    if 'notes' in data:
        data['notes'] = security_manager.encrypt_data(data['notes'])
    
    # Audit log
    security_manager.audit_log(
        action="data_storage",
        user=current_user,
        details={"patient_hash": hashed_mrn}
    )
```

## Testing Guidelines

### Test Structure

```
tests/
├── unit/           # Unit tests
├── integration/    # Integration tests
├── e2e/           # End-to-end tests
└── fixtures/      # Test data and fixtures
```

### Writing Tests

```python
# Good test example
import pytest
from unittest.mock import Mock

def test_patient_transformation(security_manager, sample_patient_data):
    """Test that patient data is properly transformed."""
    # Arrange
    etl = ETLPipeline(security_manager)
    
    # Act
    result = etl.transform_patient_data(sample_patient_data)
    
    # Assert
    assert 'birth_year' in result.columns
    assert 'birth_date' not in result.columns
    assert all(result['gender'].isin(['Male', 'Female', 'Other', 'Unknown']))
```

### Test Coverage

- Maintain minimum 80% code coverage
- Test edge cases and error conditions
- Include performance tests for critical paths
- Test security controls

## Documentation

### Code Documentation

- Use docstrings for all public functions
- Include parameter descriptions
- Provide usage examples
- Document return values and exceptions

```python
def analyze_readmissions(
    patient_cohort: List[int], 
    time_window: int = 30
) -> Dict[str, Any]:
    """
    Analyze readmission patterns for a patient cohort.
    
    Args:
        patient_cohort: List of patient IDs to analyze
        time_window: Days to consider for readmission (default: 30)
    
    Returns:
        Dictionary containing:
        - readmission_rate: Percentage of patients readmitted
        - avg_time_to_readmit: Average days until readmission
        - risk_factors: Top factors associated with readmission
    
    Raises:
        ValueError: If time_window is not positive
        DatabaseError: If query execution fails
    
    Example:
        >>> results = analyze_readmissions([1, 2, 3], time_window=30)
        >>> print(f"Readmission rate: {results['readmission_rate']}%")
    """
```

### API Documentation

- Keep API docs up to date
- Include request/response examples
- Document error codes
- Provide authentication details

## Community

### Getting Help

- Check the documentation first
- Search existing issues
- Ask in GitHub Discussions
- Join our Slack channel (if available)

### Staying Updated

- Watch the repository for updates
- Subscribe to the mailing list
- Follow our blog for announcements
- Attend community meetings

## Recognition

Contributors will be recognized in the following ways:
- Listed in CONTRIBUTORS.md
- Mentioned in release notes
- Given credit in documentation
- Eligible for contributor badges

Thank you for contributing to making healthcare data more accessible and secure! 🏥💻