#!/bin/bash


# Seteo explícito del PYTHONPATH y llamada a uvicorn
PYTHONPATH=.. uvicorn credit_risk_analysis.modeling.api.router:app --reload --port 8000

