import asyncio
import argparse
from .models import PatientCase
from .orchestrator import ChiefMedicalOfficer
from .agent import MedicalAgent
import json
import logging

logging.basicConfig(level=logging.INFO, format='%(message)s')

def main():
    parser = argparse.ArgumentParser(description="Aegis CLI: Run an autonomous medical board from the terminal.")
    parser.add_argument('--case', type=str, required=True, help="Path to JSON file containing patient case.")
    args = parser.parse_args()

    with open(args.case, 'r') as f:
        case_data = json.load(f)
    
    patient_case = PatientCase(**case_data)

    dr_house = MedicalAgent("Dr. House", "Diagnostician", "Focus on rare zebras.", "local")
    dr_chase = MedicalAgent("Dr. Chase", "Intensive Care", "Focus on survival.", "local")
    dr_cameron = MedicalAgent("Dr. Cameron", "Toxicologist", "Focus on rare tox cascades.", "local")
    
    cmo = ChiefMedicalOfficer([dr_house, dr_chase, dr_cameron])

    asyncio.run(cmo.run_board(patient_case))

if __name__ == "__main__":
    main()
