import asyncio
import argparse
import json
import sys
from .models import PatientCase
from .orchestrator import ChiefMedicalOfficer
from .agent import MedicalAgent
from .logger import logger

def main():
    parser = argparse.ArgumentParser(description="Aegis Enterprise CLI: Run autonomous medical board.")
    parser.add_argument('--case', type=str, required=True, help="Path to JSON file containing patient case (FHIR-aligned).")
    parser.add_argument('--rounds', type=int, default=1, help="Number of debate iterations.")
    args = parser.parse_args()

    try:
        with open(args.case, 'r') as f:
            case_data = json.load(f)
        
        patient_case = PatientCase(**case_data)
    except Exception as e:
        logger.error(f"Failed to parse clinical data: {e}")
        sys.exit(1)

    dr_house = MedicalAgent("Dr. House", "Diagnostician", "Focus on rare zebras.""local")
    dr_chase = MedicalAgent("Dr. Chase", "Intensive Care", "Focus on survival.""local")
    dr_cameron = MedicalAgent("Dr. Cameron", "Toxicologist", "Focus on rare tox cascades.""local")
    
    cmo = ChiefMedicalOfficer([dr_house, dr_chase, dr_cameron])

    async def run():
        try:
            consensus = await cmo.run_board(patient_case, rounds=args.rounds)
            print("\n" + "="*50)
            print("FINAL MDT CONSENSUS PAYLOAD")
            print("="*50)
            print(consensus.json(indent=4))
        finally:
            await cmo.shutdown()

    asyncio.run(run())

if __name__ == "__main__":
    main()
