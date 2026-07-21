import os
import pandas as pd
import numpy as np
from models.agentic_decision import AgenticDecisionSystem

def test_agentic_system():
    agent = AgenticDecisionSystem()
    sample_dir = 'e:/Shiv Projects/AgenticDiseaseFinder/sample_data'
    
    files = [f for f in os.listdir(sample_dir) if f.endswith('.csv')]
    
    print(f"{'File':<35} | {'Recommended Model':<15} | {'Confidence':<10}")
    print("-" * 65)
    
    for f in files:
        filepath = os.path.join(sample_dir, f)
        df = pd.read_csv(filepath)
        if 'Time' in df.columns:
            data = df.drop(columns=['Time']).values
        else:
            data = df.values
            
        decision = agent.decide_model(data=data, file_type='csv')
        model = decision['selected_model']
        confidence = decision['confidence']
        print(f"{f:<35} | {model:<15} | {confidence:.2f}")

if __name__ == '__main__':
    test_agentic_system()
