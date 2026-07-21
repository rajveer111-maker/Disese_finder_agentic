#!/usr/bin/env python3
"""
Quick demo runner for Agentic Disease Finder
"""

import subprocess
import sys
import os

def check_requirements():
    """Check if required packages are installed."""
    try:
        import streamlit
        import tensorflow
        import numpy
        import pandas
        import plotly
        print("✅ All required packages are installed")
        return True
    except ImportError as e:
        print(f"❌ Missing package: {e}")
        print("Please run: pip install -r requirements.txt")
        return False

def run_demo():
    """Run the demo script."""
    print("🧠 Running Agentic Disease Finder Demo...")
    try:
        subprocess.run([sys.executable, "demo.py"], check=True)
    except subprocess.CalledProcessError as e:
        print(f"❌ Demo failed: {e}")
        return False
    return True

def run_streamlit():
    """Run the Streamlit app."""
    print("🚀 Starting Streamlit app...")
    print("   Open your browser to: http://localhost:8501")
    try:
        subprocess.run(["streamlit", "run", "app.py"], check=True)
    except subprocess.CalledProcessError as e:
        print(f"❌ Streamlit failed: {e}")
        return False
    except KeyboardInterrupt:
        print("\n👋 Streamlit app stopped")
        return True
    return True

def main():
    """Main function."""
    print("🧠 Agentic Disease Finder")
    print("=" * 40)
    
    # Check if we're in the right directory
    if not os.path.exists("app.py"):
        print("❌ Please run this script from the project root directory")
        return
    
    # Check requirements
    if not check_requirements():
        return
    
    # Ask user what to do
    print("\nWhat would you like to do?")
    print("1. Run demo script (offline)")
    print("2. Start Streamlit web app")
    print("3. Both")
    
    choice = input("\nEnter your choice (1-3): ").strip()
    
    if choice == "1":
        run_demo()
    elif choice == "2":
        run_streamlit()
    elif choice == "3":
        run_demo()
        print("\n" + "="*40)
        run_streamlit()
    else:
        print("❌ Invalid choice")

if __name__ == "__main__":
    main()

