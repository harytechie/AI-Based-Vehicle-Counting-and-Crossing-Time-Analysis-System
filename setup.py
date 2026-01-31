"""
Setup and installation script
"""
import subprocess
import sys
from pathlib import Path


def run_command(command, description):
    """Run a command and print status"""
    print(f"\n{'='*60}")
    print(f"📦 {description}")
    print(f"{'='*60}")
    
    try:
        result = subprocess.run(command, shell=True, check=True, 
                              capture_output=True, text=True)
        print(f"✓ {description} completed successfully")
        return True
    except subprocess.CalledProcessError as e:
        print(f"✗ Error: {e}")
        print(f"Output: {e.output}")
        return False


def main():
    """Main setup function"""
    print("""
    ╔══════════════════════════════════════════════════════════╗
    ║  AI-Based Vehicle Counting System - Setup Script        ║
    ╚══════════════════════════════════════════════════════════╝
    """)
    
    # Check Python version
    print(f"Python version: {sys.version}")
    if sys.version_info < (3, 8):
        print("✗ Python 3.8 or higher is required")
        sys.exit(1)
    
    # Create virtual environment
    venv_path = Path("venv")
    if not venv_path.exists():
        if run_command("python -m venv venv", "Creating virtual environment"):
            print("✓ Virtual environment created")
        else:
            print("✗ Failed to create virtual environment")
            sys.exit(1)
    else:
        print("✓ Virtual environment already exists")
    
    # Determine activation script
    if sys.platform == "win32":
        activate_script = "venv\\Scripts\\activate"
        pip_cmd = "venv\\Scripts\\pip"
    else:
        activate_script = "source venv/bin/activate"
        pip_cmd = "venv/bin/pip"
    
    print(f"\n{'='*60}")
    print("📦 Installing dependencies...")
    print(f"{'='*60}")
    
    # Upgrade pip
    run_command(f"{pip_cmd} install --upgrade pip", "Upgrading pip")
    
    # Install requirements
    if run_command(f"{pip_cmd} install -r requirements.txt", 
                   "Installing requirements"):
        print("✓ All dependencies installed successfully")
    else:
        print("✗ Failed to install dependencies")
        print("\nTry running manually:")
        print(f"  {activate_script}")
        print(f"  pip install -r requirements.txt")
        sys.exit(1)
    
    # Create necessary directories
    print(f"\n{'='*60}")
    print("📁 Creating directories...")
    print(f"{'='*60}")
    
    directories = [
        "data/videos",
        "data/models",
        "data/results",
        "logs"
    ]
    
    for directory in directories:
        Path(directory).mkdir(parents=True, exist_ok=True)
        print(f"✓ Created {directory}")
    
    # Print success message
    print(f"\n{'='*60}")
    print("✅ Setup completed successfully!")
    print(f"{'='*60}")
    
    print("""
    🚀 Next Steps:
    
    1. Activate the virtual environment:
       Windows: venv\\Scripts\\activate
       Linux/Mac: source venv/bin/activate
    
    2. Run the application:
       streamlit run ui/app.py
    
    3. Upload a demo video and start analyzing!
    
    📚 For more information, see README.md
    """)


if __name__ == "__main__":
    main()
