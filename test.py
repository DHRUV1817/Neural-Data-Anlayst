#!/usr/bin/env python3
"""
Test script to verify the Neural Data Analyst application works correctly
Run this to check if all dependencies are available and create sample files
"""

import sys
import os
from pathlib import Path

def check_dependencies():
    """Check if all required dependencies are installed"""
    print("🔍 Checking dependencies...")
    
    required_packages = [
        'streamlit',
        'pandas', 
        'numpy',
        'plotly',
        'requests'
    ]
    
    missing_packages = []
    
    for package in required_packages:
        try:
            __import__(package)
            print(f"✅ {package}")
        except ImportError:
            print(f"❌ {package} - MISSING")
            missing_packages.append(package)
    
    # Optional packages
    optional_packages = [
        'scipy',
        'python-dotenv'
    ]
    
    print("\n🔍 Checking optional dependencies...")
    for package in optional_packages:
        try:
            __import__(package.replace('-', '_'))
            print(f"✅ {package} (optional)")
        except ImportError:
            print(f"⚠️  {package} (optional) - not installed")
    
    if missing_packages:
        print(f"\n❌ Missing required packages: {', '.join(missing_packages)}")
        print("Install them with: pip install " + " ".join(missing_packages))
        return False
    else:
        print("\n✅ All required dependencies are installed!")
        return True

def create_sample_files():
    """Create sample configuration files"""
    print("\n📁 Creating sample files...")
    
    # Create .env file template
    env_content = """# Groq API Configuration
# Get your API key from: https://console.groq.com/keys
GROQ_API_KEY=your_groq_api_key_here

# Optional: Set other environment variables
# DEBUG=True
"""
    
    env_file = Path('.env.template')
    if not env_file.exists():
        with open(env_file, 'w') as f:
            f.write(env_content)
        print(f"✅ Created {env_file}")
    else:
        print(f"ℹ️  {env_file} already exists")
    
    # Create sample CSV data
    sample_csv = Path('sample_data.csv')
    if not sample_csv.exists():
        csv_content = """customer_id,customer_name,product,sales_amount,order_date,region,sales_rep
1,Customer_1,Widget A,2147.23,2023-01-01,North,John Smith
2,Customer_2,Widget B,1823.45,2023-01-02,South,Jane Doe
3,Customer_3,Widget C,2456.78,2023-01-03,East,Bob Johnson
4,Customer_4,Gadget X,1934.56,2023-01-04,West,Alice Brown
5,Customer_5,Widget A,2234.67,2023-01-05,North,John Smith
"""
        with open(sample_csv, 'w') as f:
            f.write(csv_content)
        print(f"✅ Created {sample_csv}")
    else:
        print(f"ℹ️  {sample_csv} already exists")

def create_required_modules():
    """Create the required module files if they don't exist"""
    print("\n📝 Checking required modules...")
    
    # Check if eda_analyzer.py exists
    if not Path('eda_analyzer.py').exists():
        print("❌ eda_analyzer.py not found!")
        print("   Please save the EDA Analyzer code as 'eda_analyzer.py'")
        return False
    else:
        print("✅ eda_analyzer.py found")
    
    # Check if database_manager.py exists
    if not Path('database_manager.py').exists():
        print("❌ database_manager.py not found!")
        print("   Please save the Database Manager code as 'database_manager.py'")
        return False
    else:
        print("✅ database_manager.py found")
    
    return True

def test_imports():
    """Test if the modules can be imported"""
    print("\n🧪 Testing module imports...")
    
    try:
        from eda_analyzer import EDAAnalyzer
        print("✅ EDAAnalyzer imported successfully")
    except Exception as e:
        print(f"❌ Failed to import EDAAnalyzer: {e}")
        return False
    
    try:
        from database_manager import DatabaseManager
        print("✅ DatabaseManager imported successfully")
    except Exception as e:
        print(f"❌ Failed to import DatabaseManager: {e}")
        return False
    
    return True

def main():
    """Main test function"""
    print("🚀 Neural Data Analyst - Setup Test")
    print("=" * 50)
    
    # Check Python version
    python_version = sys.version_info
    print(f"🐍 Python version: {python_version.major}.{python_version.minor}.{python_version.micro}")
    
    if python_version < (3, 7):
        print("❌ Python 3.7+ required!")
        return False
    else:
        print("✅ Python version OK")
    
    # Run all checks
    deps_ok = check_dependencies()
    if not deps_ok:
        return False
    
    create_sample_files()
    
    modules_ok = create_required_modules()
    if not modules_ok:
        return False
    
    imports_ok = test_imports()
    if not imports_ok:
        return False
    
    print("\n🎉 Setup test completed successfully!")
    print("\n📋 Next steps:")
    print("1. Copy .env.template to .env and add your Groq API key (optional)")
    print("2. Run: streamlit run app.py")
    print("3. Upload sample_data.csv or your own data file")
    print("4. Explore the analysis features!")
    
    return True

if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)