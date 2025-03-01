#!/bin/bash

# FracTime App Launcher Script

echo "◆◆◆ FracTime App Launcher ◆◆◆"
echo "-----------------------------"

# Check if Home.py exists
if [ ! -f "Home.py" ]; then
    echo "Error: Home.py not found in current directory!"
    echo "Please run this script from the FracTime project root directory."
    exit 1
fi

# Check if streamlit is installed
if ! command -v streamlit &> /dev/null; then
    echo "Error: Streamlit not found!"
    echo "Please install streamlit before running this app:"
    echo "pip install streamlit"
    exit 1
fi

# Activate virtual environment if it exists
if [ -d ".venv" ]; then
    echo "✅ Activating virtual environment..."
    source .venv/bin/activate
elif [ -d "venv" ]; then
    echo "✅ Activating virtual environment..."
    source venv/bin/activate
fi

# Ensure streamlit directory exists
if [ ! -d ".streamlit" ]; then
    echo "✅ Creating .streamlit directory..."
    mkdir -p .streamlit
fi

# Ensure config.toml exists
if [ ! -f ".streamlit/config.toml" ]; then
    echo "✅ Creating default config.toml..."
    cat > .streamlit/config.toml << EOL
[server]
enableCORS = false
enableXsrfProtection = false

[browser]
gatherUsageStats = false

[theme]
primaryColor = "#512da8"
backgroundColor = "#ffffff"
secondaryBackgroundColor = "#f0f2f6"
textColor = "#262730"
font = "sans serif"
EOL
fi

# Run the app with the Home.py file
echo "✅ Starting FracTime application..."
echo "-----------------------------"
echo "Access the app at: http://localhost:8501"
echo "Press Ctrl+C to stop the application"
echo "-----------------------------"

# Try to run with standard settings first
streamlit run Home.py

# If that fails, try with headless mode
if [ $? -ne 0 ]; then
    echo "⚠️ Standard mode failed, trying headless mode..."
    streamlit run Home.py --server.headless=true
fi

# If still fails, try with different port
if [ $? -ne 0 ]; then
    echo "⚠️ Trying with alternative port..."
    streamlit run Home.py --server.port=8502
fi