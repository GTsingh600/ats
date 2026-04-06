#!/bin/bash
# Quick setup and test script for Groq inference

set -e

echo "🚀 Groq Inference Setup & Testing"
echo "=================================="
echo ""

# Check if GROQ_API_KEY is set
if [ -z "$GROQ_API_KEY" ]; then
    echo "❌ GROQ_API_KEY not set!"
    echo ""
    echo "Please set your Groq API key:"
    echo "  export GROQ_API_KEY='your_key_here'"
    echo ""
    echo "Then run this script again:"
    echo "  bash setup_groq_inference.sh"
    exit 1
fi

echo "✅ GROQ_API_KEY detected"
echo ""

# Check Python installation
echo "📦 Checking Python environment..."
python --version
echo "✅ Python available"
echo ""

# Check if groq package is installed
echo "📦 Checking groq package..."
python -c "import groq; print(f'✅ groq {groq.__version__} installed')" || {
    echo "❌ groq not installed"
    echo "Installing..."
    pip install groq
}
echo ""

# Check inference.py
echo "🔍 Checking inference.py..."
python -m py_compile inference.py
echo "✅ inference.py syntax valid"
echo ""

# Run test script
echo "🧪 Running inference tests..."
echo "=================================="
python test_groq_inference.py
echo ""
echo "✅ Setup complete!"
echo ""
echo "📝 Next steps:"
echo "  1. Review test results above"
echo "  2. Run full tests: pytest tests/ -v"
echo "  3. Start inference: python -m inference"
echo ""
