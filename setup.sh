echo "Creating virtual environment"
python3 -m venv .venv --clear
source venv/bin/activate

echo "Installing requirements"
pip install --upgrade --no-cache-dir -r requirements.txt

echo "Done"