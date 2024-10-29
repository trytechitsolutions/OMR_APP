python3 -m venv venv
source venv/bin/activate
pip install -r requirements.txt


# to identify and kill the existing one
sudo lsof -i :5000
kill <PID>
