set -eux

pip install -r requirements-to-freeze.txt
pip freeze > requirements.txt
