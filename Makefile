.PHONY: dev test demo profile build

dev:
python -m pip install -e .

test:
pytest -q

demo:
bash scripts/run_demo.sh

profile:
bash scripts/profile_ca.sh

build:
python -m pip wheel . -w dist
