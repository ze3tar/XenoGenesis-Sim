.PHONY: dev test demo profile build

dev:
	python -m pip install -e .

test:
	pytest -q

demo:
	bash scripts/run_demo.sh

ui:
	streamlit run src/xenogenesis/ui/live_ui.py

profile:
	bash scripts/profile_ca.sh

build:
	python -m pip wheel . -w dist
