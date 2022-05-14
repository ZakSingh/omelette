activate=. venv/bin/activate

build: venv
	$(activate) && maturin build --release

dev: venv
	$(activate) && maturin develop && python main.py

clean:
	cargo clean