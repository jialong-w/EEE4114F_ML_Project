install: venv
	. venv/bin/activate; pip3 install -Ur requirements.txt

venv:
	test -d venv || python3 -m venv venv

run:
	. venv/bin/activate; python3 PaintShape.py

clean:
	rm -rf venv
	find . -iname "*.pyc" -delete
