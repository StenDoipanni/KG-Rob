Tested with python 3.8.20. We recommend to manage your python instalation with some tool like uv or conda.

Steps to run:

1. Install uv using:
```bash
$ curl -LsSf https://astral.sh/uv/install.sh | sh
```
2. Install the python version:
```bash
$ uv python install 3.8.20
```
3. Create and source a virtual environment:
```bash
$ uv venv -p 3.8.20 && source .venv/bin/activate
```
4. Install the required packages:
```bash
$ uv pip install -r requirements.txt
```
5. Install the `spring_amr`
```bash
cd spring_amr && uv pip install -e . && cd ..
```

6. Download and unpack the AMR thingy
TODO FIX ME!!!

7. run  stuff?
