from setuptools import setup

setup(
    name='spring_amr',
    version='1.0',
    py_modules=['penman', 'tokenization_bart', 'utils', 'linearization', 'modeling_bart', 
                'optim', 'postprocessing', 'IO', 'dataset', 'entities', 'evaluation'],
    url='https://github.com/SapienzaNLP/spring',
    license='CC BY-NC-SA 4.0',
    author='Michele Bevilacqua, Rexhina Blloshmi and Roberto Navigli',
    author_email='{bevilacqua,blloshmi,navigli}@di.uniroma1.it',
    description='Parse sentences into AMR graphs and generate sentences from AMR graphs without breaking a sweat!',
    install_requires=[
        "torch",
        "transformers",
        "penman",
    ],
    python_requires=">=3.8",
) 
