from setuptools import setup, find_packages

def get_requirements()->list[str]:
    '''
        Return all the require package that are specified in requirements.txt
    '''
    req = []
    with open('requirements.txt') as file_obj:
        req = file_obj.readlines()
        req = [r.replace('\n', '') for r in req]

        END_OF_FILE = '-e .'
        if END_OF_FILE in req:
            req.remove(END_OF_FILE)

        return req

setup(
    name='ML-Project',
    version='0.0.1',
    author='Kongkham Luangkhot',
    author_email='kongkham.luangkhot@gmail.com',
    packages=find_packages(),
    requires=get_requirements()
)