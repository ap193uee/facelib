from setuptools import setup

setup( 
    name='face',
    
    version='0.1',
    description='Face detection and recognition',
    url='http://demo.vedalabs.in/',

    # Author details    
    author='Atinderpal Singh',
    author_email='atinderpalap@gmail.com',
    
    license='Commercial',

    packages=['face'],
    
    install_requires=['dlib'],
    zip_safe=False
    )