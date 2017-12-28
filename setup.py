from setuptools import setup

setup( 
    name='face',
    
    version='0.12',
    description='Face detection and recognition',
    url='http://demo.vedalabs.in/',

    # Author details    
    author='Atinderpal Singh',
    author_email='atinderpalap@gmail.com',
    
    license='Commercial',

    packages=['face'],
    package_data={
        'face':['models/*'],
    },
    
    install_requires=['dlib', 'numpy'],
    zip_safe=False
    )