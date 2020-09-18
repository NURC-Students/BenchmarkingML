# BenchmarkingML 

[![N|Solid](https://cldup.com/dTxpPi9lDf.thumb.png)](https://nodesource.com/products/nsolid)

[![Build Status](https://travis-ci.org/joemccann/dillinger.svg?branch=master)](https://travis-ci.org/joemccann/dillinger)

## Algorithm
1. Multiple Linear Regression

### IDE 

* BenchmarkingML Code built and run using Visual Studio Code 

### Prerequisites and Running on Discovery Cluster


Proceed? >> yes


1. Download the latest release of Miniconda
    ```sh
    $ wget https://repo.anaconda.com/miniconda/Miniconda2-latest-Linux-x86_64.sh
    ```

2. Change the permissions of the installation script
    ```sh
    $ chmod +x Miniconda2-latest-Linux-x86_64.sh
    ```

3. Run the installation script to install Miniconda 2 in your chose destination _(eg. /work/rc/s.chakravarty)_
    ```sh
    $ ./Miniconda2-latest-Linux-x86_64.sh
    
    Agree to license agreement >> yes
    
    Directory to install >> /work/rc/s.chakravarty
    ```
    > :warning: **Note**: Make sure to deactivate any previously activated conda environments with _conda deactivate_

4. Change into the binary folder of your conda installation
    ```sh
    $ cd /work/rc/s.chakravarty/bin
    ```

5. Activate your base miniconda environment
    ```sh
    $ source activate
    ```

6. Update all your conda packages
    ```sh
    $ conda update conda    
    ```
