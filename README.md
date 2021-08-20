# PyDAAL Benchmarking.

## A quick introduction:

The aim of the project is to compare the performace of intel's oneAPI Data Analytics Library (oneDAL). We test the python flavor available to us called pyDAAL.

It is a collection of optimized ML algorithms and related techniques available in popular libraries like scikit-learn. In fact it provides a flavor of 
scikit learn as shown [here](https://github.com/intel/scikit-learn-intelex) which is optimized to run in intel's CPUs much efficiently underneath the hood
it makes use of oneDAL.

There are three possible flavors of compute possible and it depends on the compute available at hand. They are:

- Batch processing
- Stream processing
- Distributed processing

> Not all flavors are available for each each algorithm / toolkit available in the framework. Examples can be found [here](https://software.intel.com/content/www/us/en/develop/articles/a-daal4py-introduction-and-getting-started-guide.html)

## What do you need to do before starting to benchmark ?

Get a reservation, so that you always have the same intel CPU each time to test on the cluster.

Run the following commands:

`module load intel/python-daal4py-2021.3.0`

Install the following support libraries:

`pip install --upgrade dpcpp_cpp_rt`

`pip install --upgrade impi_rt`

## What does this notebook and branch contain ?

To benchmark this library we need to test the performance improvements available to us, by using pyDAAL over scikit learn. Hence we use a large dataset and common algorithms
available in both sci-kit learn and pyDAAL. The dataset is microsoft malware dataset (BIG) which is abour have a terabyte of data uncompressed. The dataset and the 
details would be available [here](https://www.kaggle.com/c/malware-classification/data).

The work that has been done so far is to build machine learning models using just a portion of the data available to us as byte files. There are another portion called the asm files.

So all the models have been built with just vanilla scikit-learn.

The asm files have been pre-processed using vanilla multiprocessing library available in python. But we are yet to build models with this.

To understand the work in-depth please read through report available in the repo.

The entire work can be found in `/work/skunkworks/prithivirajan.a/malwareBig/` folder.

The script daalTest.py contains an implementation which needs some bug fixes to get logistic regression version of daal4py working.




