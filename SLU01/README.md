# Small Learning Unit 1 - Pandas 101


Welcome to the first SLU.

The first thing you need to do is to setup your local environment, so that you can run the all the notebooks in the SLU.

With your setup in place, you should go through the notebooks in this order:
* `Learning Notebook`: here we'll teach you all the basics of [Pandas](https://pandas.pydata.org/docs/).
* `Examples Notebook`: here you'll find a summary of the methods that you've learned in the `Learning Notebook`.
* `Exercise Notebook`: here you'll find the exercises that you should complete and submit.

Once you're done with the three notebooks, you should submit your `Exercise Notebook` for grading, in the portal.

Keep reading to learn how to setup your local environment and how to submit your assignment.


### Setup your local environment

In order for you to go through the notebooks, you need to:
* Have Python 3 installed in your machine.
* Create a python virtual environment.
* Install the requirements listed in the `requirements.txt`, in your virtual env (note: it is paramount that the packages that you install have the exact same version as in the `requirements.txt` file, because those are the versions that we're using to grade your notebooks).
* Be able open a Jypyter Notebook and run code there.

If you know what you're doing, the above instructions should be enough :p

Otherwise, we have a guide that explains how to set up a Python environment and install requirements [here](https://github.com/LDSSA/ds-prep-course#initial-setup). The parts that are sufficient and necessary for you to setup your local environment are:
* Setup
    * [Windows Setup](https://github.com/LDSSA/ds-prep-course#Windows-Setup)
    * [MacOS Setup](https://github.com/LDSSA/ds-prep-course#macos-setup)
* [Creating Python Virtual Environment and installing the necessary packages](https://github.com/LDSSA/ds-prep-course#creating-python-virtual-environment-and-installing-the-necessary-packages)
* [Working on the Learning Unit](https://github.com/LDSSA/ds-prep-course#working-on-the-learning-unit) - this part is about launching a Jupyter Notebook
* [The Exercise Notebook](https://github.com/LDSSA/ds-prep-course#the-exercise-notebook) - this part is about running code in a a Jupyter Notebook

Alternatively, you can achieve the same by using Anaconda:
* [installing Anaconda](https://docs.anaconda.com/anaconda/install/)
* [launching a Jupyter Notebook](https://docs.anaconda.com/anaconda/user-guide/getting-started/#run-python-in-a-jupyter-notebook)
* [installing a Python package on Anaconda](https://docs.anaconda.com/anaconda/user-guide/tasks/install-packages/)


### Submitting your work for grading

You've downloaded a zip file with this SLU from the portal [here](https://admissions.lisbondatascience.org/candidate/slu/slu01).

In order to pass this SLU, you have to solve the exercises from the `Exercise Notebook`, submit them for grading and get a score of at least 16 (out of 20).

> In the submissions file uploader, you should upload **only the `Exercise Notebook` with your solutions**, you don't need to create a zip file or anything else!

Example:

![](assets/submission.png)


After you submit, you'll see the grade of your submission, and whether you passed or not.
You can click in "Open" in the feedback column to see how much did you score in each question.

For SLUs, you can make as many submissions as you like until the due date. So take your time to work on this assignment :)

---

### New concepts in this unit

- Importing Pandas
- Pandas Series
- Pandas DataFrame
- Previewing a DataFrame
- Retrieving Dataframe Information
- Reading/Writing data 

### New tools in this unit

- [Series](https://pandas.pydata.org/pandas-docs/stable/generated/pandas.Series.html)
- [DataFrame](https://pandas.pydata.org/pandas-docs/stable/generated/pandas.DataFrame.html)
- [read_csv](https://pandas.pydata.org/pandas-docs/stable/generated/pandas.read_csv.html)
- [head](https://pandas.pydata.org/pandas-docs/stable/generated/pandas.DataFrame.head.html)
- [tail](https://pandas.pydata.org/pandas-docs/stable/generated/pandas.DataFrame.tail.html)
- [info](https://pandas.pydata.org/pandas-docs/stable/generated/pandas.DataFrame.info.html)
- [describe](https://pandas.pydata.org/pandas-docs/stable/generated/pandas.DataFrame.describe.html)
- [shape](https://pandas.pydata.org/pandas-docs/stable/generated/pandas.DataFrame.shape.html)
