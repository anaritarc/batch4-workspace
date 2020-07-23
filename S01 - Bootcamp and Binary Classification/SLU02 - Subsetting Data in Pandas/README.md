# Small Learning Unit 2 - Subsetting data in Pandas

Welcome to SLU02.

The first thing you need to do is to setup your local environment, so that you can run the all the notebooks in the SLU.

With your setup in place, you should go through the notebooks in this order:
* `Learning Notebook`: here we'll teach you all the basics of [Pandas](https://pandas.pydata.org/docs/).
* `Examples Notebook`: here you'll find a summary of the methods that you've learned in the `Learning Notebook`.
* `Exercise Notebook`: here you'll find the exercises that you should complete and submit.

Once you're done with the three notebooks, you should submit your `Exercise Notebook` for grading, in the portal.


### Setup your local environment

If you've already done SLU01, then you can use the same environment for this SLU.

Otherwise, you need to:
* Have Python 3 installed in your machine.
* Create a python virtual environment.
* Install the requirements listed in the `requirements.txt`, in your virtual env (note: it is paramount that the packages that you install have the exact same version as in the `requirements.txt` file, because those are the versions that we're using to grade your notebooks).
* Be able open a Jypyter Notebook and run code there.

If you need more detailed instructions than these, please check the README in SLU01.


### Submitting your work for grading

You've downloaded a zip file with SLU02 from the portal [here](https://admissions.lisbondatascience.org/candidate/slu/slu02).

In order to pass this SLU, you have to solve the exercises from the `Exercise Notebook`, submit them for grading and get a score of at least 16 (out of 20).

> In the submissions file uploader, you should upload **only the notebook with your solutions**, you don't need to create a zip file or anything else!

After you submit, you'll see the grade of your submission, and whether you passed or not.
You can click in "Open" in the feedback column to see how much did you score in each question.

For SLUs, you can make as many submissions as you like until the due date. So take your time to work on this assignment :)

---

### New concepts in this unit
- Setting pandas Dataframe index
- Selecting columns (brackets and dot notation)
- Selecting rows (loc and iloc)
- Chain indexing (not good) vs Multi-axis indexing (good)
- Masks
- Where
- Subsetting on conditions
- Removing columns and rows
- Adding columns and rows


### New tools in this unit
- [set_index](https://pandas.pydata.org/pandas-docs/stable/reference/api/pandas.DataFrame.set_index.html)
- [reset_index](https://pandas.pydata.org/pandas-docs/stable/generated/pandas.DataFrame.reset_index.html)
- [sort_index](https://pandas.pydata.org/pandas-docs/stable/generated/pandas.DataFrame.sort_index.html)
- [loc](https://pandas.pydata.org/pandas-docs/stable/reference/api/pandas.DataFrame.loc.html)
- [iloc](https://pandas.pydata.org/pandas-docs/stable/reference/api/pandas.DataFrame.iloc.html)
- [mask](https://pandas.pydata.org/pandas-docs/stable/reference/api/pandas.DataFrame.mask.html)
- [where](https://pandas.pydata.org/pandas-docs/stable/reference/api/pandas.DataFrame.where.html)
- [drop](https://pandas.pydata.org/pandas-docs/stable/reference/api/pandas.DataFrame.drop.html)
