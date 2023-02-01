# Ch00 - Pandas Exercises

<img src="../../assets/workout.png"
     alt="Hierarchical clustering"
     style="width: 100px; float: right; margin: 20px" />

__Welcome to the practice exercises for chapter 00!__

The exercises are divided into three parts, gradually increasing in difficulty. Are you up to the challenge? 

The three exercises uses slightly different techniques, and different datasets.

It is recommended to use what you know and the listed resources to complete as many exercises as possible before asking for assistance.

__Enjoy!__

---

**Tips and tricks:**
<ul>
    <li/> Pandas documentation: <a>https://pandas.pydata.org/docs/user_guide/index.html#user-guide</a>
    <li/> Pandas and data-wrangling cook-book: <a>https://chrisalbon.com/</a>
</ul>

**Content:**

* [Basic exercises](#basic)
* [Medium exercises](#medium)
* [Challenging exercises](#challenge)

---

<img src="../images/beginner.png"
     alt="Hierarchical clustering"
     style="width: 75px; float: right; margin: 20px" />


## Level: Basic<a class="anchor" id="basic"></a>

__Task 1:__

Import data from this website
https://raw.githubusercontent.com/justmarkham/DAT8/master/data/u.user and assign it to a dataframe called `users`.


```python
import pandas as pd
import numpy as np

# Enter your code below
# =====================

```

__Task 2:__

Explore the data.
<ol>
    <li/> Display (print on screen) the first 25 entries (rows)
    <li/> Display (print on screen) the last 10 entires (rows)
    <li/> Find the number of rows and columns in dataset
</ol>


```python
# Enter your code below
# =====================

```

__Task 3:__

Explore columns and rows
<ol>
    <li/> Print names of columns
    <li/> Print names of rows (index)
    <li/> Find data type of each column
</ol>


```python
# Enter your code below
# =====================

```

__Task 4:__

Explore a single column
<ol>
    <li/> Print the 'occupation' column
    <li/> Find the number of different occupations in the dataset
    <li/> What is the most frequent occupation?
</ol>


```python
# Enter your code below
# =====================

```

__Task 5:__

Let's summarize!
<ol>
    <li/> Summarise the dataframe
    <li/> Summarize the dataframe column by column in a loop. What information can you see here that is lacking in the previous task?
</ol>


```python
# Enter your code below
# =====================

```

__Task 6:__

Explore the age variable
<ol>
    <li/> What is the mean age of users?
    <li/> What is/are the age(s) with least occurence(s)?
</ol>


```python
# Enter your code below
# =====================

```

---

<img src="../images/medium.png"
     alt="Hierarchical clustering"
     style="width: 75px; float: right; margin: 20px" />

## Level: Medium <a class="anchor" id="medium"></a>

__Task 1:__

Go to https://www.kaggle.com/openfoodfacts/world-food-facts/data, download and unzip the data. Assign the TSV file to a dataframe called `food`.


```python
# Enter your code below
# =====================

```

__Task 2:__

Explore the dataset. 
<ol>
    <li/> Print the FIRST 10 rows with columns 5, 6, 7 (hint: iloc)
    <li/> How many observations are there?
    <li/> How many columns are there?
    <li/> How is the dataset indexed?
</ol>


```python
# Enter your code below
# =====================

```

__Task 3:__

Explore the columns.
<ol>
    <li/> Use a for-loop to print the names of all the columns
    <li/> Print the name of the 105th column
    <li/> Print the dtype of the 105th column
</ol>


```python
# Enter your code below
# =====================

```

__Task 4:__

What is the product name of the 19th observation?


```python
# Enter your code below
# =====================

```

__Task 5:__

Explore the realm of peanuts!
<ol>
    <li/> How many entries have a product name of 'Peanuts'?
    <li/> How many unique 'creator' values are associated with  the peanut-entries?
    <li/> Which creator is most frequent, and what is the number of entries from this creator in the peanut entries?
</ol>


```python
# Enter your code below
# =====================

```

---

<img src="../images/challenging.png"
     alt="Hierarchical clustering"
     style="width: 75px; float: right; margin: 20px" />

## Level: Challenging <a class="anchor" id="challenge"></a>

__Task 1:__

Use the following URL to save the Bysykkel JSON dataset to a dataframe called `trips_df`: https://data-legacy.urbansharing.com/oslobysykkel.no/2016/09.json.zip. Familiarize yourself with the dataset.


```python
# Enter your code below
# =====================

```

__Task 2:__

Use the `groupby` and `agg` methods to create a new dataframe, called `'trips_df_agg'` by aggregating the data by `start_station_id`.
   The aggregated dataframe should have the following index and columns:


* Index: `start_station_id`
* Column `'trip_count'`: Count of trips made from this station
* Column `'first_trip'`: First recorded trip made from this station (start timestamp)
* Column `'last_trip'`: Last recorded trip made from this station (start timestamp)


```python
# Enter your code below
# =====================

```

__Task 3:__

Now sort the `trips_df_agg` dataframe by your `trip_count` column, in descending order.


```python
# Enter your code below
# =====================

```

__Task 4:__

Going back to `trips_df`:
<ol> 
    <li/> Convert the datatypes of time-based columns to 'datetime64'
    <li/> Create a new column called 'day_of_week', containing the day number of the week for the entry. 
</ol>    

(Hint: Use the `.weekday()` function built into datetime objects)
    


```python
# Enter your code below
# =====================


```

__Task 5:__

Which two days of the week have the highest activity levels? 
<ol>
    <li/>Use a histogram to observe `day_of_week` frequencies.
    <li/>Use a list of weekday names and value_counts() to create a dictionary of type: {'monday': 3255 ... }


```python
# Enter your code below
# =====================


```

___
___
