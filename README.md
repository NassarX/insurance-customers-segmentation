# Data-Driven Customer Segmentation and Targeting for A2Z Insurance.
Course project of **`Data Mining`**  course - [MDSAA-DS](www.novaims.unl.pt/MDSAA-DS) - Fall 2022

## Details of the Project

### Introduction
>A2Z Insurance (A2Z) is a portuguese long-standing insurance company that
serves a wide array of insurance services: Motor, Household, Health, Life and
Work Compensation. Although A2Z primarily serves portuguese customers, a
significant portion of their customer acquisition comes from their website.
Customers can sign up to A2Z services through their branches, by telephone, or
on the website.
> 
>In 2016, A2Z became one of the largest insurers in Portugal. However, the lack
of a data driven culture in the company ultimately led to poorly maintained
databases over the years. A2Z is trying to make better use of the database it
has regarding its customers. So far, it has simply mass-marketed everything.
All potential and existing customers get the same promotions, and there are no
attempts to identify target markets for cross-selling opportunities. Now, A2Z
wants start differentiating customers, and developing more focused programs.
>
>A2Z provided you an ABT (Analytic Based Table) with data regarding a sample of
10.290 Customers from its active database. These are customers that had at
least one insurance service with the company at the time the dataset was
extracted. Your job is to **segment the database and find the relevant clusters
of customers. To do this, we suggest you segment the customers using different
perspectives and approaches, as well as combine and analyze the results. A2Z
would like to understand the value and demographics of each customer segment,
as well as understand which types of insurance they will be more interested in
buying**.

### Objective
1. Explore the data and identify the variables that should be used to segment customers.
2. Identify customer segments
3. Justify the number of clusters you chose (taking in consideration the business use as well).
4. Explain the clusters found.
5. Suggest business applications for the findings and define general marketing approaches for each cluster.

### Methodology
1. Review the data provided in the ABT to get a better understanding of the characteristics of the customers in the database. This may include looking at demographic information such as age, salary, and area, as well as information about their insurance history and preferences.
2. Identify relevant segmentation variables that can be used to group the customers into different segments. These variables could be based on characteristics such as geographic area, demographic characteristics, psycho-graphic characteristics (such as salary or education), or insurance history characteristics (such as total claims or paid premiums).
3. Use a combination of statistical analysis and visualization techniques to group the customers into segments based on the segmentation variables you have identified. Some common techniques for segmenting customer data include cluster analysis, decision tree analysis, and principal component analysis.
4. Analyze each customer segment to understand their value and demographics, as well as their insurance preferences and interests. This may include looking at factors such as average customer lifetime value, average insurance premium, and insurance coverage needs.
5. Use the insights gained from the analysis of each customer segment to develop targeted marketing programs and cross-selling opportunities. This may include developing tailored promotions and offers, or creating customized insurance packages based on the specific needs and preferences of each customer segment.
6. Monitor the performance of the marketing programs and cross-selling efforts to see how well they are resonating with each customer segment, and make adjustments as needed.

### Dataset Metadata

| Variable 		      | Description                              | Additional Information |
|---------------------|------------------------------------------|------------------------|
| ID 		          | ID 		                                 |                        |
| First Policy 		  | Year of the customer’s first policy      | (1)                    |
| Birthday            | Customer’s Birthday Year                 | (2)                    |
| Education           | Academic Degree                          |                        |
| Salary              | Gross monthly salary (€) 		         |                        |
| Area 		          | Living area 		                     | (3)                    |
| Children 		      | Binary variable (Y=1) 		             |                        |
| CMV 		          | Customer Monetary Value 		         | (4)                    |
| Claims 		      | Claims Rate 		                     | (5)                    |
| Motor 		      | Premiums (€) in LOB: Motor 		         | (6)                    |
| Household 		  | Premiums (€) in LOB: Household 		     | (6)                    |
| Health 		      | Premiums (€) in LOB: Health 		     | (6)                    |
| Life 		          | Premiums (€) in LOB: Life 		         | (6)                    |
| Work Compensation   | Premiums (€) in LOB: Work Compensations  | (6)                    |

### Data Preprocessing

Before building the model, the dataset was pre-processed in the following steps:

- Data Cleansing
  - @TODO
- Feature Engineering
  - @TODO
- Feature Scaling
  - @TODO
- Feature Encoding
  - @TODO 

### Feature Selection
- Feature Selection for Numeric Values
  - @TODO
- Feature Selection for Categorical Values
  - @TODO

### Modeling

@TODO

### Evaluation

@TODO

### Conclusion

@TODO
### Future Work

@TODO
### Requirements

This project requires **Python** and the following Python libraries installed:

- [NumPy](http://www.numpy.org/)
- [Pandas](http://pandas.pydata.org/)
- [scikit-learn](http://scikit-learn.org/stable/)
- [matplotlib](http://matplotlib.org/)
- [seaborn](https://seaborn.pydata.org)

You will also need to have software installed to run and execute a [Jupyter Notebook](http://jupyter.org/install.html).

If you do not have Python installed yet, it is highly recommended that you install the [Anaconda](https://www.anaconda.com/download/) distribution of Python, which already has the above packages and more included. 

### Code

Template code is provided in the `a2z_insurance.ipynb` notebook file. You will also be required to use the included dataset files `datasets` folder to complete your work. While some code has already been implemented to get you started, you will need to implement additional functionality when requested to successfully complete the project.

### Run

1. Clone the repository to your local machine.
2. In a terminal or command window, navigate to the top-level project directory `DM-200175-Project/` 
3. Run one of the following commands to execute the code:

```bash
ipython notebook a2z_insurance.ipynb
```  
or
```bash
jupyter notebook a2z_insurance.ipynb
```
or open with Juoyter Lab
```bash
jupyter lab
```

This will open the Jupyter Notebook software and project file in your browser.
