# Logistic_Regression_Problem

Research Question

Which variables have the highest correlation with customer churn and can help to build a logistical regression model for prediction? Which variables then, will provide the most control over customer churn?

Objectives and Goals

The main objectives for this project are interdependent and threefold. First, those predictor variables with high association to the target variable churn must be determined. Second, a predictive model must be engineered with the corresponding predictor and target variables. Third, real world actionable items for strategic business management must be derived in order to permit useful recommendations to the business. 

Summary of Assumptions

Logistical regression models assume certain aspects of the data being input. One of those assumptions is the classification problem. Classification, in short, is the process of assigning variable X1 a bin in which it may be recalled. There are a few methods for classification: nominal, binary, and Boolean. Another assumption is the data type of the variable being input into the model. 

Appropriate Technique

There are several forms of models, some of those being: linear, multiple, logistic, and polynomial. Logistic is absolutely an appropriate technique for evaluating this data set as it uses relational Newtonian logic to purvey those predictor variables associated with the target. It is appropriate because it applies to the dataset and because it accomplishes the research question. The goal of the project is to determine those predictor variables which are most associated to the target churn. Then to build a model and provide effective recommendations for strategic busines initiatives. Logistic regression accomplishes this task.

Steps to Prepare the Data
1.	Read in the Data and Python Packages

2.	Transform the Data into Binary Classifications

3.	Compile and Fit the Model

4.	Determine Accuracy of the Model

5.	Determine Correlation of Predictor Variables to the Target Variable, Plot Visualization of Correlations

6.	Reduce the Dataset to Only Associated Variables

7.	Plot Correlation to Verify Assocation Within the Reduced Dataset, Visualize the Correlation

8.	Load and Fit the Model with the Reduced Dataset

9.	Engineer a Confusion Matrix

10.	Determine Accuracy of the New Model

### Initial Model

Summary Statistics

Dep. Variable:	Churn	No. Observations:	8000
Model:	MNLogit	Df Residuals:	7953
Method:	MLE	Df Model:	46
Date:	Tue, 24 Nov 2020	Pseudo R-squ.:	0.5766
Time:	16:57:09	Log-Likelihood:	-1947.2
converged:	True	LL-Null:	-4599.0
Covariance Type:	nonrobust	LLR p-value:	0.000
Churn=1	coef	std err	z	P>|z|	[0.025	0.975]
CaseOrder	-6.3828	0.478	-13.351	0.000	-7.320	-5.446
Customer_id	0.0171	0.082	0.209	0.834	-0.143	0.177
Interaction	-3.1357	1.7e+07	-1.85e-07	1.000	-3.33e+07	3.33e+07
UID	-3.1357	1.7e+07	-1.85e-07	1.000	-3.33e+07	3.33e+07
City	-9.519e-10	3.84e+09	-2.48e-19	1.000	-7.53e+09	7.53e+09
State	0.0676	0.084	0.800	0.424	-0.098	0.233
County	-0.0250	0.082	-0.303	0.762	-0.187	0.137
Zip	-1.3417	1.242	-1.080	0.280	-3.776	1.092
Lat	0.6342	1.162	0.546	0.585	-1.643	2.911
Lng	0.8721	0.451	1.935	0.053	-0.011	1.756
Population	-0.0075	0.083	-0.090	0.928	-0.170	0.155
Area	-0.0005	0.086	-0.006	0.995	-0.169	0.168
TimeZone	-0.0340	0.084	-0.405	0.685	-0.199	0.131
Job	0.0184	0.082	0.224	0.823	-0.143	0.180
Children	-0.0165	0.082	-0.202	0.840	-0.177	0.144
Age	-0.0535	0.082	-0.654	0.513	-0.214	0.107
Income	0.0155	0.082	0.190	0.849	-0.145	0.176
Marital	0.0223	0.083	0.267	0.789	-0.141	0.186
Gender	0.2823	0.082	3.432	0.001	0.121	0.444
Outage_sec_perweek	0.0191	0.082	0.233	0.816	-0.141	0.179
Email	-0.0130	0.083	-0.157	0.875	-0.176	0.150
Contacts	0.2119	0.086	2.467	0.014	0.044	0.380
Yearly_equip_failure	-0.0176	0.087	-0.203	0.839	-0.188	0.153
Techie	1.0225	0.108	9.511	0.000	0.812	1.233
Contract	3.0789	0.108	28.597	0.000	2.868	3.290
Port_modem	0.0697	0.082	0.852	0.394	-0.091	0.230
Tablet	-0.1068	0.089	-1.195	0.232	-0.282	0.068
InternetService	0.7675	0.093	8.291	0.000	0.586	0.949
Phone	-0.2832	0.138	-2.047	0.041	-0.554	-0.012
Multiple	1.5328	0.101	15.153	0.000	1.335	1.731
OnlineSecurity	-0.1204	0.085	-1.410	0.158	-0.288	0.047
OnlineBackup	0.7299	0.090	8.075	0.000	0.553	0.907
DeviceProtection	0.4479	0.084	5.335	0.000	0.283	0.613
TechSupport	0.3461	0.086	4.030	0.000	0.178	0.514
StreamingTV	2.5960	0.114	22.724	0.000	2.372	2.820
StreamingMovies	3.1005	0.128	24.209	0.000	2.849	3.352
PaperlessBilling	0.1829	0.083	2.196	0.028	0.020	0.346
PaymentMethod	-0.3703	0.083	-4.457	0.000	-0.533	-0.207
Tenure	-4.4759	0.989	-4.526	0.000	-6.414	-2.538
MonthlyCharge	0.1680	0.139	1.212	0.225	-0.104	0.439
Bandwidth_GB_Year	-0.6767	0.981	-0.690	0.490	-2.599	1.245
item1	-0.1321	0.128	-1.031	0.302	-0.383	0.119
item2	0.0943	0.091	1.035	0.301	-0.084	0.273
item3	-0.0063	0.120	-0.052	0.958	-0.242	0.229
item4	-0.1400	0.116	-1.209	0.227	-0.367	0.087
item5	-0.1051	0.116	-0.906	0.365	-0.332	0.122
item6	-0.0632	0.119	-0.531	0.595	-0.296	0.170
item7	-0.1660	0.087	-1.918	0.055	-0.336	0.004
item8	-0.0458	0.116	-0.396	0.692	-0.273	0.181
Model:	MNLogit	Pseudo R-squared:	0.577
Dependent Variable:	Churn	AIC:	3988.3048
Date:	2020-11-24 16:57	BIC:	4316.7031
No. Observations:	8000	Log-Likelihood:	-1947.2
Df Model:	46	LL-Null:	-4599.0
Df Residuals:	7953	LLR p-value:	0.0000
Converged:	1.0000	Scale:	1.0000
No. Iterations:	8.0000		
Churn = 0	Coef.	Std.Err.	t	P>|t|	[0.025	0.975]
CaseOrder	-6.3828	0.4781	-13.3511	0.0000	-7.3198	-5.4458
Customer_id	0.0171	0.0817	0.2089	0.8345	-0.1431	0.1773
Interaction	-3.1357	16977348.9141	-0.0000	1.0000	-33274995.5603	33274989.2889
UID	-3.1357	16977348.9141	-0.0000	1.0000	-33274995.5603	33274989.2889
City	-0.0000	3839734391.9746	-0.0000	1.0000	-7525741118.4700	7525741118.4700
State	0.0676	0.0845	0.7999	0.4238	-0.0980	0.2331
County	-0.0250	0.0825	-0.3030	0.7619	-0.1867	0.1367
Zip	-1.3417	1.2420	-1.0803	0.2800	-3.7759	1.0925
Lat	0.6342	1.1618	0.5459	0.5851	-1.6428	2.9112
Lng	0.8721	0.4507	1.9348	0.0530	-0.0113	1.7555
Population	-0.0075	0.0830	-0.0901	0.9282	-0.1701	0.1551
Area	-0.0005	0.0859	-0.0061	0.9952	-0.1688	0.1677
TimeZone	-0.0340	0.0840	-0.4054	0.6852	-0.1987	0.1306
Job	0.0184	0.0822	0.2239	0.8228	-0.1428	0.1796
Children	-0.0165	0.0818	-0.2022	0.8398	-0.1770	0.1439
Age	-0.0535	0.0818	-0.6541	0.5131	-0.2139	0.1068
Income	0.0155	0.0817	0.1898	0.8495	-0.1447	0.1757
Marital	0.0223	0.0833	0.2672	0.7893	-0.1410	0.1856
Gender	0.2823	0.0823	3.4321	0.0006	0.1211	0.4435
Outage_sec_perweek	0.0191	0.0818	0.2333	0.8155	-0.1412	0.1793
Email	-0.0130	0.0830	-0.1569	0.8753	-0.1756	0.1496
Contacts	0.2119	0.0859	2.4666	0.0136	0.0435	0.3802
Yearly_equip_failure	-0.0176	0.0870	-0.2026	0.8395	-0.1882	0.1529
Techie	1.0225	0.1075	9.5108	0.0000	0.8118	1.2332
Contract	3.0789	0.1077	28.5975	0.0000	2.8679	3.2899
Port_modem	0.0697	0.0818	0.8519	0.3943	-0.0906	0.2299
Tablet	-0.1068	0.0894	-1.1952	0.2320	-0.2820	0.0683
InternetService	0.7675	0.0926	8.2908	0.0000	0.5860	0.9489
Phone	-0.2832	0.1383	-2.0470	0.0407	-0.5543	-0.0120
Multiple	1.5328	0.1012	15.1532	0.0000	1.3345	1.7310
OnlineSecurity	-0.1204	0.0854	-1.4103	0.1585	-0.2877	0.0469
OnlineBackup	0.7299	0.0904	8.0749	0.0000	0.5527	0.9071
DeviceProtection	0.4479	0.0840	5.3345	0.0000	0.2834	0.6125
TechSupport	0.3461	0.0859	4.0298	0.0001	0.1778	0.5144
StreamingTV	2.5960	0.1142	22.7238	0.0000	2.3721	2.8200
StreamingMovies	3.1005	0.1281	24.2090	0.0000	2.8495	3.3515
PaperlessBilling	0.1829	0.0833	2.1955	0.0281	0.0196	0.3461
PaymentMethod	-0.3703	0.0831	-4.4568	0.0000	-0.5332	-0.2075
Tenure	-4.4759	0.9888	-4.5264	0.0000	-6.4140	-2.5378
MonthlyCharge	0.1680	0.1385	1.2125	0.2253	-0.1035	0.4394
Bandwidth_GB_Year	-0.6767	0.9806	-0.6901	0.4902	-2.5986	1.2452
item1	-0.1321	0.1281	-1.0313	0.3024	-0.3833	0.1190
item2	0.0943	0.0912	1.0350	0.3007	-0.0843	0.2730
item3	-0.0063	0.1202	-0.0523	0.9583	-0.2418	0.2292
item4	-0.1400	0.1158	-1.2088	0.2267	-0.3670	0.0870
item5	-0.1051	0.1159	-0.9064	0.3647	-0.3322	0.1221
item6	-0.0632	0.1189	-0.5311	0.5953	-0.2962	0.1699
item7	-0.1660	0.0865	-1.9181	0.0551	-0.3356	0.0036
item8	-0.0458	0.1158	-0.3958	0.6923	-0.2729	0.1812

Justification of Model Reduction

There were several justifications for the reduction in model size and those reductions are justified. There were forty-nine total predictors, which is a ludicrous amount for how simple the target variable is. This led to the engineering of a correlation matrix in order to determine which predictors were strongly associated with the target which was customer churn. Those variables which had high correlations were chosen for the reduced model. Another item considered was the accuracy of the model using the Accuracy Score test. The original model’s accuracy score was .871. The reduced model was .8695. This slight difference meant the reduced model was practically the same accuracy without the cloud of non-associated variables. The less predictor variables the better to simplify the model’s analysis.

In short, the methods used were correlation and accuracy. The model evaluation metric was the Sklearn package ‘Accuracy Score’ test. This reduction helps in the answering of the research question as it adds to the usability of the model. This allows for a faster, more in depth understanding of the target variable.  

### Reduced Logistic Regression Model

Summary Statistics

Dep. Variable:	Churn	No. Observations:	8000
Model:	MNLogit	Df Residuals:	7990
Method:	MLE	Df Model:	9
Date:	Tue, 24 Nov 2020	Pseudo R-squ.:	0.5142
Time:	16:57:14	Log-Likelihood:	-2234.3
converged:	True	LL-Null:	-4599.0
Covariance Type:	nonrobust	LLR p-value:	0.000
Churn=1	coef	std err	z	P>|z|	[0.025	0.975]
const	-4.2296	0.124	-34.229	0.000	-4.472	-3.987
Tenure	-4.6617	0.889	-5.241	0.000	-6.405	-2.918
Bandwidth_GB_Year	0.1876	0.881	0.213	0.831	-1.539	1.914
Lng	1.0977	0.435	2.521	0.012	0.244	1.951
Zip	-2.3898	1.258	-1.900	0.057	-4.855	0.075
Lat	1.4438	1.184	1.220	0.223	-0.877	3.764
MonthlyCharge	0.9799	0.098	9.992	0.000	0.788	1.172
StreamingMovies	2.2452	0.099	22.641	0.000	2.051	2.440
Contract	2.6224	0.095	27.684	0.000	2.437	2.808
StreamingTV	1.8934	0.092	20.610	0.000	1.713	2.073
Model:	MNLogit	Pseudo R-squared:	0.514
Dependent Variable:	Churn	AIC:	4488.5162
Date:	2020-11-24 16:57	BIC:	4558.3881
No. Observations:	8000	Log-Likelihood:	-2234.3
Df Model:	9	LL-Null:	-4599.0
Df Residuals:	7990	LLR p-value:	0.0000
Converged:	1.0000	Scale:	1.0000
No. Iterations:	8.0000		
Churn = 0	Coef.	Std.Err.	t	P>|t|	[0.025	0.975]
const	-4.2296	0.1236	-34.2286	0.0000	-4.4718	-3.9874
Tenure	-4.6617	0.8895	-5.2409	0.0000	-6.4051	-2.9184
Bandwidth_GB_Year	0.1876	0.8808	0.2129	0.8314	-1.5388	1.9139
Lng	1.0977	0.4354	2.5210	0.0117	0.2443	1.9511
Zip	-2.3898	1.2578	-1.9000	0.0574	-4.8551	0.0755
Lat	1.4438	1.1839	1.2195	0.2226	-0.8766	3.7641
MonthlyCharge	0.9799	0.0981	9.9918	0.0000	0.7877	1.1721
StreamingMovies	2.2452	0.0992	22.6413	0.0000	2.0508	2.4395
Contract	2.6224	0.0947	27.6835	0.0000	2.4367	2.8080
StreamingTV	1.8934	0.0919	20.6100	0.0000	1.7133	2.0734


Logic of the Variable Selection Technique

The maxim by which the predictors were assessed helpful correlation, association, and model evaluation metrics. Models are more reliable when they contain less predictors and those predictors are associated with the target. Model reduction is simply the process of getting rid of predictors which do not pertain to the variable in question. The only predictors selected were those which had high correlation, high accuracy scoring, and non-negative effects on the confusion matrix. The results of these techniques are below in the model evaluation metric section.

Model Evaluation Metrics

1.	Accuracy Score Test

a.	First Model:

.871

b.	Second Model:

.8695

2.	Confusion Matrix

a.	First Model:

[[1339  105]
[ 154  402]]

b.	Second Model:

[[1332  112]
 [ 145  411]]


Output and Calculations

Correlation Matrix

1.	First Model

Calculation … Code Used:

df_corr = df[df.columns[0:100]].corr()['Churn'][:-1].sort_values()
df_corr.sort_values()

Result:
Tenure                 -4.722056e-01
Bandwidth_GB_Year      -4.699397e-01
Lng                    -3.484895e-01
Zip                    -3.449842e-01
Lat                    -3.416919e-01
Phone                  -2.629656e-02
PaymentMethod          -2.044219e-02
item3                  -1.971728e-02
item1                  -1.753668e-02
item7                  -1.490504e-02
County                 -1.395544e-02
OnlineSecurity         -1.353957e-02
Job                    -1.279765e-02
Outage_sec_perweek     -1.002418e-02
Marital                -8.040453e-03
item2                  -7.758320e-03
item6                  -7.074368e-03
item5                  -6.975402e-03
Yearly_equip_failure   -6.444189e-03
UID                    -5.891240e-03
Interaction            -5.891240e-03
Customer_id            -4.706202e-03
TimeZone               -2.956353e-03
Tablet                 -2.778734e-03
item4                  -1.671866e-03
City                    2.801586e-14
State                   2.787014e-04
Population              9.063447e-04
Children                5.043986e-03
CaseOrder               5.891240e-03
PaperlessBilling        7.030185e-03
Port_modem              8.157067e-03
Income                  8.157102e-03
Age                     8.794852e-03
Contacts                1.090031e-02
Area                    1.555441e-02
TechSupport             1.883838e-02
Email                   2.225617e-02
Gender                  2.702074e-02
OnlineBackup            5.050847e-02
DeviceProtection        5.648949e-02
InternetService         5.847173e-02
Techie                  6.672207e-02
Multiple                1.317712e-01
StreamingTV             2.301509e-01
Contract                2.676533e-01
StreamingMovies         2.892619e-01
MonthlyCharge           3.009994e-01
Churn                   1.000000e+00


2.	Reduced Model

Calculation … Code Used:

ddff = df[['Churn','Tenure','Bandwidth_GB_Year','Lng','Zip','Lat','MonthlyCharge','StreamingMovies','Contract','StreamingTV']]
df_corr = ddff[ddff.columns[0:70]].corr()['Churn'][:].sort_values()
df_corr

Result:
Tenure              -0.472206
Bandwidth_GB_Year   -0.469940
Lng                 -0.348490
Zip                 -0.344984
Lat                 -0.341692
StreamingTV          0.230151
Contract             0.267653
StreamingMovies      0.289262
MonthlyCharge        0.300999
Churn                1.000000



Accuracy Score Test

1.	First Model:

Calculation . . . Code Used:
display('Accuracy Score:', metrics.accuracy_score(y_test, predictions)) 

Result:

.871

2.	Second Model:

Calculation . . . Code Used:

display('Accuracy Score:', metrics.accuracy_score(y_test, predictions)) 

Result:

.8695

Confusion Matrix

1.	First Model:

Calculation . . . Code Used:

logit = LogisticRegression(random_state= 0)
logit.fit(x_train, y_train)

y_predicted = logit.predict(x_test)
cm = confusion_matrix(y_test, y_predicted)
print(cm)


Result:

[[1339  105]
[ 154  402]]

2.	Second Model:

Calculation . . . Code Used

logit = LogisticRegression(random_state= 0)
logit.fit(x_train, y_train)

y_predicted = logit.predict(x_test)
cm = confusion_matrix(y_test, y_predicted)
print(cm)

Result:

[[1332  112]
 [ 145  411]]


Regression Equation
Y = (e^ (-4.2296 (Constant)  - -4.6617 (Tenure) + 0.1876 (Bandwidth) + 1.0977 (Longitude) – 2.3898 (Zip Code) + 1.4438 (Latitude) + 0.9799 (Monthly Charge) + 2.2452 (Streaming Movies) + 1.8934 (Streaming TV) + 2.6224 (Contract))    /    ((e^ (-4.2296 (Constant)  - -4.6617 (Tenure) + 0.1876 (Bandwidth) + 1.0977 (Longitude) – 2.3898 (Zip Code) + 1.4438 (Latitude) + 0.9799 (Monthly Charge) + 2.2452 (Streaming Movies) + 1.8934 (Streaming TV) + 2.6224 (Contract) ) + 1)


Interpretation of Coefficients

The coefficients represent a predictors value as the partnering variables equal zero. For example, when bandwidth, longitude, zip code, latitude, monthly charge, and streaming movies/TV are equal to zero; then tenure is equal to the natural log of tenure’s coefficient plus the intercept. By using this equation, it allows for the plugging in of predictor variables to predict the target.

Limitations of the Analysis

There are three main areas that limit this model’s practicality. 

1.	Data Acquisition Inaccuracies
2.	Generalization through Binary Classification of Predictor Variables
3.	Specificity of the Predictors allows for exact customer identification

Aligning with the Research Question

The logistical model which has been engineered and refined answers the research question. It points to those predictors which are most associated with the target variable churn and lead to possible recommendations.


Recommendations

The research question is, “Which variables have the highest association with customer churn and can help to build a logistical regression model for prediction? Which variables then, will provide the most control over customer churn?”

From the analysis and model, it is determinable that the variables most associated with customer churn are: Tenure, Bandwidth, Longitude, Latitude, Monthly Charge, Streaming Movies/TV, and Contract. These variables are listed in the order in which they are correlated with churn. It is thereby determinable that Tenure and Bandwidth are the predictor variables that should be the focus of strategic business management. 

The recommendation is this, to focus attention on those customers with low tenure and bandwidth, for these are the demographics most associated with churn. Then when possible to focus strategic marketing and retention efforts toward all variables throughout the reduced dataset.
