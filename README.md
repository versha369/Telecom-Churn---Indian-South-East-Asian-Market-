# Telecom-Churn---Indian-South-East-Asian-Market-
Telecom Churn - Indian &amp; South East Asian Market 
<h1 style="text-align:center">   
      <font color = purple >
            <span style='font-family:Georgia'>
                Telecom Churn - Indian & South East Asian Market : A Case Study 
            </span>   
        </font>    
</h1>
<h3 style="text-align:right">   
      <font color = gray >
            <span style='font-family:Georgia'>
                By : Kunal & Ranjan & Versha
            </span>   
        </font>    
</h3>


<h3>   
      <font color = darkgreen>
            <span style='font-family:Georgia'>
            Business Problem Overview :
            </span>   
        </font>    
</h3>
<div>
    <span style='font-family:Georgia'>
        In the telecom industry, customers are able to choose from multiple service providers and actively switch from one operator to another. In this highly competitive market, the telecommunications industry experiences an average of 15-25% annual churn rate. Given the fact that it costs 5-10 times more to acquire a new customer than to retain an existing one, <b>customer retention</b> has now become even more important than customer acquisition.<br>
        For many incumbent operators, retaining high profitable customers is the number one business goal.To reduce customer churn, telecom companies need to predict which customers are at high risk of churn.<br>
        In this project, we will analyse customer-level data of a leading telecom firm, build predictive models to identify customers at high risk of churn and identify the main indicators of churn.
    </span>
</div>
<hr>
<h3>   
      <font color = darkgreen>
            <span style='font-family:Georgia'>
            Understanding & Defining Churn :
            </span>   
        </font>    
</h3>
<div>
    <span style='font-family:Georgia'>
        There are two main models of payment in the telecom industry - <b>Postpaid</b> (customers pay a monthly/annual bill after using the services) and <b>Prepaid</b> (customers pay/recharge with a certain amount in advance and then use the services).<br>
        In the postpaid model, when customers want to switch to another operator, they usually inform the existing operator to terminate the services, and we directly know that this is an instance of churn.<br>
        However, in the prepaid model, customers who want to switch to another network can simply stop using the services without any notice, and it is hard to know whether someone has actually churned or is simply not using the services temporarily (e.g. someone may be on a trip abroad for a month or two and then intend to resume using the services again).<br>
        Thus, churn prediction is usually more critical (and non-trivial) for prepaid customers, and the term ‘churn’ should be defined carefully.  Also, prepaid is the most common model in India and southeast Asia, while postpaid is more common in Europe in North America.This project is based on the Indian and Southeast Asian market.
    </span>
</div>
<hr>
<h3>   
      <font color = darkgreen>
            <span style='font-family:Georgia'>
            Definitions of Churn :
            </span>   
        </font>    
</h3>
<div>
    <span style='font-family:Georgia'>
        There are various ways to define churn, such as:<br>
        <ol>
            <li><b><font color = 'blue'>Revenue-based churn:</font></b> Customers who have not utilised any revenue-generating facilities such as mobile internet, outgoing calls, SMS etc. over a given period of time. One could also use aggregate metrics such as ‘customers who have generated less than INR 4 per month in total/average/median revenue’.<br><br>
                The main shortcoming of this definition is that there are customers who only receive calls/SMSes from their wage-earning counterparts, i.e. they don’t generate revenue but use the services. For example, many users in rural areas only receive calls from their wage-earning siblings in urban areas.</li><br>
            <li><b><font color = 'blue'>Usage-based churn:</font></b> Customers who have not done any usage, either incoming or outgoing - in terms of calls, internet etc. over a period of time.<br><br>
                A potential shortcoming of this definition is that when the customer has stopped using the services for a while, it may be too late to take any corrective actions to retain them. For e.g., if you define churn based on a ‘two-months zero usage’ period, predicting churn could be useless since by that time the customer would have already switched to another operator.</li>
        </ol>
        In this project, we will use the usage-based definition to define churn.
    </span>
</div>
<hr>
<h3>   
      <font color = darkgreen>
            <span style='font-family:Georgia'>
            High Value Churn :
            </span>   
        </font>    
</h3>
<div>
    <span style='font-family:Georgia'>
        In the Indian and the southeast Asian market, approximately 80% of revenue comes from the top 20% customers (called high-value customers). Thus, if we can reduce churn of the high-value customers, we will be able to reduce significant revenue leakage.<br>
        In this project, we will define high-value customers based on a certain metric and predict churn only on high-value customers.
    </span>
</div>

<h3>   
      <font color = darkgreen>
            <span style='font-family:Georgia'>
            Understanding the Business Objective & Data :
            </span>   
        </font>    
</h3>
<div>
    <span style='font-family:Georgia'>
        The dataset contains customer-level information for a span of four consecutive months - June, July, August and September. The months are encoded as 6, 7, 8 and 9, respectively. The business objective is to predict the churn in the last (i.e. the ninth) month using the data (features) from the first three months. To do this task well, understanding the typical customer behaviour during churn will be helpful.
    </span>
</div>
<hr>
<h3>   
      <font color = darkgreen>
            <span style='font-family:Georgia'>
            Understanding Customer Behavior during Churn :
            </span>   
        </font>    
</h3>
<div>
    <span style='font-family:Georgia'>
        Customers usually do not decide to switch to another competitor instantly, but rather over a period of time (this is especially applicable to high-value customers). In churn prediction, we assume that there are three phases of customer lifecycle :<br>
        <ul>
            <li><b><font color = 'blue'>The ‘good’ phase: </font></b> In this phase, the customer is happy with the service and behaves as usual.</li>
            <li><b><font color = 'blue'>The ‘action’ phase:</font></b> The customer experience starts to sore in this phase, for e.g. he/she gets a compelling offer from a  competitor, faces unjust charges, becomes unhappy with service quality etc. In this phase, the customer usually shows different behaviour than the ‘good’ months. Also, it is crucial to identify high-churn-risk customers in this phase, since some corrective actions can be taken at this point (such as matching the competitor’s offer/improving the service quality etc.)</li>
            <li><b><font color = 'blue'>The ‘churn’ phase:</font></b> In this phase, the customer is said to have churned. We define churn based on this phase. Also, it is important to note that at the time of prediction (i.e. the action months), this data is not available to us for prediction. Thus, after tagging churn as 1/0 based on this phase, we discard all data corresponding to this phase.</li>
        </ul>
        In this case, since we are working over a four-month window, the first two months are the ‘good’ phase, the third month is the ‘action’ phase, while the fourth month is the ‘churn’ phase.
    </span>
</div>

# Importing Libraries

import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
%matplotlib inline
import time
import warnings
warnings.filterwarnings('ignore') ## Suppress Warnings
from sklearn.metrics import recall_score, accuracy_score, confusion_matrix, f1_score,classification_report
from sklearn.metrics import precision_score, auc, roc_auc_score, roc_curve, precision_recall_curve,plot_roc_curve


# Displaying all Columns without restrictions
pd.set_option('display.max_columns', None)
pd.set_option('display.max_rows', None)
pd.set_option('display.max_colwidth', -1)

## Importing Dataset and checking its head and shape

tel_df = pd.read_csv('telecom_churn_data.csv')

tel_df.head()

tel_df.shape

tel_df.info(verbose= True)


<div class="alert alert-block alert-info">
    <span style='font-family:Georgia'>
        <b>Insight: </b><br>
         Data has 99999 rows and 226 columns. The data types of the columns are float, integer and object type. We have 179 columsn with float values, 35 columns with integer values and 12 columns with Object Values. Some of the columns are actually Date, they need to be converted to Date Type format. We can see there are some null values in the column. Let's inspect the null values first 
    </span>    
</div>

### Initial Statistical Analysis of the Data

tel_df.describe().T

#### There seem to be columns with single value only. These won't contribute to understand anything in the model. Hence identifying and dropping them

single_val_columns = []
for i in tel_df.columns:
    if tel_df[i].nunique()==1:
        single_val_columns.append(i)
    else:
        pass
    
tel_df.drop(single_val_columns, axis = 1, inplace = True)
print('dropped columns: ',single_val_columns)

#### Checking for missing values (null values) ratio in dataset on remaining columns

((tel_df.isnull().sum()/tel_df.shape[0])*100).round(2).sort_values(ascending=False)

#### There are many columns with over 70% null values which shall eventually be dropped/handled.
<div class="alert alert-block alert-warning">
    <span style='font-family:Georgia'>
        <b>Strategy:</b><br>
        <ul>
            <li>There are 40 columns which have more than 70% null values. Ideally we can remove these columns from calculation as imputing values with such high null values will not result in accurate predictor values. But, we need to first analyze the types of columns that have more than 70% null values.</li>
            <li>Some of the columns like night_pack_X or fb_user_X (where X signified the month value 6/7/8/9) have high null values. In these cases, we can consider that the particular customer did not take those packages or they did not opt for social media utilities. For such columns, we can impute null values with 0.</li>
            <li>We also have to calculate the <b> High Value Customer </b> where as High value customers can be considered as those who have <b>recharged with an amount more than or equal to X, where X is the 70th percentile of the average recharge amount in the first two months</b>, which signifies the good phase.</li> 
        </ul>
    </span>
    <br>
    <span style='font-family:Georgia'>
        Thus, we have to first treat the columns related to recharge amount to identify high value customer. <br>
        <ul>
            <li>total_rech_data_6        - 74.85 % missing values </li>
            <li>total_rech_data_7        - 74.43 % missing values </li>
            <li>av_rech_amt_data_6       - 74.85 % missing values </li>
            <li>av_rech_amt_data_7       - 74.43 % missing values </li>
            <li>date_of_last_rech_data_6 - 74.85 % missing values </li>
            <li>date_of_last_rech_data_7 - 74.43 % missing values </li>
        </ul>
    </span>
</div>

The datetime values are being read incorrectly as object

date_cols = tel_df.select_dtypes(include=['object'])

# Converting the columns to datetime format
for i in date_cols.columns:
    tel_df[i]=pd.to_datetime(tel_df[i])
    
tel_df.shape

Rechecking the data types

tel_df.info(verbose=True)

### Handling missing values

Last recharge data seems to be meaningful, then trying to chekc and impute over it

tel_df[['date_of_last_rech_data_6','total_rech_data_6','max_rech_data_6']].head()

#### The nulls/nans in last recharge data means the customer has not recharged which is meaningful. Hence imputing the values with 0. It simply means the customer has not recharged mobile internet in a while.

#### Imputing last_rech_data for all the 4 months i.e. 6,7,8,9

for i in range(len(tel_df)):
    if pd.isnull((tel_df.total_rech_data_6[i]) and (tel_df.max_rech_data_6[i])):
        if pd.isnull(tel_df.date_of_last_rech_data_6[i]):
            tel_df.total_rech_data_6[i]=0
            tel_df.max_rech_data_6[i]=0
            
    if pd.isnull((tel_df.total_rech_data_7[i]) and (tel_df.max_rech_data_7[i])):
        if pd.isnull(tel_df.date_of_last_rech_data_7[i]):
            tel_df.total_rech_data_7[i]=0
            tel_df.max_rech_data_7[i]=0
            
    if pd.isnull((tel_df.total_rech_data_8[i]) and (tel_df.max_rech_data_8[i])):
        if pd.isnull(tel_df.date_of_last_rech_data_8[i]):
            tel_df.total_rech_data_8[i]=0
            tel_df.max_rech_data_8[i]=0
    
    if pd.isnull((tel_df.total_rech_data_9[i]) and (tel_df.max_rech_data_9[i])):
        if pd.isnull(tel_df.date_of_last_rech_data_9[i]):
            tel_df.total_rech_data_9[i]=0
            tel_df.max_rech_data_9[i]=0

### lets check for recharge count of 2g and 3g now

tel_df[['count_rech_2g_6','count_rech_3g_6','total_rech_data_6']].head()

### Given total_rech_data is count_rech_2g + count_rech_3g, it will pop the issue of multicolinearity. Hence dropping count_rech_2g and count_rech_3g for every month.

tel_df.drop(['count_rech_2g_6','count_rech_3g_6','count_rech_2g_7','count_rech_3g_7','count_rech_2g_8','count_rech_3g_8','count_rech_2g_9','count_rech_3g_9'],axis=1, inplace=True)

tel_df.shape

Checking and handling the ARPU columns now

tel_df[['arpu_3g_6','arpu_2g_6','av_rech_amt_data_6']].head()

### Checking correlation between arpu_3g, arpu_2g, avg_rech_amt

for i in range(6,10):
    print("\n\nCorrelation Month: ",i,"\n")
    arpu_3g_i = 'arpu_3g_'+str(i)
    arpu_2g_i = 'arpu_2g_'+str(i)
    av_rech_amt_data_i = 'av_rech_amt_data_'+str(i)
    print(tel_df[[arpu_3g_i,arpu_2g_i,av_rech_amt_data_i]].corr())

### Dropping the columns arpu_2g and arpu_3g as they have very high correlation with av_rech_amt_data as can be seen from above

tel_df.drop(['arpu_3g_6','arpu_2g_6','arpu_3g_7','arpu_2g_7','arpu_3g_8','arpu_2g_8','arpu_3g_9','arpu_2g_9'],axis=1, inplace=True)

tel_df.shape

### Dropping the columns with high no. of missing values which do not look important

tel_df.drop(['fb_user_6','fb_user_7','fb_user_8','fb_user_9',
                  'night_pck_user_6','night_pck_user_7','night_pck_user_8','night_pck_user_9'],
                  axis=1, inplace=True)

tel_df.shape

Handling av_rech_amt_data for every month

tel_df[['av_rech_amt_data_7','max_rech_data_7','total_rech_data_7']].head()

### From the above table it is deduced that the missing values for the av_rech_amt_data for each month can be replaced with 0 if total_rech_data for each month from 6 to 9  respectively is 0 i.e. if the total recharge done is 0 then the average recharge amount shall also be 0

for i in range(len(tel_df)):
    if (pd.isnull(tel_df['av_rech_amt_data_6'][i]) and (tel_df['total_rech_data_6'][i]==0)):
        tel_df['av_rech_amt_data_6'][i]=0
        
    if (pd.isnull(tel_df['av_rech_amt_data_7'][i]) and (tel_df['total_rech_data_7'][i]==0)):
        tel_df['av_rech_amt_data_7'][i]=0
        
    if (pd.isnull(tel_df['av_rech_amt_data_8'][i]) and (tel_df['total_rech_data_8'][i]==0)):
        tel_df['av_rech_amt_data_8'][i]=0
        
    if (pd.isnull(tel_df['av_rech_amt_data_9'][i]) and (tel_df['total_rech_data_9'][i]==0)):
        tel_df['av_rech_amt_data_9'][i]=0

((tel_df.isnull().sum()/tel_df.shape[0])*100).round(0).sort_values(ascending=False)

tel_df.info()

#### There are no values in date_of_last_rech_data to any of the months as total_rech_data and max_rech_data are enough. Also, as their null values are high, we can drop them

tel_df.drop(["date_of_last_rech_data_6","date_of_last_rech_data_7","date_of_last_rech_data_8","date_of_last_rech_data_9"], axis=1, inplace=True)

####  Dropping the date_of_last_rech_data as well

tel_df.drop(["date_of_last_rech_6","date_of_last_rech_7","date_of_last_rech_8","date_of_last_rech_9"], axis=1, inplace=True)

tel_df.shape

####  Since the columns used to determine the High Value Customer is clear of null values, we can filter the overall data and then handle the remaining missing values in each of the other columns

####  Filtering the High Value Customer from Good Phase

# Calculating the total recharge amount done for data alone in months 6,7,8 and 9

tel_df['total_rech_amt_data_6'] = tel_df['av_rech_amt_data_6'] * tel_df['total_rech_data_6']
tel_df['total_rech_amt_data_7'] = tel_df['av_rech_amt_data_7'] * tel_df['total_rech_data_7']

# Calculating the overall recharge amount for the months 6,7,8 and 9

tel_df['overall_rech_amt_6'] = tel_df['total_rech_amt_data_6']+tel_df['total_rech_amt_6']
tel_df['overall_rech_amt_7'] = tel_df['total_rech_amt_data_7']+tel_df['total_rech_amt_7']

# Calculating the average recharge done by customer in months June and July (i.e. 6th and 7th month)
tel_df['avg_rech_amt_6_7'] = (tel_df['overall_rech_amt_6'] + tel_df['overall_rech_amt_7'])/2

# Finding the value of 70th percentage in the overall revenues defining the high value customer criteria for the company
cut_off = tel_df['avg_rech_amt_6_7'].quantile(0.7)
print("The 70th quantile value to determing the High Value customer is: ",cut_off,'\n')

# Filtering the data to get top 30% High Value Customers
tel_df = tel_df[tel_df['avg_rech_amt_6_7']>=cut_off]

tel_df.shape

#### The total number of customers is now limited to 30k who lie under the High value customer criteria. We shall build our model upon these

#### Checking for null values in these

((tel_df.isnull().sum()/tel_df.shape[0])*100).round(0).sort_values(ascending=False)

#### Imputing the rest of the attributes using KNNImputer

num_col = tel_df.select_dtypes(include=['int64','float64']).columns.tolist()

from sklearn.impute import KNNImputer
from sklearn.preprocessing import MinMaxScaler

scaler = MinMaxScaler()

tel_df[num_col]=scaler.fit_transform(tel_df[num_col])

knn = KNNImputer(n_neighbors=3)

tel_df_knn = pd.DataFrame(knn.fit_transform(tel_df[num_col]))
tel_df_knn.columns = tel_df[num_col].columns

tel_df_knn.isnull().sum().sum()

#### The KNN Imputer has replaced all the null values in the numerical column using K-means algorithm successfully

#### Restoring the scaled values used in imputing to the original form for further works

tel_df[num_col] = scaler.inverse_transform(tel_df_knn)

tel_df.head()

#### Rechecking the nulls

((tel_df.isnull().sum()/tel_df.shape[0])*100).round(0).sort_values(ascending=False)

tel_df.isnull().sum().sum()

#### Creating the dependent variable: Churn

churn_col = ['total_ic_mou_9','total_og_mou_9','vol_2g_mb_9','vol_3g_mb_9']
tel_df[churn_col].info()

tel_df['churn'] =0

tel_df['churn'] = np.where(tel_df[churn_col].sum(axis=1)==0,1,0)

tel_df.head()

### Checking churn vs non-churn percentage and visualizing using pie chart

print((tel_df['churn'].value_counts()/len(tel_df))*100)
((tel_df['churn'].value_counts()/len(tel_df))*100).plot(kind="pie")
plt.show()

#### There is definitely a class imbalance as is evident from above. 

#### First lets drop all the 4 columns from which the churn variable is created

churn_phase_cols = [col for col in tel_df.columns if '_9' in col]
tel_df.drop(churn_phase_cols, axis=1, inplace= True)
tel_df.shape

#### We can drop the good phase variables as well as we have already created the derived variables out of it

tel_df.drop(['total_rech_amt_data_6','av_rech_amt_data_6','total_rech_data_6','total_rech_amt_6','total_rech_amt_data_7','av_rech_amt_data_7','total_rech_data_7','total_rech_amt_7'], axis=1, inplace=True)

#### Lets check for collinearity of these independent variables to understand their inter-dependencies before proceding to handle the missing values in other variables

mon_6_cols = [col for col in tel_df.columns if '_6' in col]
mon_7_cols = [col for col in tel_df.columns if '_7' in col]
mon_8_cols = [col for col in tel_df.columns if '_8' in col]

tel_df_corr = tel_df.corr()

plt.figure(figsize=(20,20))
sns.heatmap(tel_df_corr)
plt.show()

tel_df_corr.loc[:,:] = np.tril(tel_df_corr,k=-1)
tel_df_corr = tel_df_corr.stack()
tel_df_corr[(tel_df_corr>0.80) | (tel_df_corr < -0.80)].sort_values(ascending=False)

#### Dropping the columns at the set criteria of 85% as they are highly correlated with other predictors

tel_df.drop(['total_rech_amt_8','isd_og_mou_8','isd_og_mou_7','sachet_2g_8','total_ic_mou_6',
            'total_ic_mou_8','total_ic_mou_7','std_og_t2t_mou_6','std_og_t2t_mou_8','std_og_t2t_mou_7',
            'std_og_t2m_mou_7','std_og_t2m_mou_8'],axis=1,inplace=True)

tel_df.shape

#### Deriving new variables to understand the data

# We have a column called 'aon'. Using it to create a variable 'tenure'

tel_df['tenure'] = (tel_df['aon']/30).round(0)

tel_df.drop('aon',axis=1, inplace=True)

sns.distplot(tel_df['tenure'],bins=30)
plt.show()

tn_range = [0,6,12,24,60,61]
tn_label = [ '0-6 Months', '6-12 Months', '1-2 Yrs', '2-5 Yrs', '5 Yrs and above']
tel_df['tenure_range'] = pd.cut(tel_df['tenure'],tn_range, labels=tn_label)
tel_df['tenure_range'].head()

plt.figure(figsize=[12,7])
sns.barplot(x='tenure_range',y='churn', data=tel_df)
plt.show()

#### Maximum churn rate is in 0-6 Months but decreasing gradually as time passes

#### The average revenue per user in good phase is given by arpu in 6th and 7th Month. Since we have 2 separate values, we can average them out and drop the original ones

tel_df['avg_arpu_6_7'] = (tel_df['arpu_6']+tel_df['arpu_7'])/2
tel_df['avg_arpu_6_7'].head()

tel_df.drop(['arpu_6','arpu_7'],axis=1, inplace=True)
tel_df.shape

sns.distplot(tel_df['avg_arpu_6_7'])
plt.show()

#### Checking for correlation between target variable with other variables in dataset

plt.figure(figsize=(10,50))
heatmap_churn = sns.heatmap(tel_df.corr()[['churn']].sort_values(ascending=False, by='churn'),annot=True, cmap='crest')
heatmap_churn.set_title('Features correlating to churn variable',fontsize=12)


####  Avg Outgoing calls and calls on roaming for 6th and 7th months are positively correlated with Churn while Avg Revenue, no. of recharges for 8th month has negative correlation with churn

tel_df[['total_rech_num_8','arpu_8']].plot.scatter(x='total_rech_num_8',y='arpu_8')
plt.show()

sns.boxplot(x=tel_df.churn, y=tel_df.tenure)
plt.show()

#### Tenured customers do not seem to churn as much as the new ones

#### Plotting churn vs recharge amount

ax = sns.kdeplot(tel_df.max_rech_amt_8[(tel_df["churn"]==0)], color='Red',shade=True)
ax = sns.kdeplot(tel_df.max_rech_amt_8[(tel_df["churn"]==1)],ax=ax, color='Blue', shade=True)
ax.legend(["No-Churn",'Churn'],loc="lower right")
ax.set_ylabel('Density')
ax.set_xlabel('cost based on volume')
ax.set_title('Distribution of Avg Rech Amt vs Churn')
plt.show()

tel_df.shape

#### Categorizing on month 8 column: totalrecharge and count

tel_df['total_rech_data_group_8'] = pd.cut(tel_df['total_rech_data_8'],[-1,0,10,25,100],labels=["No_Recharge","<=10_Recharges","10-25_Recharges",">25_Recharges"])
tel_df['total_rech_num_group_8']=pd.cut(tel_df['total_rech_num_8'],[-1,0,10,25,1000],labels=["No_Recharge","<=10_Recharges","10-25_Recharges",">25_Recharges"])

plt.figure(figsize=[12,5])
sns.countplot(data=tel_df,x='total_rech_data_group_8',hue='churn')
print(tel_df['total_rech_data_group_8'].value_counts())
plt.show()

plt.figure(figsize=[12,5])
sns.countplot(data=tel_df,x="total_rech_num_group_8",hue='churn')
print(tel_df['total_rech_num_group_8'].value_counts())
plt.show()

#### As the number of recharge rate increases, the churn rate clearly decreases

#### Creating a dummy variable for some of the categorical variables and dropping the first one

dummy_df = pd.get_dummies(tel_df[['total_rech_data_group_8','total_rech_num_group_8','tenure_range']], drop_first=True)
dummy_df.head()

df = tel_df[:].copy()

df.drop(['tenure_range','mobile_number','total_rech_data_group_8','total_rech_num_group_8','sep_vbc_3g','tenure'], axis=1, inplace=True)

df.head()

Creatig X Dataset for model building

X= df.drop(['churn'],axis=1)

X.head()

y=df['churn']
y.head()

#### Consolidated comments

# split the dateset into train and test datasets
from sklearn.model_selection import train_test_split

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, train_size=0.7, random_state=1)
print("Dimension of X_train:", X_train.shape)
print("Dimension of X_test:", X_test.shape)

X_train.info(verbose=True)

num_col = X_train.select_dtypes(include = ['int64','float64']).columns.tolist()

# apply scaling on the dataset
from sklearn import preprocessing
from sklearn.preprocessing import MinMaxScaler

scaler = MinMaxScaler()
X_train[num_col] = scaler.fit_transform(X_train[num_col])

X_train.head()

## Data Imbalance Handling

#### Using SMOTE method, we can balance the data w.r.t. churn variable and proceed further

from imblearn.over_sampling import SMOTE
sm = SMOTE(random_state=42)
X_train_sm,y_train_sm = sm.fit_resample(X_train,y_train)


print((y_train_sm.value_counts()/len(y_train_sm))*100)
((y_train_sm.value_counts()/len(y_train_sm))*100).plot(kind="pie")
plt.show()

print("Dimension of X_train_sm Shape:", X_train_sm.shape)
print("Dimension of y_train_sm Shape:", y_train_sm.shape)

## Logistic Regression

Importing necessary libraries for Model creation

import statsmodels.api as sm

#### Logistic regression model

logm1 = sm.GLM(y_train_sm,(sm.add_constant(X_train_sm)), family = sm.families.Binomial())
logm1.fit().summary()

#### Logistic Regression using Feature Selection (RFE method)

from sklearn.linear_model import LogisticRegression
logreg = LogisticRegression()

from sklearn.feature_selection import RFE

# running RFE with 20 variables as output
rfe = RFE(logreg,n_features_to_select=20)             
rfe = rfe.fit(X_train_sm, y_train_sm)

rfe.support_

rfe_columns=X_train_sm.columns[rfe.support_]
print("The selected columns by RFE for modelling are: \n\n",rfe_columns)

list(zip(X_train_sm.columns, rfe.support_, rfe.ranking_))

X_train_SM = sm.add_constant(X_train_sm[rfe_columns])
logm2 = sm.GLM(y_train_sm,X_train_SM, family = sm.families.Binomial())
res = logm2.fit()
res.summary()

# From the p-value of the individual columns, 
    # we can drop the column 'loc_ic_t2t_mou_8' as it has high p-value of 0.80
rfe_columns_1=rfe_columns.drop('loc_ic_t2t_mou_8',1)
print("\nThe new set of edited featured are:\n",rfe_columns_1)

# Training the model with the edited feature list
X_train_SM = sm.add_constant(X_train_sm[rfe_columns_1])
logm2 = sm.GLM(y_train_sm,X_train_SM, family = sm.families.Binomial())
res = logm2.fit()
res.summary()

# From the p-value of the individual columns, 
    # we can drop the column 'loc_ic_t2m_mou_8' as it has high p-value of 0.80
rfe_columns_2=rfe_columns_1.drop('loc_ic_t2m_mou_8',1)
print("\nThe new set of edited featured are:\n",rfe_columns_2)

# Training the model with the edited feature list
X_train_SM = sm.add_constant(X_train_sm[rfe_columns_2])
logm2 = sm.GLM(y_train_sm,X_train_SM, family = sm.families.Binomial())
res = logm2.fit()
res.summary()

# Getting the predicted values on the train set
y_train_sm_pred = res.predict(X_train_SM)
y_train_sm_pred = y_train_sm_pred.values.reshape(-1)
y_train_sm_pred[:10]

#### Creating a dataframe with the actual churn flag and the predicted probabilities

y_train_sm_pred_final = pd.DataFrame({'Converted':y_train_sm.values, 'Converted_prob':y_train_sm_pred})
y_train_sm_pred_final.head()

y_train_sm_pred_final = pd.DataFrame({'Converted':y_train_sm.values, 'Converted_prob':y_train_sm_pred})
y_train_sm_pred_final.head()

#### Creating new column 'churn_pred' with 1 if Churn_Prob > 0.5 else 0

y_train_sm_pred_final['churn_pred'] = y_train_sm_pred_final.Converted_prob.map(lambda x: 1 if x > 0.5 else 0)

# Viewing the prediction results
y_train_sm_pred_final.head()

from sklearn import metrics

# Confusion matrix 
confusion = metrics.confusion_matrix(y_train_sm_pred_final.Converted, y_train_sm_pred_final.churn_pred )
print(confusion)

# Predicted     not_churn    churn
# Actual
# not_churn        15661      3627
# churn            2775       16513  

# Checking the overall accuracy.
print("The overall accuracy of the model is:",metrics.accuracy_score(y_train_sm_pred_final.Converted, y_train_sm_pred_final.churn_pred))

# Check for the VIF values of the feature variables. 
from statsmodels.stats.outliers_influence import variance_inflation_factor

# Create a dataframe that will contain the names of all the feature variables and their respective VIFs
vif = pd.DataFrame()
vif['Features'] = X_train_sm[rfe_columns_2].columns
vif['VIF'] = [variance_inflation_factor(X_train_sm[rfe_columns].values, i) for i in range(X_train_sm[rfe_columns_2].shape[1])]
vif['VIF'] = round(vif['VIF'], 2)
vif = vif.sort_values(by = "VIF", ascending = False)
vif

Type Markdown and LaTeX:  𝛼2

Metrics beyond simply accuracy

TP = confusion[1,1] # true positive 
TN = confusion[0,0] # true negatives
FP = confusion[0,1] # false positives
FN = confusion[1,0] # false negatives

# Let's see the sensitivity of our logistic regression model
print("Sensitivity = ",TP / float(TP+FN))

# Let us calculate specificity
print("Specificity = ",TN / float(TN+FP))

# Calculate false postive rate - predicting churn when customer does not have churned
print("False Positive Rate = ",FP/ float(TN+FP))

# positive predictive value 
print ("Precision = ",TP / float(TP+FP))

# Negative predictive value
print ("True Negative Prediction Rate = ",TN / float(TN+ FN))

 ## Plotting the ROC Curve

# Defining a function to plot the roc curve
def draw_roc( actual, probs ):
    fpr, tpr, thresholds = metrics.roc_curve( actual, probs,
                                              drop_intermediate = False )
    auc_score = metrics.roc_auc_score( actual, probs )
    plt.figure(figsize=(5, 5))
    plt.plot( fpr, tpr, label='ROC curve (area = %0.2f)' % auc_score )
    plt.plot([0, 1], [0, 1], 'k--')
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('False Positive Rate or [1 - True Negative Prediction Rate]')
    plt.ylabel('True Positive Rate')
    plt.title('Receiver operating characteristic example')
    plt.legend(loc="lower right")
    plt.show()

    return None

# Defining the variables to plot the curve
fpr, tpr, thresholds = metrics.roc_curve( y_train_sm_pred_final.Converted, y_train_sm_pred_final.Converted_prob, drop_intermediate = False )

# Plotting the curve for the obtained metrics
draw_roc(y_train_sm_pred_final.Converted, y_train_sm_pred_final.Converted_prob)

# Let's create columns with different probability cutoffs 
numbers = [float(x)/10 for x in range(10)]
for i in numbers:
    y_train_sm_pred_final[i]= y_train_sm_pred_final.Converted_prob.map(lambda x: 1 if x > i else 0)
y_train_sm_pred_final.head()

# Now let's calculate accuracy sensitivity and specificity for various probability cutoffs.
cutoff_df = pd.DataFrame( columns = ['probability','accuracy','sensitivity','specificity'])
from sklearn.metrics import confusion_matrix

# TP = confusion[1,1] # true positive 
# TN = confusion[0,0] # true negatives
# FP = confusion[0,1] # false positives
# FN = confusion[1,0] # false negatives

num = [0.0,0.1,0.2,0.3,0.4,0.5,0.6,0.7,0.8,0.9]
for i in num:
    cm1 = metrics.confusion_matrix(y_train_sm_pred_final.Converted, y_train_sm_pred_final[i] )
    total1=sum(sum(cm1))
    accuracy = (cm1[0,0]+cm1[1,1])/total1
    
    specificity = cm1[0,0]/(cm1[0,0]+cm1[0,1])
    sensitivity = cm1[1,1]/(cm1[1,0]+cm1[1,1])
    cutoff_df.loc[i] =[ i ,accuracy,sensitivity,specificity]
print(cutoff_df)

# plotting accuracy sensitivity and specificity for various probabilities calculated above.
cutoff_df.plot.line(x='probability', y=['accuracy','sensitivity','specificity'])
plt.show()

Initially we selected the optimum point of classification as 0.5.<br><br>From the above graph, we can see the optimum cutoff is slightly higher than 0.5 but lies lower than 0.6. So lets tweek a little more within this range.

# Let's create columns with refined probability cutoffs 
numbers = [0.50,0.51,0.52,0.53,0.54,0.55,0.56,0.57,0.58,0.59]
for i in numbers:
    y_train_sm_pred_final[i]= y_train_sm_pred_final.Converted_prob.map(lambda x: 1 if x > i else 0)
y_train_sm_pred_final.head()

# Now let's calculate accuracy sensitivity and specificity for various probability cutoffs.
cutoff_df = pd.DataFrame( columns = ['probability','accuracy','sensitivity','specificity'])
from sklearn.metrics import confusion_matrix

# TP = confusion[1,1] # true positive 
# TN = confusion[0,0] # true negatives
# FP = confusion[0,1] # false positives
# FN = confusion[1,0] # false negatives

num = [0.50,0.51,0.52,0.53,0.54,0.55,0.56,0.57,0.58,0.59]
for i in num:
    cm1 = metrics.confusion_matrix(y_train_sm_pred_final.Converted, y_train_sm_pred_final[i] )
    total1=sum(sum(cm1))
    accuracy = (cm1[0,0]+cm1[1,1])/total1
    
    specificity = cm1[0,0]/(cm1[0,0]+cm1[0,1])
    sensitivity = cm1[1,1]/(cm1[1,0]+cm1[1,1])
    cutoff_df.loc[i] =[ i ,accuracy,sensitivity,specificity]
print(cutoff_df)

# plotting accuracy sensitivity and specificity for various probabilities calculated above.
cutoff_df.plot.line(x='probability', y=['accuracy','sensitivity','specificity'])
plt.show()

From the above graph we can conclude, the optimal cutoff point in the probability to define the predicted churn variabe converges at 0.54

#### From the curve above, 0.2 is the optimum point to take it as a cutoff probability.

y_train_sm_pred_final['final_churn_pred'] = y_train_sm_pred_final.Converted_prob.map( lambda x: 1 if x > 0.54 else 0)

y_train_sm_pred_final.head()

# Calculating the ovearall accuracy again
print("The overall accuracy of the model now is:",metrics.accuracy_score(y_train_sm_pred_final.Converted, y_train_sm_pred_final.final_churn_pred))

confusion2 = metrics.confusion_matrix(y_train_sm_pred_final.Converted, y_train_sm_pred_final.final_churn_pred )
print(confusion2)

TP2 = confusion2[1,1] # true positive 
TN2 = confusion2[0,0] # true negatives
FP2 = confusion2[0,1] # false positives
FN2 = confusion2[1,0] # false negatives

# Let's see the sensitivity of our logistic regression model
print("Sensitivity = ",TP2 / float(TP2+FN2))

# Let us calculate specificity
print("Specificity = ",TN2 / float(TN2+FP2))

# Calculate false postive rate - predicting churn when customer does not have churned
print("False Positive Rate = ",FP2/ float(TN2+FP2))

# positive predictive value 
print ("Precision = ",TP2 / float(TP2+FP2))

# Negative predictive value
print ("True Negative Prediction Rate = ",TN2 / float(TN2 + FN2))

Precision and recall tradeoff

from sklearn.metrics import precision_recall_curve

p, r, thresholds = precision_recall_curve(y_train_sm_pred_final.Converted, y_train_sm_pred_final.Converted_prob)

# Plotting the curve
plt.plot(thresholds, p[:-1], "g-")
plt.plot(thresholds, r[:-1], "r-")
plt.show()

# Scaling the test data
X_test[num_col] = scaler.transform(X_test[num_col])
X_test.head()

# Feature selection
X_test=X_test[rfe_columns_2]
X_test.head()

# Adding constant to the test model.
X_test_SM = sm.add_constant(X_test)

y_test_pred = res.predict(X_test_SM)
print("\n The first ten probability value of the prediction are:\n",y_test_pred[:10])

y_pred = pd.DataFrame(y_test_pred)
y_pred.head()

y_pred=y_pred.rename(columns = {0:"Conv_prob"})

y_test_df = pd.DataFrame(y_test)
y_test_df.head()

y_pred_final = pd.concat([y_test_df,y_pred],axis=1)
y_pred_final.head()

y_pred_final['test_churn_pred'] = y_pred_final.Conv_prob.map(lambda x: 1 if x>0.54 else 0)
y_pred_final.head()

# Checking the overall accuracy of the predicted set.
metrics.accuracy_score(y_pred_final.churn, y_pred_final.test_churn_pred)

# Metrics Evaluation

# Confusion Matrix
confusion2_test = metrics.confusion_matrix(y_pred_final.churn, y_pred_final.test_churn_pred)
print("Confusion Matrix\n",confusion2_test)

# Calculating model validation parameters
TP3 = confusion2_test[1,1] # true positive 
TN3 = confusion2_test[0,0] # true negatives
FP3 = confusion2_test[0,1] # false positives
FN3 = confusion2_test[1,0] # false negatives

# Let's see the sensitivity of our logistic regression model
print("Sensitivity = ",TP3 / float(TP3+FN3))

# Let us calculate specificity
print("Specificity = ",TN3 / float(TN3+FP3))

# Calculate false postive rate - predicting churn when customer does not have churned
print("False Positive Rate = ",FP3/ float(TN3+FP3))

# positive predictive value 
print ("Precision = ",TP3 / float(TP3+FP3))

# Negative predictive value
print ("True Negative Prediction Rate = ",TN3 / float(TN3+FN3))

## Explaining the results

print("The accuracy of the predicted model is: ",round(metrics.accuracy_score(y_pred_final.churn, y_pred_final.test_churn_pred),2)*100,"%")
print("The sensitivity of the predicted model is: ",round(TP3 / float(TP3+FN3),2)*100,"%")

print("\nAs the model created is based on a sentivity model, i.e. the True positive rate is given more importance as the actual and prediction of churn by a customer\n") 

# ROC curve for the test dataset

# Defining the variables to plot the curve
fpr, tpr, thresholds = metrics.roc_curve(y_pred_final.churn,y_pred_final.Conv_prob, drop_intermediate = False )
# Plotting the curve for the obtained metrics
draw_roc(y_pred_final.churn,y_pred_final.Conv_prob)

### The AUC score for train dataset is 0.90 and the test dataset is 0.87.<br> This model can be considered as a good model.

# Logistic Regression using PCA

# split the dateset into train and test datasets

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, train_size=0.7, random_state=1)
print("Dimension of X_train:", X_train.shape)
print("Dimension of X_test:", X_test.shape)

# apply scaling on the dataset

scaler = MinMaxScaler()
X_train[num_col] = scaler.fit_transform(X_train[num_col])
X_test[num_col] = scaler.transform(X_test[num_col])

# Applying SMOTE technique for data imbalance correction

sm = SMOTE(random_state=42)
X_train_sm,y_train_sm = sm.fit_resample(X_train,y_train)
print("Dimension of X_train_sm Shape:", X_train_sm.shape)
print("Dimension of y_train_sm Shape:", y_train_sm.shape)

X_train_sm.head()

# importing PCA
from sklearn.decomposition import PCA
pca = PCA(random_state=42)

# applying PCA on train data
pca.fit(X_train_sm)

X_train_sm_pca=pca.fit_transform(X_train_sm)
print("Dimension of X_train_sm_pca: ",X_train_sm_pca.shape)

X_test_pca=pca.transform(X_test)
print("Dimension of X_test_pca: ",X_test_pca.shape)

#Viewing the PCA components
pca.components_

 Performing Logistic Regression

from sklearn.linear_model import LogisticRegression
from sklearn import metrics
logreg_pca = LogisticRegression()
logreg_pca.fit(X_train_sm_pca, y_train_sm)

# making the predictions
y_pred = logreg_pca.predict(X_test_pca)

# converting the prediction into a dataframe
y_pred_df = pd.DataFrame(y_pred)
print("Dimension of y_pred_df:", y_pred_df.shape)

from sklearn.metrics import confusion_matrix, accuracy_score

# Checking the Confusion matrix
print("Confusion Matirx for y_test & y_pred\n",confusion_matrix(y_test,y_pred),"\n")

# Checking the Accuracy of the Predicted model.
print("Accuracy of the logistic regression model with PCA: ",accuracy_score(y_test,y_pred))

plt.bar(range(1,len(pca.explained_variance_ratio_)+1),pca.explained_variance_ratio_)
plt.show()

var_cumu = np.cumsum(pca.explained_variance_ratio_)

# Making a scree plot
fig = plt.figure(figsize=[12,7])
plt.plot(var_cumu)
plt.xlabel('no of principal components')
plt.ylabel('explained variance - cumulative')
plt.show()

np.cumsum(np.round(pca.explained_variance_ratio_, decimals=3)*100)

90% of the data can be explained with 30 PCA components

Fitting the dataset with the 30 explainable components

pca_30 = PCA(n_components=35)

train_pca_30 = pca_30.fit_transform(X_train_sm)
print("Dimension for Train dataset using PCA: ", train_pca_30.shape)

test_pca_30 = pca_30.transform(X_test)
print("Dimension for Test dataset using PCA: ", test_pca_30.shape)

# Function to generate model evaluation metrics and graphs
def classification_algo_metrics(y_actual, y_pred):
    print("Classification report:\n", classification_report(y_actual,y_pred))
    
    accuracy = round(accuracy_score(y_actual, y_pred),4)
    precision = round(precision_score(y_actual, y_pred),4)
    recall = round(recall_score(y_actual, y_pred),4)
    f1 = round(f1_score(y_actual, y_pred),4)
    conf_matrix = confusion_matrix(y_actual, y_pred) # confusion matrix
    model_roc_auc = round(roc_auc_score(y_actual, y_pred),4) # roc_auc_score
    
    print("Accuracy Score   : ", accuracy)
    print("Precision Score  : ", precision)
    print("Recall Score     : ", recall) 
    print("F1 Score         : ", f1)  
    print("Area under curve : ", model_roc_auc,"\n")
     
    # Confusion Matrix
    cm = metrics.confusion_matrix( y_actual, y_pred, [0,1] )
    sns.heatmap(cm, annot=True, fmt='.0f', cmap="PuBu",
    xticklabels = ["Not Churned", "Churned"] ,
    yticklabels = ["Not Churned", "Churned"] )
    plt.ylabel('True labels')
    plt.xlabel('Predicted labels')
    plt.show()  
    return (accuracy, precision, recall, f1, model_roc_auc)

logreg_pca_30 = LogisticRegression()
logreg_pca_30.fit(train_pca_30, y_train_sm)

# making the predictions
y_pred_30 = logreg_pca_30.predict(test_pca_30)

# converting the prediction into a dataframe
y_pred_df_30 = pd.DataFrame(y_pred_30)
print("Dimension of y_pred_df_30: ", y_pred_df_30.shape)

# Checking the Confusion matrix
print("Confusion Matirx for y_test & y_pred\n",confusion_matrix(y_test,y_pred_30),"\n")

# Checking the Accuracy of the Predicted model.
print("Accuracy of the logistic regression model with PCA: ",accuracy_score(y_test,y_pred_30))

accuracy, precision, recall, f1, model_roc_auc = classification_algo_metrics(y_test,y_pred_30)

### Random forest - With SMOTE input 

from sklearn.ensemble import RandomForestClassifier
clf_rf1 = RandomForestClassifier( random_state=0)
clf_rf1.fit(X_train_sm,y_train_sm)
y_rf_pred = clf_rf1.predict(X_test)
print("Confusion Matirx for y_test & y_pred\n",confusion_matrix(y_test,y_rf_pred),"\n")

# Checking the Accuracy of the Predicted model.
print("Accuracy of the logistic regression model with PCA: ",accuracy_score(y_test,y_rf_pred))

accuracy, precision, recall, f1, model_roc_auc = classification_algo_metrics(y_test,y_rf_pred)

feature_imp = pd.DataFrame(sorted(zip(clf_rf1.feature_importances_,X.columns)), columns=['Value','Feature'])
feature_imp_top10=feature_imp.sort_values(by="Value", ascending=False).head(10)
plt.figure(figsize=(15, 10))
sns.barplot(x="Value", y="Feature", data=feature_imp_top10.sort_values(by="Value", ascending=False))
plt.title('Random Forest Features')
plt.tight_layout()
plt.show()

## Random Forest with Grid Search

# Now let's do hyperparametertuning for RandomForest and try to find best parameters to improve the precision and recall score : 
from sklearn.model_selection import train_test_split,GridSearchCV,KFold,cross_val_score
params = { 'n_estimators':[500,1000],
         'max_features': ['auto','sqrt'] ,
         'max_depth' : [10,20] , 
         'min_samples_leaf':[50,100],
         'min_samples_split':[100,150]
        }

# param_grid = { 
#     'n_estimators': [200, 500],
#     'max_features': ['auto', 'sqrt', 'log2'],
#     'max_depth' : [4,5,6,7,8],
#     'criterion' :['gini', 'entropy']
# }
g_search_rf = GridSearchCV(estimator = RandomForestClassifier(random_state = 100,n_jobs = -1), scoring = 'recall', cv=3,
                                                          param_grid = params)
g_search_rf.fit(X_train_sm,y_train_sm)
print("Random Forest Best Score : " ,g_search_rf.best_score_)
print("Random Forest Best Params : " ,g_search_rf.best_params_)

#### Below code of gridsearch is taking time so dont run.
#### below model is created with best paramter obtained from grid search

from sklearn.ensemble import RandomForestClassifier
clf_rf_o = RandomForestClassifier( max_depth= 20, max_features= 'auto', min_samples_leaf= 50, min_samples_split= 100, n_estimators= 1000) ###grid search parameter
clf_rf_o.fit(X_train_sm,y_train_sm)
y_rf_pred_o = clf_rf_o.predict(X_test)
print("Confusion Matirx for y_test & y_pred\n",confusion_matrix(y_test,y_rf_pred_o),"\n")

# Checking the Accuracy of the Predicted model.
print("Accuracy of the logistic regression model with PCA: ",accuracy_score(y_test,y_rf_pred_o))
accuracy, precision, recall, f1, model_roc_auc = classification_algo_metrics(y_test,y_rf_pred_o)

feature_imp = pd.DataFrame(sorted(zip(clf_rf1.feature_importances_,X.columns)), columns=['Value','Feature'])
feature_imp_top10=feature_imp.sort_values(by="Value", ascending=False).head(10)
plt.figure(figsize=(15, 10))
sns.barplot(x="Value", y="Feature", data=feature_imp_top10.sort_values(by="Value", ascending=False))
plt.title('Random Forest Features')
plt.tight_layout()
plt.show()

### lightgbm - With Smote and Gridsearch

#### Below code of gridsearch is taking time so dont run.
#### below model is created with best paramter obtained from grid search

# from lightgbm import LGBMClassifier
# params_lgb ={'num_leaves': [50,70,100], 
#         'max_depth' :[10,20,30],
#         'learning_rate': [0.1,0.5,1],
#         'n_estimators':[600],
#         'min_child_samples': [20,50], 
#         'subsample': [0.1,0.3,0.5,1], 
#         'colsample_bytree': [0.1,0.5,1]
#        }

# lgb = LGBMClassifier(objective = 'binary', n_jobs = -1, random_state = 100)
# g_search_lgb = GridSearchCV(estimator = lgb, param_grid = params_lgb, scoring='recall',cv=3)

# g_search_lgb.fit(X_train_sm,y_train_sm)

# print("LightGBM Best Score : " ,g_search_lgb.best_score_)
# print("LightGBM Best Params : " ,g_search_lgb.best_params_)


from lightgbm import LGBMClassifier
model_lgbm = LGBMClassifier(boosting_type='gbdt',
                            n_estimators=600,
                            learning_rate=0.1, 
                            max_depth=10, 
                            min_child_samples=20,
                            num_leaves=100, 
                            objective='binary', 
                            random_state=100,
                            subsample=.1,
                            colsample_bytree=1,
                            n_jobs=-1,
                            silent=True)
model_lgbm.fit(X_train_sm,y_train_sm)
# Checking the performance of the train dataset
y_pred_lgbm = model_lgbm.predict(X_test)
print("Evaluation on training data set: \n")
print("Confusion Matirx for y_test & y_pred\n",confusion_matrix(y_test,y_pred_lgbm),"\n")

# Checking the Accuracy of the Predicted model.
print("Accuracy of the logistic regression model with PCA: ",accuracy_score(y_test,y_pred_lgbm))
accuracy, precision, recall, f1, model_roc_auc = classification_algo_metrics(y_test,y_pred_lgbm)

### Feature Importance

feature_imp = pd.DataFrame(sorted(zip(model_lgbm.feature_importances_,X.columns)), columns=['Value','Feature'])
feature_imp_top10=feature_imp.sort_values(by="Value", ascending=False).head(10)
plt.figure(figsize=(15, 10))
sns.barplot(x="Value", y="Feature", data=feature_imp_top10.sort_values(by="Value", ascending=False))
plt.title('LightGBM Features')
plt.tight_layout()
plt.show()

## Important Attributes

#### .


<div>
    <span style='font-family:Georgia'>
    Based on top 20 features , 14 out of 20 features are from action phase, where as rest 6 features are from good phase. We can propose following business decisision which will help retain the high value customer who may be planning to leave : <br>
        <ul>
            <li><b><font color = 'blue'> Incoming Calls : </font></b> Total incoming minutes, local & std incoming minutes are in top 10 features. It shows that more incoming calls have high impact on retaining the customer. Incoming calls (local & std) should be made free for all customer </li>
            <li><b><font color = 'blue'> Total Recharge Number : </font></b> Total Recharge Number (total_rech_num) is 4th highest coefficient. It shows that customers with high frequency of recharge numbers tends to stay. The company should launch small value top-up option for calls. Many times customer may hesitate to top up (prepaid option) with higher value amount. For these customer retention, small value top-up will be ideal. The similar logic is applicable for Total Recharge Number for data (total_rech_data) which has the 6th largest coefficient. Similar small top-up values for data might be useful. Also a combination of calling & data which is cost efficient might attract customers to stay with the telecom company </li>
            <li><b><font color = 'blue'> Outgoing Calls : </font></b> Local, STD and Total outgoing call minutes are among the top 20 feature. It shows that higher the outgoing calls, the better chances of the customer to stay with the telecom company. To ensure that, telecom company should launch various discounted offer on outgoing calls. Lower price/minute or free outgoing calls to same telecom company provided connections will also benefit in retaining customer group. </li>
            <li><b><font color = 'blue'> Roaming : </font></b> Incoming and Outgoing calls during roaming are critical factor as these are part of top 20 coefficients. Incoming call roaming charges should be made free within the country. STD outgoing call should be same and shouldn't attract extra cost if the roaming option is within the country. For outside country incoming and outgoing roaming as well as data connection, company should launch special country wise roaming packages valid for 7 days, 15 days, 1 month etc. These benefits will attract customer to stay with the telecom company for longer periof</li>
            <li><b><font color = 'blue'> Total Amount Recharge : </font></b> Total Amount Recharge on calls and data also have high coefficent value. Telecom company should launch discounted price for higher /longer terms (days/month etc) recharge options on calls and data. This may ensure that the customer is ready to available long terms plans which also indicates that the customer may tend to stay longer. </li>
            <li> Introduce more services to the existing customer or informing them and generating more average revenue would lessen their chances to churn out. </li>
        </ul>
        In general, when the above features have a decline trend from the good phase to the action phase, then the customer care executives can reach them out and try to understand if there is any issues. This step would involve an additional smaller cost but this would help to prevent the high value customer to churn out, who generates average revenue per month atleast more than Rs. 240.75. <br>
        If the goal is to engage and talk to the customers to prevent them from churning, its ok to engage with those who are mistakenly tagged as 'not churned,' ( False Positives) as it does not cause any negative problem. It could potentially make them even happier for the extra attention they are getting.
    </span>
</div>




![](https://github.com/versha369/Telecom-Churn---Indian-South-East-Asian-Market-/blob/main/images.jpeg)




