#!/bin/env python
# Author: Linji Wang
# Date of Creation: 04/17/20
# Description: This python script generates summary figures for presenting analysis results for dataset in assignment-10

# inport libraries
import pandas as pd
import numpy as np
import scipy.stats as sts
import matplotlib.pyplot as plt

def ReadData( fileName ):
    """This function takes a filename as input, and returns a dataframe with
    the metrics from the assignment on descriptive statistics and 
    environmental metrics.  Works for both annual and monthly metrics. 
    Date column should be used as the index for the new dataframe.  Function 
    returns the completed DataFrame."""
    # define dataset column names, import dataset, redefine index to date
    colNames = ['agency_cd', 'site_no', 'Date', 'Discharge', 'Quality']
    DataDF = pd.read_csv(fileName, header=1, names=colNames,delimiter=r"\s+",parse_dates=[2], comment='#',na_values=['Eqp'])
    DataDF = DataDF.set_index('Date')
    # Check for negative values and replace them with NaN
    DataDF['Discharge']=DataDF['Discharge'].mask(DataDF['Discharge']<0,np.nan)
    MissingValues = DataDF['Discharge'].isna().sum()
    return( DataDF, MissingValues)

def ClipData( DataDF, startDate, endDate ):
    """This function clips the given time series dataframe to a given range 
    of dates. Function returns the clipped dataframe and and the number of 
    missing values."""
    # clip the dataframe with the given date range
    DataDF=DataDF.loc[startDate:endDate]
    # quantify the number of missing values
    MissingValues = DataDF["Discharge"].isna().sum()
    return( DataDF, MissingValues )

def ReadMetrics( fileName ):
    """This function takes a filename as input, and returns a dataframe with
    the metrics from the assignment on descriptive statistics and 
    environmental metrics.  Works for both annual and monthly metrics. 
    Date column should be used as the index for the new dataframe.  Function 
    returns the completed DataFrame."""
    
    # read data by its filename
    DataDF=pd.read_csv(fileName,header=0,delimiter=',',parse_dates=['Date'],comment='#',index_col=['Date']) 
    return( DataDF )

# the following condition checks whether we are running as a script, in which 
# case run the test code, otherwise functions are being imported so do not.
# put the main routines from your code after this conditional check.

if __name__ == '__main__':

    # define full river names as a dictionary so that abbreviations are not used in figures
    riverName = { "Wildcat": "Wildcat Creek",
                  "Tippe": "Tippecanoe River" }
    
    # Define full file names as a dictionary
    fileName = { "Wildcat": "WildcatCreek_Discharge_03335000_19540601-20200315.txt","Tippe": "TippecanoeRiver_Discharge_03331500_19431001-20200315.txt" }
    Metrics = {"Annual": "Annual_Metrics.csv", "Monthly": "Monthly_Metrics.csv"}
    
    
    ## Create an empty dataframe for storing data
    DataDF={}
    MissingValues={}
        
    # clip data for the 5 desinated water year
    for file in fileName.keys():
        DataDF[file],MissingValues[file] = ReadData(fileName[file])
        DataDF[file],MissingValues[file] = ClipData(DataDF[file],startDate='2014-10-01',endDate='2019-09-30')
        # generate plot
        plt.plot(DataDF[file]['Discharge'],label=riverName[file])
    plt.legend()
    plt.xlabel('Date',fontsize=20)
    plt.ylabel('Discharge (cfs)',fontsize=20)
    plt.title('Daily flow for both streams for the last 5 years of the record',fontsize=20)
    plt.savefig('Daily flow for both streams for the last 5 years of the record.png',dpi=96)
    plt.close()
        
    ## Create an empty directory
    MetricsDF={}
    # import metrics data
    for file in Metrics.keys():
        MetricsDF[file]=ReadMetrics(Metrics[file])

    # generate plots for coefficient variables
    plt.plot(MetricsDF['Annual'].loc[MetricsDF['Annual']['Station']=='Wildcat']['Coeff Var'],'bo')
    plt.plot(MetricsDF['Annual'].loc[MetricsDF['Annual']['Station']=='Tippe']['Coeff Var'],'ro')   
    plt.legend([riverName['Wildcat'],riverName['Tippe']])
    plt.xlabel('Date',fontsize=20)
    ax=plt.gca()
    ax.set_xticklabels(np.arange(1969,2019,10))
    plt.ylabel('Coefficient of Variation',fontsize=20)
    plt.title('Annual Coefficient of Variation',fontsize=20)
    plt.savefig('Annual Coefficient of Variation.png',dpi=96)
    plt.close()
    
    # Generate plot for Tqmean
    plt.plot(MetricsDF['Annual'].loc[MetricsDF['Annual']['Station']=='Wildcat']['TQmean'],'bo')
    plt.plot(MetricsDF['Annual'].loc[MetricsDF['Annual']['Station']=='Tippe']['TQmean'],'ro')   
    plt.legend([riverName['Wildcat'],riverName['Tippe']])
    plt.xlabel('Date',fontsize=20)
    ax=plt.gca()
    ax.set_xticklabels(np.arange(1969,2019,10))
    plt.ylabel('TQmean',fontsize=20)
    plt.title('Annual Time Series of TQmean',fontsize=20)
    plt.savefig('Annual time series of TQmean.png',dpi=96)       
    plt.close()

    # generate plot for R-B index
    plt.plot(MetricsDF['Annual'].loc[MetricsDF['Annual']['Station']=='Wildcat']['R-B Index'],'bo')
    plt.plot(MetricsDF['Annual'].loc[MetricsDF['Annual']['Station']=='Tippe']['R-B Index'],'ro')   
    plt.legend([riverName['Wildcat'],riverName['Tippe']])
    plt.xlabel('Date',fontsize=20)
    ax=plt.gca()
    ax.set_xticklabels(np.arange(1969,2019,10))

    plt.ylabel('R-B Index',fontsize=20)
    plt.title('Annual Time Series of R-B Index',fontsize=20)
   
    plt.savefig('Annual time series of R-B index.png',dpi=96)  
    plt.close()     
    
    
    # inport monthly data
    MonthlyData=ReadMetrics(Metrics['Monthly'])
    MonthlyData=MonthlyData.groupby('Station')
    
    for name, data in MonthlyData:
        columns=['Mean Flow']
        m=[3,4,5,6,7,8,9,10,11,0,1,2]
        index=0
        aveData=pd.DataFrame(0,index=range(1,13),columns=columns)
        # export data for plotTING
        for i in range(12):
            aveData.iloc[index,0]=data['Mean Flow'][m[index]::12].mean()
            index+=1
        # plot average annual monthly flow
        plt.scatter(aveData.index.values,aveData['Mean Flow'].values, label=riverName[name])
    plt.legend()
    plt.xlabel('Month',fontsize=20)
    plt.ylabel('Discharge (cfs)',fontsize=20)
    plt.title('Average Annual Monthly Flow',fontsize=20)
    plt.savefig('	Average annual monthly flow.png',dpi=96)
    plt.close()
                
    
    # import data
    epdata=ReadMetrics(Metrics['Annual'])
    # clean up the dataset
    epdata=epdata.drop(columns=['site_no','Mean Flow','Median','Coeff Var','Skew','TQmean','R-B Index','7Q','3xMedian'])
    epdata=epdata.groupby('Station')
    # Calculate exceedance probability
    for name, data in epdata:
        flow=data.sort_values('Peak Flow',ascending=False)
        ranks1=sts.rankdata(flow['Peak Flow'],method='average')
        ranks2=ranks1[::-1]
        ep=[100*(ranks2[i]/(len(flow)+1)) for i in range(len(flow))]
        # Generate Plot
        plt.plot(ep,flow['Peak Flow'],label=riverName[name])
        # add grid lines to both axes
    plt.grid(which='both')
    plt.legend()
    plt.xlabel('Exceedance Probability (%)',fontsize=20)
    plt.ylabel('Dishcarge (cfs)',fontsize=20)
    plt.xticks(range(0,100,5))
    plt.tight_layout()
    plt.title('Exceedance Probability',fontsize=20)
    plt.savefig('Exceedance Probability.png',dpi=96)
    plt.close() 