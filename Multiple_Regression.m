
function [Results_Regression, Residuals] = Multiple_Regression(Data,Predictor_Variables, Response_Variables)

%    Multiple Linear Regression 
%    This function was written by Atif Khan @ University of Rochester NY 
%    Last Modified: 12-July-2018
%    Example of function usage: [Results_Regression, Residuals] = Multiple_Regression(Data,Predictor_Variables, Response_Variables)
%    where:
%    Predictor_Variables: the name of the predictor variable. 
%    Response_Variables: the name of the response variables.  
%    The results are stored as a strucure 'Results_Regression' and the regression residuals are stored in the Residual variable. 

tic  % Start the clock, for performance evualtion. 

    % Multiple linear regression loop with one outcome variable at a time.
    
 for i=1:length(Response_Variables)    % Loops through the number of response variables.  
    % Fit a linear regression model with given response variable and predictors. 
    % Fitted mdoel is stored in the lm object. 
    lm = fitlm(Data,'ResponseVar',Data.Properties.VariableNames{Response_Variables},'PredictorVars',Data.Properties.VariableNames{Predictor_Variables}) % Two predictors {'Age','Sex'} and one response variable.
    % F test for whole model (assumes constant term)
    ssr = lm.SST - lm.SSE;       % Calcuate regression sum of squares 
    nobs = lm.NumObservations;   % No of observatiosn 
    dfr = lm.NumEstimatedCoefficients - 1; 
    dfe = nobs - 1 - dfr;  % Calcaute degrees of freedom
    f = (ssr./dfr) / (lm.SSE/dfe); % Calculate F-statistics.
    p = fcdf(1./f,dfe,dfr); % p-value, upper tail.
    Residuals(i,1:size(Data,2))=table2array(lm.Residuals(:,1));    % Store the residuals 
    Results_Regression(i,1:4)=lm.Coefficients(1,1:4);     % Store the indivisual estimate, statndard error, t-statistics, and p-value for the Intercept.
    Results_Regression(i,5:8)=lm.Coefficients(2,1:4);     % Store the indivisual estimate, statndard error, t-statistics, and p-value for the Age.
    Results_Regression(i,9:12)=lm.Coefficients(3,1:4);    % Store the indivisual estimate, statndard error, t-statistics, and p-value for the Sex.
    Results_Regression(i,13)=array2table(lm.Rsquared.Ordinary); % Store ordinary R square. 
    Results_Regression(i,14)=array2table(lm.Rsquared.Adjusted); % Store adjusted R square. 
    Results_Regression(i,15)=array2table(lm.RMSE); % Store root mean square error. 
    Results_Regression(i,16)=array2table(lm.MSE); % Store mean square error. 
    Results_Regression(i,17)=array2table(f); % Store F-Statistics. 
    Results_Regression(i,18)=array2table(p); % Store the model p-value. 
 end;
 
 toc   % End the clock and display total elapsed time. 

