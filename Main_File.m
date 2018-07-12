%    A Regression-SVM based computational pipeline. 
%    This script was written by Atif Khan @ University of Rochester NY
%    Last Modified: 12-July-2018.

%########### Load and pre-processing the data file. ###########%

filename = 'DataFile.xlsx';  % Get the file name to load. 
Data = xlsread(filename);    % Read the data file. 
Names = Data.Properties.VariableNames; % Get all the variable names in the data file. 

% A column that contains Cotinine levels is converted to binary calss label as following. 
Data.Cotinine(Data.Cotinine<=10)=-1; % Set all the cotinine levels <=10 as being negative class instances. 
Data.Cotinine(Data.Cotinine>10)=1; % Set all the cotinine levels >10 as being postive class instances. 

Label=Data.Cotanine;   % Store class labels for binary calssification 
Names = Data.Properties.VariableNames; % Get all the variable names in the data file. 


%########### Support Vector Machine Classification (SVM) on Raw Data ###########%

% Run a linear SVM classifier on corrected data with linear kernal and  binary classes. 

SVMModel = fitcsvm(Data,Label,'Standardize',true,'KernelFunction','linear');
Training_Loss = (1-resubLoss(SVMModel))*100;    % Get the training loss. 
fprintf(1, '\n');
disp(['Training Accuracy = ' num2str(Training_Loss),'%']);   % Print training accuracy.
CVSVMModel = crossval(SVMModel);              % Cross validate model with 10-fold cross validation.
Cross_Validation_Loss = (1-kfoldLoss(CVSVMModel))*100;   % Calculate the 10-fold CV accuracy. 
disp(['Cross Validation Accuracy = ' num2str(Cross_Validation_Loss),'%']);  % print CV accuracy. 

% Calcualte ROC-AUC for the cross validated model. 
[label,score,cost] = kfoldPredict(CVSVMModel);     % Get the prediction scores and labels for testing instances in each fold. 
[Xsvm_temp,Ysvm_temp,Tsvm_temp,AUCsvm] = perfcurve(Label,score(:,2),1);     % Calcuate AUC and produce the performance curve.
plot(Xsvm_temp,Ysvm_temp)   % Plot ROC-AUC curve
xlabel('False positive rate')  % Label x-axis 
ylabel('True positive rate')   % Label y-axis
title('ROC for Classification by SVM') % Add title to the plot. 
grid on;   % Turn on the plot grids. 

% Pass the corrected data, class labels, and variable names to the SVM_RFE function
% for support vector machine classification with recurssive feature elimination. 

Results_SVM_RFE = SVM_RFE(Residuals, Label, Names)

% One the optimal numebr of features are obtained use the above ROC-AUC
% section of the code to produce the ROC-AUC plots with optimal numebr of features. 


%########### Multiple Linear Regression for Age and Sex Correction ###########%

% Set the predictor variables in the dataset.
Predictor_Variables= {'Age','Sex'} 
Response_Variables= Data(1:end,7:end)  % Set the response variables. In original data file, first six columns represent samples IDs, age, sex, and other infromation. 

% Pass the predictor and response variable data to the multiple linear regression function. 
% Perform regression and store estimates, p-values, and residuals.  
[Results_Regression, Residuals] = Multiple_Regression(Data,Predictor_Variables, Response_Variables); 

%########### Support Vector Machine Classification (SVM) on Age-Sex Adjusted Data ###########%

% Run a linear SVM classifier on age and sex corrected data with linear kernal and  binary classes. 

SVMModel = fitcsvm(Residuals,Label,'Standardize',true,'KernelFunction','linear');
Training_Loss = (1-resubLoss(SVMModel))*100;    % Get the training loss. 
fprintf(1, '\n');
disp(['Training Accuracy = ' num2str(Training_Loss),'%']);   % Print training accuracy.
CVSVMModel = crossval(SVMModel);              % Cross validate model with 10-fold cross validation.
Cross_Validation_Loss = (1-kfoldLoss(CVSVMModel))*100;   % Calculate the 10-fold CV accuracy. 
disp(['Cross Validation Accuracy = ' num2str(Cross_Validation_Loss),'%']);  % print CV accuracy. 

% Calcualte ROC-AUC for the cross validated model. 
[label,score,cost] = kfoldPredict(CVSVMModel);     % Get the prediction scores and labels for testing instances in each fold. 
[Xsvm_temp,Ysvm_temp,Tsvm_temp,AUCsvm] = perfcurve(Label,score(:,2),1);     % Calcuate AUC and produce the performance curve.
plot(Xsvm_temp,Ysvm_temp)   % Plot ROC-AUC curve
xlabel('False positive rate')  % Label x-axis 
ylabel('True positive rate')   % Label y-axis
title('ROC for Classification by SVM') % Add title to the plot. 
grid on;   % Turn on the plot grids. 

% Pass the corrected data, class labels, and variable names to the SVM_RFE function
% for support vector machine classification with recurssive feature elimination. 

Results_SVM_RFE = SVM_RFE(Residuals, Label, Names)

% One the optimal numebr of features are obtained use the above ROC-AUC
% section of the code to produce the ROC-AUC plots with optimal numebr of features. 
