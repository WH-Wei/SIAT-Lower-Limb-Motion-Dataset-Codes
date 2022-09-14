clc;clear;close all;

examples_folder1_path = 'D:/Files/WHsProjects/sEMG_IR/classification20220824/features_standard/20220901093633';
examples_folder2_path = 'D:/Files/WHsProjects/sEMG_IR/classification20220824/features_standard/20220901093927';

fig1_savepath = 'D:\Files\WHsProjects\KNN_movements_classification20220831.jpg';
fig2_savepath = 'D:\Files\WHsProjects\KNN_movements_ConfusionMatrix20220831.jpg';

%% load data
examples_head_path = fullfile(examples_folder1_path,'Head.mat');
load(examples_head_path)
load(path)

subjects_list_classification = subjects_list;
examples_classification = examples;
num_labels_classification = num_labels;
details_subject_classification = details_subject;

examples_head_path = fullfile(examples_folder2_path,'Head.mat');
load(examples_head_path)
load(path)

subjects_list_classification = [subjects_list_classification, subjects_list];
examples_classification = [examples_classification; examples];
num_labels_classification = [num_labels_classification, num_labels];
details_subject_classification = [details_subject_classification, details_subject];

subjects_num = max(size(subjects_list_classification));
[movements_serial, num_labels_serial, OneHot_labels_serial] = get_serial__labels(movements_label_list, num_label_list, OneHot_label_list);

examples_classification = reshape(examples_classification,size(examples_classification,1),size(examples_classification,2)*size(examples_classification,3));
examples_classification = [examples_classification, double(num_labels_classification)'];

subjects_results ={};
for subject_check = 1:subjects_num
    subject= subjects_list_classification(subject_check);
    subject_name = ['Subject',num2str(subject)];
    
    examples_pre_subject = examples_classification(details_subject_classification == subject,:);
    
    R_examples_sub = find(examples_pre_subject(:,end)==8);
    examples_pre_subject(R_examples_sub,:) = [];

    [trainedClassifier, validationAccuracy, true_labels, predict_labels] = KNN_trainClassifier(examples_pre_subject);

    ConfusionMatrix = confusionmat(true_labels, predict_labels);
    ConfusionMatrix_percentage = ConfusionMatrix_num2percentage(ConfusionMatrix);
    ConfusionMatrix_percentage = roundn(ConfusionMatrix_percentage, -1);

    results_output(subject_name, validationAccuracy, ConfusionMatrix_percentage, movements_serial)
    subject_result = {subject_name, validationAccuracy, true_labels, predict_labels, ConfusionMatrix, ConfusionMatrix_percentage};
    subjects_results{end+1} = subject_result;
end

%% plot
acc_pre_subject = zeros(1,subjects_num);
sub_acc_pre_subject = zeros(subjects_num,length(movements_serial));
true_labels_all = [];
predict_labels_all = [];
for subject_check = 1:subjects_num
    acc_pre_subject(subject_check) = subjects_results{subject_check}{2};
    ConfusionMatrix_percentage = subjects_results{subject_check}{6};
    true_labels_all = [true_labels_all; subjects_results{subject_check}{3}];
    predict_labels_all = [predict_labels_all; subjects_results{subject_check}{4}];
    for movement_check = 1:length(movements_serial)
        sub_acc_pre_subject(subject_check, movement_check) = ConfusionMatrix_percentage(movement_check, movement_check);
    end
end

fig1 = figure('color','w');
set(gcf,'position',[50,50,1200,400])
subplot(1,1,1,'position',[0.12,0.25,0.85,0.63])
hold on
grid on
bar_name = categorical(movements_serial, movements_serial,'Ordinal',true);
ave_sub_acc_pre_subject = mean(sub_acc_pre_subject,1);
bar(bar_name,ave_sub_acc_pre_subject,'FaceColor',[0.8,0.8,0.8],'EdgeColor',[0.5,0.5,0.5],'linewidth',0.01,'basevalue',0.001)
for subject_check = 1:subjects_num
    plot(sub_acc_pre_subject(subject_check,:),'o-','linewidth',1.5)
end
title('Classification Accuracy','fontsize',30)
% xlabel('Movement','FontSize',30)
ylabel('Accuracy(%)','FontSize',30)
set(gca,'FontSize',25)
set(gca,'ytick', 0:10:100,'linewidth',2.0)
saveas(fig1, fig1_savepath)

mean(acc_pre_subject)


ConfusionMatrix_all = confusionmat(true_labels_all, predict_labels_all);
ConfusionMatrix_percentage_all = ConfusionMatrix_num2percentage(ConfusionMatrix_all);
ConfusionMatrix_percentage_all = roundn(ConfusionMatrix_percentage_all, -1);
PlotConfusionMatrix('All Subjects KNN Confusion Matrix', ConfusionMatrix_percentage_all, movements_serial, fig2_savepath)


%% function
function [trainedClassifier, validationAccuracy, response, validationPredictions] = KNN_trainClassifier(trainingData)
% returns a trained classifier and its accuracy. This code recreates the
% classification model trained in Classification Learner app. Use the
% generated code to automate training the same model with new data, or to
% learn how to programmatically train models.
%
%  Input:
%      trainingData: a matrix with the same number of columns and data type
%       as imported into the app.
%
%  Output:
%      trainedClassifier: a struct containing the trained classifier. The
%       struct contains various fields with information about the trained
%       classifier.
%
%      trainedClassifier.predictFcn: a function to make predictions on new
%       data.
%
%      validationAccuracy: a double containing the accuracy in percent. In
%       the app, the History list displays this overall accuracy score for
%       each model.
%
% Use the code to train the model with new data. To retrain your
% classifier, call the function from the command line with your original
% data or new data as the input argument trainingData.
%
% For example, to retrain a classifier trained with the original data set
% T, enter:
%   [trainedClassifier, validationAccuracy] = trainClassifier(T)
%
% To make predictions with the returned 'trainedClassifier' on new data T2,
% use
%   yfit = trainedClassifier.predictFcn(T2)
%
% T2 must be a matrix containing only the predictor columns used for
% training. For details, enter:
%   trainedClassifier.HowToPredict

% Auto-generated by MATLAB on 2022-08-23 11:12:45


% Extract predictors and response
% This code processes the data into the right shape for training the
% model.
% Convert input to table
inputTable = array2table(trainingData, 'VariableNames', {'column_1', 'column_2', 'column_3', 'column_4', 'column_5', 'column_6', 'column_7', 'column_8', 'column_9', 'column_10', 'column_11', 'column_12', 'column_13', 'column_14', 'column_15', 'column_16', 'column_17', 'column_18', 'column_19', 'column_20', 'column_21', 'column_22', 'column_23', 'column_24', 'column_25', 'column_26', 'column_27', 'column_28', 'column_29', 'column_30', 'column_31', 'column_32', 'column_33', 'column_34', 'column_35', 'column_36', 'column_37', 'column_38', 'column_39', 'column_40', 'column_41', 'column_42', 'column_43', 'column_44', 'column_45', 'column_46', 'column_47', 'column_48', 'column_49', 'column_50', 'column_51', 'column_52', 'column_53', 'column_54', 'column_55'});

predictorNames = {'column_1', 'column_2', 'column_3', 'column_4', 'column_5', 'column_6', 'column_7', 'column_8', 'column_9', 'column_10', 'column_11', 'column_12', 'column_13', 'column_14', 'column_15', 'column_16', 'column_17', 'column_18', 'column_19', 'column_20', 'column_21', 'column_22', 'column_23', 'column_24', 'column_25', 'column_26', 'column_27', 'column_28', 'column_29', 'column_30', 'column_31', 'column_32', 'column_33', 'column_34', 'column_35', 'column_36', 'column_37', 'column_38', 'column_39', 'column_40', 'column_41', 'column_42', 'column_43', 'column_44', 'column_45', 'column_46', 'column_47', 'column_48', 'column_49', 'column_50', 'column_51', 'column_52', 'column_53', 'column_54'};
predictors = inputTable(:, predictorNames);
response = inputTable.column_55;
isCategoricalPredictor = [false, false, false, false, false, false, false, false, false, false, false, false, false, false, false, false, false, false, false, false, false, false, false, false, false, false, false, false, false, false, false, false, false, false, false, false, false, false, false, false, false, false, false, false, false, false, false, false, false, false, false, false, false, false];

% Train a classifier
% This code specifies all the classifier options and trains the classifier.
classificationKNN = fitcknn(...
    predictors, ...
    response, ...
    'Distance', 'Euclidean', ...
    'Exponent', [], ...
    'NumNeighbors', 5, ...
    'DistanceWeight', 'Equal', ...
    'Standardize', false, ...
    'ClassNames', [0; 1; 2; 3; 4; 5; 6; 7; 9; 10; 11; 12]);

% Create the result struct with predict function
predictorExtractionFcn = @(x) array2table(x, 'VariableNames', predictorNames);
knnPredictFcn = @(x) predict(classificationKNN, x);
trainedClassifier.predictFcn = @(x) knnPredictFcn(predictorExtractionFcn(x));

% Add additional fields to the result struct
trainedClassifier.ClassificationKNN = classificationKNN;
trainedClassifier.About = 'This struct is a trained model exported from Classification Learner R2018b.';
trainedClassifier.HowToPredict = sprintf('To make predictions on a new predictor column matrix, X, use: \n  yfit = c.predictFcn(X) \nreplacing ''c'' with the name of the variable that is this struct, e.g. ''trainedModel''. \n \nX must contain exactly 54 columns because this model was trained using 54 predictors. \nX must contain only predictor columns in exactly the same order and format as your training \ndata. Do not include the response column or any columns you did not import into the app. \n \nFor more information, see <a href="matlab:helpview(fullfile(docroot, ''stats'', ''stats.map''), ''appclassification_exportmodeltoworkspace'')">How to predict using an exported model</a>.');

% Extract predictors and response
% This code processes the data into the right shape for training the
% model.
% Convert input to table
inputTable = array2table(trainingData, 'VariableNames', {'column_1', 'column_2', 'column_3', 'column_4', 'column_5', 'column_6', 'column_7', 'column_8', 'column_9', 'column_10', 'column_11', 'column_12', 'column_13', 'column_14', 'column_15', 'column_16', 'column_17', 'column_18', 'column_19', 'column_20', 'column_21', 'column_22', 'column_23', 'column_24', 'column_25', 'column_26', 'column_27', 'column_28', 'column_29', 'column_30', 'column_31', 'column_32', 'column_33', 'column_34', 'column_35', 'column_36', 'column_37', 'column_38', 'column_39', 'column_40', 'column_41', 'column_42', 'column_43', 'column_44', 'column_45', 'column_46', 'column_47', 'column_48', 'column_49', 'column_50', 'column_51', 'column_52', 'column_53', 'column_54', 'column_55'});

predictorNames = {'column_1', 'column_2', 'column_3', 'column_4', 'column_5', 'column_6', 'column_7', 'column_8', 'column_9', 'column_10', 'column_11', 'column_12', 'column_13', 'column_14', 'column_15', 'column_16', 'column_17', 'column_18', 'column_19', 'column_20', 'column_21', 'column_22', 'column_23', 'column_24', 'column_25', 'column_26', 'column_27', 'column_28', 'column_29', 'column_30', 'column_31', 'column_32', 'column_33', 'column_34', 'column_35', 'column_36', 'column_37', 'column_38', 'column_39', 'column_40', 'column_41', 'column_42', 'column_43', 'column_44', 'column_45', 'column_46', 'column_47', 'column_48', 'column_49', 'column_50', 'column_51', 'column_52', 'column_53', 'column_54'};
predictors = inputTable(:, predictorNames);
response = inputTable.column_55;
isCategoricalPredictor = [false, false, false, false, false, false, false, false, false, false, false, false, false, false, false, false, false, false, false, false, false, false, false, false, false, false, false, false, false, false, false, false, false, false, false, false, false, false, false, false, false, false, false, false, false, false, false, false, false, false, false, false, false, false];

% Perform cross-validation
partitionedModel = crossval(trainedClassifier.ClassificationKNN, 'KFold', 5);

% Compute validation predictions
[validationPredictions, validationScores] = kfoldPredict(partitionedModel);

% Compute validation accuracy
validationAccuracy = 1 - kfoldLoss(partitionedModel, 'LossFun', 'ClassifError');
end

function PlotConfusionMatrix(title_text, ConfusionMatrix,labels,fig_savepath)
fig1 = figure('color','w');
set(gcf,'position',[50,50,1200,1200])
% 混淆矩阵主题颜色
% 可通过各种拾色器获得rgb色值
maxcolor = [191,54,12]; % 最大值颜色
mincolor = [255,255,255]; % 最小值颜色

% 绘制坐标轴
m = length(ConfusionMatrix);
imagesc(1:m,1:m,ConfusionMatrix)
title(title_text,'fontsize',30)
xticks(1:m)
xlabel('Predict class','fontsize',30)
xticklabels(labels)
xtickangle(45);
yticks(1:m)
ylabel('Actual class','fontsize',30)
yticklabels(labels)
set(gca,'FontSize',30)

% 构造渐变色
mymap = [linspace(mincolor(1)/255,maxcolor(1)/255,64)',...
         linspace(mincolor(2)/255,maxcolor(2)/255,64)',...
         linspace(mincolor(3)/255,maxcolor(3)/255,64)'];

colormap(mymap)
colorbar()

% 色块填充数字
for i = 1:m
    for j = 1:m
        text(i,j,num2str(ConfusionMatrix(j,i)),...
            'horizontalAlignment','center',...
            'verticalAlignment','middle',...
            'fontname','Times New Roman',...
            'fontsize',25);
    end
end

% 图像坐标轴等宽
ax = gca;
ax.FontName = 'Times New Roman';
set(gca,'box','on','xlim',[0.5,m+0.5],'ylim',[0.5,m+0.5]);
axis square
saveas(fig1,fig_savepath)
end

function ConfusionMatrix_percentage = ConfusionMatrix_num2percentage(ConfusionMatrix)
examples_num = sum(ConfusionMatrix,2);
examples_num = repmat(examples_num,1, size(ConfusionMatrix,1));
ConfusionMatrix_percentage = ConfusionMatrix./ examples_num *100;
end

function [movements, num_labels, OneHot_labels] = get_serial__labels(movements_label_list, num_label_list, OneHot_label_list)
movements = {};
num_labels = {};
OneHot_labels = {};
for label_check = 1:max(size(num_label_list))
    min_sub = find(num_label_list == min(num_label_list));
%     moviement_name = strtrim(string(movements_label_list(min_sub,:)));
    moviement_name = strtrim(movements_label_list(min_sub,:));
    if strcmp(moviement_name,'R') ~= 1
        movements = [movements, moviement_name];
        num_labels = [num_labels, num2str(num_label_list(min_sub))];
        OneHot_labels = [OneHot_labels, num2str(OneHot_label_list(min_sub,:))];
    end
    num_label_list(min_sub) = inf;
end
end

function results_output(subject_name, validationAccuracy, ConfusionMatrix, movements)
Sub_Acc = [];
for mov_check = 1:max(size(ConfusionMatrix))
    Sub_Acc = [Sub_Acc, movements{mov_check}, ':', num2str(ConfusionMatrix(mov_check,mov_check)),'   '];
end
fprintf([subject_name,'   ACC:', num2str(validationAccuracy),  '\n'])
fprintf([Sub_Acc,'\n','\n'])
end